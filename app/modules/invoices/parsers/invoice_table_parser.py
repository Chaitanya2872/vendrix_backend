"""Line-item table extraction from pdfplumber tables.

Design notes (see module docstrings in gst_utils.py / money_utils.py for the
same rationale elsewhere): everything here is driven by *detected* headers,
never fixed column indexes, because vendor layouts vary arbitrarily.
"""
from __future__ import annotations

import re
from decimal import Decimal

from ..dto import ParsedLineItem
from .money_utils import parse_amount

HEADER_ALIASES: dict[str, list[str]] = {
    "description": ["description", "item", "item description", "product", "particulars"],
    "hsn_sac": ["hsn", "sac", "hsn/sac", "hsn code", "hsn sac"],
    "quantity": ["qty", "quantity"],
    "unit": ["unit", "uom"],
    "unit_price": ["rate", "price", "unit price"],
    "discount": ["discount", "disc"],
    "gst_rate": ["gst", "gst %", "tax %", "gst rate"],
    "taxable_value": ["taxable value", "taxable amount"],
    "total_amount": ["amount", "total", "line total", "net amount"],
}

_SUMMARY_ROW_TOKENS = (
    "subtotal", "sub total", "taxable value", "taxable amount", "discount",
    "cgst", "sgst", "igst", "round off", "grand total", "total amount",
    "amount in words", "invoice total", "net payable", "amount payable",
    "total tax",
)

_NUMERIC_FIELDS = {"quantity", "unit_price", "discount", "gst_rate", "taxable_value", "total_amount"}


def normalize_header(raw: str) -> str:
    text = (raw or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def map_headers(header_row: list[str | None]) -> dict[int, str]:
    """Build {column_index: canonical_field_name} from a table's header row."""
    column_mapping: dict[int, str] = {}
    for index, raw_cell in enumerate(header_row):
        normalized = normalize_header(raw_cell or "")
        if not normalized:
            continue
        for canonical, aliases in HEADER_ALIASES.items():
            if canonical in column_mapping.values():
                continue
            if normalized in aliases or any(normalized == alias for alias in aliases):
                column_mapping[index] = canonical
                break
        else:
            # try substring match as a looser fallback (e.g. "item description / hsn")
            for canonical, aliases in HEADER_ALIASES.items():
                if canonical in column_mapping.values():
                    continue
                if any(alias in normalized for alias in aliases):
                    column_mapping[index] = canonical
                    break
    return column_mapping


def is_header_row(row: list[str | None]) -> bool:
    mapped = map_headers(row)
    # A real header row maps at least description + one numeric-ish column.
    return "description" in mapped.values() and len(mapped) >= 2


def is_summary_row(row: list[str | None]) -> bool:
    joined = " ".join(cell or "" for cell in row).lower()
    return any(token in joined for token in _SUMMARY_ROW_TOKENS)


def _row_has_numeric_signal(row_values: dict[str, str]) -> bool:
    for field_name in ("quantity", "unit_price", "total_amount", "taxable_value"):
        if field_name in row_values and parse_amount(row_values[field_name]) is not None:
            return True
    return False


def score_table_as_line_items(rows: list[list[str | None]]) -> int:
    """Score how likely a detected table is *the* line-item table, based on
    how many recognizable invoice-line headers its header row contains."""
    if not rows:
        return 0
    header_candidates = rows[0]
    mapping = map_headers(header_candidates)
    score = len(mapping)
    if "description" in mapping.values():
        score += 3
    if "total_amount" in mapping.values() or "taxable_value" in mapping.values():
        score += 2
    return score


def select_line_item_table(tables: list[list[list[str | None]]]) -> list[list[str | None]] | None:
    """Do not assume the first detected table is the line-item table — score
    every candidate table on the page and pick the best-scoring one."""
    best_table = None
    best_score = 0
    for table in tables:
        score = score_table_as_line_items(table)
        if score > best_score:
            best_score = score
            best_table = table
    return best_table if best_score >= 3 else None


def _cells_to_field_dict(row: list[str | None], column_mapping: dict[int, str]) -> dict[str, str]:
    result = {}
    for index, field_name in column_mapping.items():
        if index < len(row) and row[index] is not None and str(row[index]).strip():
            result[field_name] = str(row[index]).strip()
    return result


def _to_line_item(field_values: dict[str, str], raw_row: list[str]) -> ParsedLineItem:
    def dec(key: str) -> Decimal | None:
        return parse_amount(field_values[key]) if key in field_values else None

    return ParsedLineItem(
        description=field_values.get("description"),
        hsn_sac=field_values.get("hsn_sac"),
        quantity=dec("quantity"),
        unit=field_values.get("unit"),
        unit_price=dec("unit_price"),
        discount=dec("discount"),
        gst_rate=dec("gst_rate"),
        taxable_value=dec("taxable_value"),
        total_amount=dec("total_amount"),
        raw_row=raw_row,
    )


def parse_line_item_tables(tables_per_page: list[list[list[list[str | None]]]]) -> tuple[list[ParsedLineItem], list[str]]:
    """Parse line items across every page of a multi-page invoice.

    `tables_per_page` is a list (one entry per page) of pdfplumber-style
    `extract_tables()` results (a list of tables, each a list of rows, each
    row a list of cell strings/None).

    Repeated header rows on later pages are detected and skipped so items
    keep accumulating across pages instead of resetting.
    """
    warnings: list[str] = []
    line_items: list[ParsedLineItem] = []
    column_mapping: dict[int, str] | None = None

    for page_index, tables in enumerate(tables_per_page):
        table = select_line_item_table(tables)
        if table is None:
            continue

        rows = table
        start_index = 0
        if column_mapping is None or is_header_row(rows[0]):
            column_mapping = map_headers(rows[0])
            start_index = 1
            if not column_mapping:
                warnings.append(f"Could not map line-item table headers on page {page_index + 1}.")
                continue

        pending_item: ParsedLineItem | None = None
        for row in rows[start_index:]:
            if all((cell is None or not str(cell).strip()) for cell in row):
                continue
            if is_header_row(row):
                # repeated header on a later page/table — skip, keep same mapping
                continue
            if is_summary_row(row):
                continue

            field_values = _cells_to_field_dict(row, column_mapping)
            if not field_values.get("description") and not _row_has_numeric_signal(field_values):
                continue

            has_numeric = _row_has_numeric_signal(field_values)
            if not has_numeric and field_values.get("description"):
                # Likely a continuation of the previous row's description —
                # merge conservatively rather than creating a bogus new item.
                if pending_item is not None:
                    pending_item.description = (
                        f"{pending_item.description} {field_values['description']}".strip()
                        if pending_item.description else field_values["description"]
                    )
                    pending_item.raw_row = pending_item.raw_row + [str(c) for c in row if c]
                    continue
                # No prior row to merge into — treat as its own (rare) item.

            item = _to_line_item(field_values, [str(c) for c in row if c is not None])
            line_items.append(item)
            pending_item = item

    if not line_items:
        warnings.append("No invoice line-item table detected; only header-level totals may be available.")

    return line_items, warnings
