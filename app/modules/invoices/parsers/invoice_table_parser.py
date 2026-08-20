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
        if "rate" in normalized and "incl" in normalized and "tax" in normalized:
            # Prefer the exclusive-tax Rate column when both are present;
            # quantity times that rate is the taxable line amount.
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
    mapping = max((map_headers(row) for row in rows), key=len, default={})
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
        header_index = next((index for index, row in enumerate(rows) if is_header_row(row)), None)
        if column_mapping is None or header_index is not None:
            if header_index is None:
                continue
            column_mapping = map_headers(rows[header_index])
            start_index = header_index + 1
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
            first_line_row = [str(cell).splitlines()[0].strip() if cell else "" for cell in row]
            if is_summary_row(first_line_row):
                continue
            field_values = _cells_to_field_dict(row, column_mapping)
            # Tally may merge tax summary lines into the last item row. The
            # first visual line is the item cell; the remaining lines belong
            # to CGST/SGST/round-off rows below it.
            field_values = {name: value.splitlines()[0].strip()
                            for name, value in field_values.items()}
            if is_summary_row(list(field_values.values())) or \
                    field_values.get("description", "").strip().lower() == "total":
                continue
            if not field_values.get("description") and not any(
                field_values.get(name) for name in ("hsn_sac", "quantity", "unit_price")
            ):
                continue
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


_OCR_ITEM_DETAIL = re.compile(
    r"^(?P<hsn>\d{4,8})\s+(?P<quantity>[\d,.]+)\s+(?P<unit>[A-Za-z]+)\s+"
    r"(?P<inclusive_rate>[\d,.]+)\s+(?P<unit_price>[\d,.]+)\s+[A-Za-z]+$"
)
_OCR_INLINE_ITEM_DETAIL = re.compile(
    r"^(?P<hsn>\d{4,8})\s+(?P<quantity>[\d,.]+)\s+(?P<unit>[A-Za-z]+)\s+"
    r"(?P<unit_price>[\d,.]+)\s+[A-Za-z]+(?:\s+(?P<discount>[\d,.]+)\s*%)?\s+"
    r"(?P<amount>[\d,.]+)$"
)
_OCR_SIMPLE_ITEM_DETAIL = re.compile(
    r"^(?P<hsn>\d{4,8})\s+(?P<quantity>[\d,.]+)\s+(?P<unit>[A-Za-z]+)\s+"
    r"(?P<unit_price>[\d,.]+)\s+[A-Za-z]+$"
)
_OCR_ITEM_DESCRIPTION = re.compile(r"^\s*\d+\s+(.+\S)\s*$")


def parse_line_items_from_ocr_text(text: str) -> list[ParsedLineItem]:
    """Recover rows when OCR flattens a ruled table into visual line order.

    Tally commonly emits amount, numeric columns, then serial/description on
    three adjacent lines. The arithmetic check is mandatory, so unrelated
    numbers in the header cannot become a fabricated line item.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        header = next(index for index, line in enumerate(lines)
                      if re.search(r"descr.?ption of goods", line.lower()))
    except StopIteration:
        return []

    items: list[ParsedLineItem] = []
    for index in range(header + 1, len(lines)):
        if lines[index].lower() in {"cgst", "sgst", "igst", "total", "round off"}:
            if items:
                break
        detail = _OCR_INLINE_ITEM_DETAIL.fullmatch(lines[index])
        legacy_detail = _OCR_ITEM_DETAIL.fullmatch(lines[index]) if detail is None else None
        simple_detail = _OCR_SIMPLE_ITEM_DETAIL.fullmatch(lines[index]) if detail is None and legacy_detail is None else None
        matched = detail or legacy_detail or simple_detail
        if matched is None:
            continue

        quantity_text = matched.group("quantity")
        quantity = (Decimal(quantity_text.replace(",", "."))
                    if re.fullmatch(r"\d+,\d{2}", quantity_text)
                    else parse_amount(quantity_text))
        unit_price = parse_amount(matched.group("unit_price"))
        amount = parse_amount(detail.group("amount")) if detail else None
        if quantity is None or unit_price is None:
            continue
        discount = parse_amount(detail.group("discount")) if detail and detail.group("discount") else Decimal("0")
        expected = quantity * unit_price * (Decimal("1") - discount / Decimal("100"))
        if amount is None:
            nearby = lines[max(header + 1, index - 1):index] + lines[index + 1:index + 4]
            candidates = [parsed for value in nearby if (parsed := parse_amount(value)) is not None]
            amount = next((candidate for candidate in candidates
                           if abs(expected - candidate) <= Decimal("1.00")), None)
        if amount is None:
            continue
        if abs(expected - amount) > Decimal("1.00"):
            continue

        description = None
        raw_row = [lines[index - 1], lines[index]]
        for candidate in lines[max(header + 1, index - 3):index] + lines[index + 1:index + 4]:
            match = _OCR_ITEM_DESCRIPTION.fullmatch(candidate)
            if match:
                description = match.group(1).strip()
                raw_row.append(candidate)
                break
        if not description:
            continue

        items.append(ParsedLineItem(
            description=description,
            hsn_sac=matched.group("hsn"),
            quantity=quantity,
            unit=matched.group("unit"),
            unit_price=unit_price,
            discount=discount or None,
            taxable_value=amount,
            total_amount=amount,
            raw_row=raw_row,
        ))
    return items or _parse_column_stream_items(lines, header)


def _parse_column_stream_items(lines: list[str], header: int) -> list[ParsedLineItem]:
    """Recover OCR tables emitted as independent vertical column streams."""
    anchors: list[tuple[int, str]] = []
    expected_serial = 1
    for index in range(header + 1, len(lines)):
        match = re.match(r"^\s*(\d+)\s+([A-Za-z].+)$", lines[index])
        if (match and int(match.group(1)) == expected_serial
                and match.group(2).strip().lower() not in {"nos", "no", "pcs", "pc"}):
            anchors.append((index, match.group(2).strip()))
            expected_serial += 1
        if anchors and lines[index].lower() == "total":
            break
    if not anchors:
        return []

    items: list[ParsedLineItem] = []
    for index, description in anchors:
        window = range(max(header + 1, index - 5), min(len(lines), index + 6))

        hsn_candidates = []
        quantity_candidates = []
        money_candidates = []
        for candidate_index in window:
            line = lines[candidate_index]
            hsn = re.fullmatch(r"\d{6,8}", line)
            if hsn:
                hsn_candidates.append((abs(candidate_index - index), line))
            # In this column-stream form quantities are integral (`15 NOS`);
            # decimal values followed by NOS are the rate-per column.
            quantity_match = re.search(r"(?<![\d.,])(\d+)\s+NOS\b", line, re.IGNORECASE)
            if quantity_match:
                raw_quantity = quantity_match.group(1)
                quantity = (Decimal(raw_quantity.replace(",", "."))
                            if re.fullmatch(r"\d+,\d{2}", raw_quantity)
                            else parse_amount(raw_quantity))
                if quantity is not None:
                    quantity_candidates.append((abs(candidate_index - index), quantity))
            for raw_money in re.findall(r"\d[\d,]*\.\d{2}", line):
                value = parse_amount(raw_money)
                if value is not None:
                    money_candidates.append((abs(candidate_index - index), candidate_index, value))

        if not quantity_candidates or len(money_candidates) < 2:
            continue
        quantity = min(quantity_candidates, key=lambda candidate: candidate[0])[1]

        reconciled = []
        for rate_distance, rate_index, rate in money_candidates:
            for amount_distance, amount_index, amount in money_candidates:
                if rate_index == amount_index:
                    continue
                if abs(quantity * rate - amount) <= Decimal("1.00"):
                    reconciled.append((rate_distance + amount_distance, rate, amount))
        if not reconciled:
            continue
        _, unit_price, amount = min(reconciled, key=lambda candidate: candidate[0])
        hsn_sac = min(hsn_candidates, default=(0, None), key=lambda candidate: candidate[0])[1]
        items.append(ParsedLineItem(
            description=description,
            hsn_sac=hsn_sac,
            quantity=quantity,
            unit="NOS",
            unit_price=unit_price,
            taxable_value=amount,
            total_amount=amount,
            raw_row=[lines[candidate] for candidate in window],
        ))
    return items
