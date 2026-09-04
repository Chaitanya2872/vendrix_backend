"""The geometry-aware extraction path, end to end.

Takes a file and produces the same `ParsedInvoiceResult` the deterministic
text parser produces — so everything downstream (persistence, the review UI,
the duplicate check) is unchanged — but arrives at it through positioned
text rather than a flat string.

    file -> pages (native text or preprocessed+OCR'd pixels)
         -> layout (columns, regions, label->value bindings)
         -> tables (line items, ruled or inferred)
         -> scored candidates -> resolved fields
         -> validation -> confidence

The result carries its own evidence: which box on which page each value came
from, and why that candidate beat its rivals. That is what makes the review
screen able to highlight a field's source, and what makes a wrong extraction
diagnosable instead of merely wrong.

**Multi-page.** Header fields are taken from the best-scoring candidate
across all pages rather than from page one, because a two-page invoice
routinely carries its totals on page two. Line items accumulate across pages,
joining tables that continue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from decimal import Decimal
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.modules.invoices.dto import (
    ParsedInvoiceResult,
    ParsedLineItem,
    ParsedParty,
)
from app.modules.invoices.parsers import lexicon
from app.modules.invoices.parsers.field_scoring import Resolution, extract_fields
from app.modules.invoices.parsers.money_utils import parse_amount
from app.modules.invoices.services.confidence_service import (
    ConfidenceReport,
    score_document,
)
from app.modules.invoices.services.validation_service import (
    ValidationReport,
    validate,
)
from app.modules.ocr import image_preprocessing_service, pdf_service
from app.modules.ocr import service as ocr_service
from app.modules.ocr import table_extraction_service
from app.modules.ocr.dto import OcrDocument, OcrPage
from app.modules.ocr.exceptions import OcrError, OcrPageFailed
from app.modules.ocr.geometry import BoundingBox
from app.modules.ocr.layout_service import (
    REGION_TABLE,
    PageLayout,
    analyse_page,
)
from app.utils import file_utils

logger = logging.getLogger(__name__)

PARSER_NAME = "structured"
PARSER_VERSION = "1.0"

# Scored fields that map straight onto a ParsedInvoiceResult attribute of the
# same name. Fields needing interpretation (party attribution, currency) are
# handled separately below.
DIRECT_FIELDS: tuple[str, ...] = (
    "invoice_number", "invoice_date", "due_date",
    "purchase_order_number", "purchase_order_date",
    "subtotal", "taxable_amount", "discount_amount",
    "cgst_amount", "sgst_amount", "igst_amount",
    "tax_amount", "round_off", "total_amount",
    "amount_paid", "amount_due",
    "payment_terms", "place_of_supply",
)

CURRENCY_HINTS: tuple[tuple[str, str], ...] = (
    ("₹", "INR"), ("inr", "INR"), ("rs.", "INR"), ("rs ", "INR"),
    ("$", "USD"), ("usd", "USD"),
    ("€", "EUR"), ("eur", "EUR"),
    ("£", "GBP"), ("gbp", "GBP"),
)


@dataclass
class FieldEvidence:
    """Where a value came from, and why it was chosen."""

    field: str
    value: str
    page_number: int
    box: dict | None
    score: float
    margin: float
    reasons: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "page_number": self.page_number,
            "box": self.box,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "reasons": list(self.reasons),
        }


@dataclass
class StructuredExtraction:
    parsed: ParsedInvoiceResult
    validation: ValidationReport
    confidence: ConfidenceReport
    evidence: list[FieldEvidence] = dataclass_field(default_factory=list)
    ocr_document: OcrDocument | None = None
    layouts: list[PageLayout] = dataclass_field(default_factory=list)
    page_decisions: list[dict] = dataclass_field(default_factory=list)

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_document and self.ocr_document.used_ocr)

    def evidence_dict(self) -> dict[str, dict]:
        return {item.field: item.to_dict() for item in self.evidence}


# --- reading pages ---------------------------------------------------------


def read_pages(
    path: str | Path,
    on_page: Callable[[int, int], None] | None = None,
) -> tuple[OcrDocument, list[dict]]:
    """Turn a file into positioned text, page by page.

    Each page takes the cheapest route that will work: an intact text layer
    is read directly, and only pixels go through preprocessing and OCR.
    """
    path = Path(path)
    extension = path.suffix.lower()
    file_format = file_utils.EXTENSION_TO_FORMAT.get(extension)

    if file_format == file_utils.PDF:
        return _read_pdf(path, on_page)
    if file_format in file_utils.RASTER_FORMATS:
        return _read_raster(path, on_page)
    raise OcrPageFailed(f"Structured extraction does not handle '{extension}' files")


def _read_pdf(path: Path, on_page) -> tuple[OcrDocument, list[dict]]:
    analysis = pdf_service.analyse(path)
    pages: list[OcrPage] = []

    all_ocr_pages = [page for page in analysis.pages if page.needs_ocr]
    limit = settings.ocr_max_pages if settings.ocr_max_pages > 0 else len(all_ocr_pages)
    ocr_pages = all_ocr_pages[:limit]
    selected_ocr_pages = {page.page_number for page in ocr_pages}
    if len(ocr_pages) < len(all_ocr_pages):
        logger.info(
            "structured_extraction.ocr_page_limit_applied pages=%s read=%s",
            len(all_ocr_pages), len(ocr_pages),
        )
    completed = 0

    for page in analysis.pages:
        if not page.needs_ocr:
            pages.append(page.native)
            continue
        if page.page_number not in selected_ocr_pages:
            continue
        prepared = image_preprocessing_service.preprocess(
            page.image,
            # A rendered PDF page has no page boundary inside the frame, so
            # any quadrilateral found is a table border — warping to it
            # destroys the page.
            allow_perspective=False,
        )
        pages.append(
            ocr_service.recognize_page(
                prepared.image,
                page_number=page.page_number,
                preprocessing_applied=prepared.applied,
                rotation_applied=prepared.rotation_applied,
            )
        )
        completed += 1
        if on_page is not None:
            on_page(completed, len(ocr_pages))

    pages.sort(key=lambda item: item.page_number)
    return OcrDocument(pages=pages, filename=path.name), analysis.decisions()


def _read_raster(path: Path, on_page) -> tuple[OcrDocument, list[dict]]:
    """Read a photographed or scanned image, including multi-frame TIFF."""
    import cv2
    import numpy as np

    frames: list = []
    if path.suffix.lower() in {".tif", ".tiff"}:
        # Multi-frame TIFF is how departmental scanners emit a stapled
        # invoice; reading only the first frame silently drops pages.
        from PIL import Image, ImageSequence

        with Image.open(path) as image:
            for frame in ImageSequence.Iterator(image):
                frames.append(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
    else:
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise OcrPageFailed(f"Unable to decode {path.name}")
        frames.append(decoded)

    pages: list[OcrPage] = []
    for index, frame in enumerate(frames, start=1):
        prepared = image_preprocessing_service.preprocess(frame, allow_perspective=True)
        pages.append(
            ocr_service.recognize_page(
                prepared.image,
                page_number=index,
                preprocessing_applied=prepared.applied,
                rotation_applied=prepared.rotation_applied,
            )
        )
        if on_page is not None:
            on_page(index, len(frames))

    decisions = [
        {"page_number": page.page_number, "route": "raster",
         "reason": "raster image has no text layer",
         "preprocessing": page.preprocessing_applied}
        for page in pages
    ]
    return OcrDocument(pages=pages, filename=path.name), decisions


# --- merging across pages --------------------------------------------------


def merge_resolutions(per_page: list[dict[str, Resolution]]) -> dict[str, Resolution]:
    """Keep the best-scoring candidate for each field across every page.

    Not "page one wins": a two-page invoice routinely carries its totals on
    page two, and a page-one preference would take a partial subtotal over
    the real grand total.
    """
    merged: dict[str, Resolution] = {}
    for page_resolutions in per_page:
        for name, resolution in page_resolutions.items():
            existing = merged.get(name)
            if existing is None or resolution.winner.score > existing.winner.score:
                merged[name] = resolution
    return merged


# --- party attribution -----------------------------------------------------


def attribute_parties(
    layouts: list[PageLayout],
    resolutions: dict[str, Resolution],
) -> tuple[ParsedParty, ParsedParty]:
    """Decide which GSTIN belongs to the supplier and which to the customer.

    Geometrically, by proximity to a party label — because that is the only
    evidence that survives when both blocks say nothing but "GSTIN: ...".
    Reading order is not enough: plenty of invoices put the buyer block
    above or left of the seller.
    """
    from app.modules.invoices.parsers.field_scoring import parse_value

    # Names come from the scored resolutions, not from re-walking the pairs.
    # Scoring already weighed every competing reading of "who is the buyer";
    # taking the first label match here instead would quietly override that
    # with a worse answer — and *did*, picking up an invoice number that
    # happened to sit under a `To` label.
    vendor = ParsedParty(name=_resolved_text(resolutions, "vendor_name"))
    customer = ParsedParty(name=_resolved_text(resolutions, "customer_name"))

    vendor_anchors: list[BoundingBox] = []
    customer_anchors: list[BoundingBox] = []
    gstin_pairs: list[tuple[str, BoundingBox]] = []

    for layout in layouts:
        for pair in layout.pairs:
            if lexicon.match_label(pair.label, lexicon.VENDOR_LABELS) is not None:
                vendor_anchors.append(pair.label_box)
            elif lexicon.match_label(pair.label, lexicon.CUSTOMER_LABELS) is not None:
                customer_anchors.append(pair.label_box)

            if lexicon.match_label(pair.label, lexicon.GSTIN_LABELS) is not None:
                value = parse_value("gstin", pair.value)
                if value:
                    gstin_pairs.append((value, pair.value_box))

    for value, box in gstin_pairs:
        vendor_distance = min((box.distance_to(anchor) for anchor in vendor_anchors), default=None)
        customer_distance = min((box.distance_to(anchor) for anchor in customer_anchors), default=None)

        if vendor_distance is not None and (customer_distance is None or vendor_distance < customer_distance):
            if not vendor.gstin:
                vendor.gstin = value
        elif customer_distance is not None:
            if not customer.gstin:
                customer.gstin = value
        elif not vendor.gstin:
            # No party labels at all. The supplier's own GSTIN is the one
            # that appears first on an invoice far more often than not, but
            # this is a guess and the confidence stage should see it as one.
            vendor.gstin = value
        elif not customer.gstin:
            customer.gstin = value

    return vendor, customer


def _resolved_text(resolutions: dict[str, Resolution], name: str) -> str | None:
    resolution = resolutions.get(name)
    if resolution is None:
        return None
    value = str(resolution.winner.parsed_value or "").strip()
    return value or None


def detect_currency(document: OcrDocument) -> str | None:
    text = document.text.lower()
    for hint, code in CURRENCY_HINTS:
        if hint in text:
            return code
    return None


# --- line items ------------------------------------------------------------


def extract_line_items(layouts: list[PageLayout], pages: list[OcrPage]) -> list[ParsedLineItem]:
    """Recover line items from every page's item table."""
    items: list[ParsedLineItem] = []

    for layout, page in zip(layouts, pages):
        table_region = layout.region(REGION_TABLE)
        # Narrowing to the typed table region gives cleaner column
        # boundaries, but no region is typed when the page has no whitespace
        # gaps to block on. Falling back to the whole page is safe because
        # the borderless detector validates its own candidate — a page with
        # no table returns nothing rather than a table of prose.
        region_lines = table_region.lines if table_region else None
        detected = table_extraction_service.extract_tables(page, region_lines=region_lines)
        for table in detected:
            items.extend(_rows_to_items(table))
    return items


def _rows_to_items(table) -> list[ParsedLineItem]:
    mapping = table_extraction_service.map_columns(table)
    if not mapping:
        # Without a header row the columns cannot be identified, and guessing
        # by position is a per-vendor template by another name.
        return []

    items: list[ParsedLineItem] = []
    grid = table.grid()
    start = (table.header_row + 1) if table.header_row is not None else 0

    for row in grid[start:]:
        values: dict[str, str] = {}
        for column_index, canonical in mapping.items():
            if column_index < len(row) and row[column_index].strip():
                values[canonical] = row[column_index].strip()
        if not values.get("description") and not values.get("total_amount"):
            continue
        if _is_summary_row(row):
            continue

        items.append(
            ParsedLineItem(
                description=values.get("description"),
                hsn_sac=values.get("hsn_sac"),
                quantity=_decimal(values.get("quantity")),
                unit=values.get("unit"),
                unit_price=_decimal(values.get("unit_price")),
                discount=_decimal(values.get("discount")),
                gst_rate=_decimal(values.get("gst_rate")),
                taxable_value=_decimal(values.get("taxable_value")),
                total_amount=_decimal(values.get("total_amount")),
                raw_row=list(row),
            )
        )
    return items


def _is_summary_row(row: list[str]) -> bool:
    """Whether a table row is really a totals row.

    Vendors put the subtotal and taxes inside the item table's ruling, so
    the grid contains them; taking them as line items would double-count the
    invoice.
    """
    joined = " ".join(cell for cell in row if cell).lower()
    return any(token in joined for token in (
        "subtotal", "sub total", "total", "cgst", "sgst", "igst", "round off",
        "taxable value", "amount in words", "discount",
    ))


def _decimal(value: str | None) -> Decimal | None:
    return parse_amount(value) if value else None


# --- assembly --------------------------------------------------------------


def build_result(
    resolutions: dict[str, Resolution],
    layouts: list[PageLayout],
    document: OcrDocument,
    line_items: list[ParsedLineItem],
) -> tuple[ParsedInvoiceResult, list[FieldEvidence]]:
    result = ParsedInvoiceResult(
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        used_ocr=document.used_ocr,
    )

    evidence: list[FieldEvidence] = []
    for name in DIRECT_FIELDS:
        resolution = resolutions.get(name)
        if resolution is None:
            continue
        setattr(result, name, resolution.winner.parsed_value)
        evidence.append(_evidence_for(name, resolution))

    result.vendor, result.customer = attribute_parties(layouts, resolutions)
    result.currency = detect_currency(document)
    result.line_items = line_items

    # Derive the tax total from its components when the invoice states the
    # parts but not the sum, which many do.
    if result.tax_amount is None:
        components = [value for value in
                      (result.cgst_amount, result.sgst_amount, result.igst_amount)
                      if value is not None]
        if components:
            result.tax_amount = sum(components)

    return result, evidence


def _evidence_for(name: str, resolution: Resolution) -> FieldEvidence:
    winner = resolution.winner
    return FieldEvidence(
        field=name,
        value=winner.raw_value,
        page_number=winner.page_number,
        box=winner.box.to_dict() if winner.box else None,
        score=winner.score,
        margin=resolution.margin,
        reasons=list(winner.reasons),
    )


def extract(
    path: str | Path,
    on_page: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> StructuredExtraction:
    """Full geometry-aware extraction for one file."""
    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage("reading")
    document, decisions = read_pages(path, on_page=on_page)

    stage("layout")
    layouts = [analyse_page(page) for page in document.pages]

    stage("tables")
    line_items = extract_line_items(layouts, document.pages)

    stage("fields")
    per_page = [extract_fields(layout) for layout in layouts]
    resolutions = merge_resolutions(per_page)

    parsed, evidence = build_result(resolutions, layouts, document, line_items)

    stage("validating")
    validation = validate(parsed)
    confidence = score_document(
        resolutions,
        validation,
        line_confidences=[],
    )

    parsed.parsing_confidence = confidence.document_confidence
    for finding in validation.warnings:
        parsed.add_warning(finding.message)
    for finding in validation.errors:
        parsed.add_validation_error(finding.message)

    logger.info(
        "structured_extraction.completed file=%s pages=%d ocr=%s fields=%d "
        "line_items=%d confidence=%.3f",
        Path(path).name, document.page_count, document.used_ocr,
        len(resolutions), len(line_items), confidence.document_confidence,
    )

    return StructuredExtraction(
        parsed=parsed,
        validation=validation,
        confidence=confidence,
        evidence=evidence,
        ocr_document=document,
        layouts=layouts,
        page_decisions=decisions,
    )


def is_supported(path: str | Path) -> bool:
    """Whether this path can take the structured route.

    Office formats keep exact cell structure of their own and are better
    served by the existing text parser, so they are deliberately excluded
    rather than forced through a geometry pipeline built for pixels.
    """
    extension = Path(path).suffix.lower()
    file_format = file_utils.EXTENSION_TO_FORMAT.get(extension)
    return file_format == file_utils.PDF or file_format in file_utils.RASTER_FORMATS
