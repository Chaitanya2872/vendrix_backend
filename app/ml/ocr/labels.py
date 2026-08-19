"""Field taxonomy shared by every stage of the OCR field-extraction model.

The model is a *line* classifier: each text line produced by the extraction
pass (native text or OCR) is assigned exactly one of these labels. The labels
map onto attributes of `ParsedInvoiceResult`, so a prediction can be folded
into the same DTO the deterministic parsers already produce — the model is an
extra source of field values, not a parallel data model.

`kind` drives value parsing (see extract.py): the classifier decides *which*
field a line carries, and the kind decides how to turn that line's text into
a typed value. Splitting those two jobs keeps the learned part small — it
never has to learn how to parse a rupee amount, only how to recognise the
line that holds one.
"""
from __future__ import annotations

from dataclasses import dataclass

OTHER = "OTHER"


@dataclass(frozen=True)
class FieldSpec:
    label: str
    kind: str           # money | date | gstin | identifier | text | none
    target: str | None  # attribute path on ParsedInvoiceResult, None if not a field


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("INVOICE_NUMBER", "identifier", "invoice_number"),
    FieldSpec("INVOICE_DATE", "date", "invoice_date"),
    FieldSpec("DUE_DATE", "date", "due_date"),
    FieldSpec("VENDOR_NAME", "text", "vendor.name"),
    FieldSpec("VENDOR_GSTIN", "gstin", "vendor.gstin"),
    FieldSpec("CUSTOMER_NAME", "text", "customer.name"),
    FieldSpec("CUSTOMER_GSTIN", "gstin", "customer.gstin"),
    FieldSpec("SUBTOTAL", "money", "subtotal"),
    FieldSpec("CGST_AMOUNT", "money", "cgst_amount"),
    FieldSpec("SGST_AMOUNT", "money", "sgst_amount"),
    FieldSpec("IGST_AMOUNT", "money", "igst_amount"),
    FieldSpec("TAX_AMOUNT", "money", "tax_amount"),
    FieldSpec("TOTAL_AMOUNT", "money", "total_amount"),
    # Line items are recognised but not reduced to a single document-level
    # value; the existing table parser owns that structure.
    FieldSpec("LINE_ITEM", "none", None),
    FieldSpec(OTHER, "none", None),
)

SPEC_BY_LABEL: dict[str, FieldSpec] = {spec.label: spec for spec in FIELD_SPECS}
ALL_LABELS: tuple[str, ...] = tuple(spec.label for spec in FIELD_SPECS)

# Labels that reduce to one value per document. Everything else is either
# structural (LINE_ITEM) or background (OTHER).
SINGLE_VALUE_LABELS: tuple[str, ...] = tuple(
    spec.label for spec in FIELD_SPECS if spec.target is not None
)

# Formats the model is trained and evaluated on. "Different formats" is the
# whole point of the exercise: the same invoice content reaches the parser
# through very different pipelines (native text vs OCR of a degraded scan),
# and the model has to hold up across all of them.
FORMATS: tuple[str, ...] = (
    "pdf_native",
    "pdf_scan",
    "jpg",
    "png",
    "webp",
    "docx",
    "xlsx",
)

# Formats whose text can only be recovered by OCR. Used by the evaluation
# report to separate "clean text" accuracy from "OCR text" accuracy.
OCR_FORMATS: frozenset[str] = frozenset({"pdf_scan", "jpg", "png", "webp"})
