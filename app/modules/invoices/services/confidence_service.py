"""Confidence: how much should anyone trust this extraction?

The number that decides whether an invoice is posted automatically or put in
front of a human. Getting it wrong is expensive in both directions — too high
and wrong data reaches the ledger, too low and the automation saves nobody
any work — so it is built from evidence that is actually independent rather
than from one blended guess.

Four inputs per field, each answering a different question:

  **extraction score** — how confidently was this value chosen over its
                         rivals? (field_scoring)
  **margin**           — how close was the runner-up? A field won by a hair
                         is worth a human's glance even if it scored well.
  **OCR confidence**   — did the recogniser actually read the characters?
  **validation**       — does the value survive the cross-checks it takes
                         part in?

Document confidence is deliberately *not* the mean of its fields. A document
whose total is wrong is not 90% right because nine other fields were fine —
the total is what the invoice is for. Fields are weighted by consequence, and
any validation error caps the document outright.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field

from app.modules.invoices.services.validation_service import (
    SEVERITY_ERROR,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# What each field is worth to the document score. Weighted by consequence:
# posting an invoice with the wrong total is a financial error, with the
# wrong payment terms an inconvenience.
FIELD_WEIGHTS: dict[str, float] = {
    "total_amount": 5.0,
    "invoice_number": 4.0,
    "invoice_date": 3.0,
    "vendor_gstin": 3.0,
    "vendor_name": 2.5,
    "taxable_amount": 2.0,
    "subtotal": 2.0,
    "tax_amount": 2.0,
    "cgst_amount": 1.5,
    "sgst_amount": 1.5,
    "igst_amount": 1.5,
    "customer_gstin": 1.5,
    "customer_name": 1.0,
    "due_date": 1.0,
    "purchase_order_number": 1.0,
    "round_off": 0.5,
    "place_of_supply": 0.5,
    "payment_terms": 0.5,
}
DEFAULT_FIELD_WEIGHT = 0.5

# A field the document needs but which was not found at all scores zero and
# still counts against the total — otherwise "found nothing" would score the
# same as "found everything", since both have no bad fields.
REQUIRED_FIELDS: tuple[str, ...] = (
    "invoice_number", "invoice_date", "total_amount", "vendor_name",
)

# A validation error caps document confidence here regardless of how well
# every individual field scored. Arithmetic that does not reconcile means
# something is wrong that per-field confidence provably cannot see.
ERROR_CAP = 0.50
# Each warning costs this much, to a floor.
WARNING_PENALTY = 0.05
MAX_WARNING_PENALTY = 0.25

# A field implicated in a validation finding is penalised directly, so the
# reviewer's attention lands on the field at fault rather than on all of them.
IMPLICATED_FIELD_PENALTY = 0.45

# Below this, a document goes to review rather than being accepted.
REVIEW_THRESHOLD = 0.80
# Below this, an individual field is flagged for the reviewer's attention
# even when the document as a whole passes.
FIELD_ATTENTION_THRESHOLD = 0.70


@dataclass
class FieldConfidence:
    field: str
    confidence: float
    extraction_score: float = 0.0
    margin: float = 1.0
    ocr_confidence: float = 1.0
    penalised_by: list[str] = dataclass_field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return self.confidence < FIELD_ATTENTION_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "confidence": round(self.confidence, 4),
            "extraction_score": round(self.extraction_score, 4),
            "margin": round(self.margin, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "needs_attention": self.needs_attention,
            "penalised_by": list(self.penalised_by),
        }


@dataclass
class ConfidenceReport:
    document_confidence: float
    fields: dict[str, FieldConfidence] = dataclass_field(default_factory=dict)
    line_item_confidence: float | None = None
    capped_by_errors: bool = False
    missing_required: list[str] = dataclass_field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.document_confidence < REVIEW_THRESHOLD or self.capped_by_errors

    def fields_needing_attention(self) -> list[str]:
        return sorted(
            name for name, entry in self.fields.items() if entry.needs_attention
        )

    def to_dict(self) -> dict:
        return {
            "document_confidence": round(self.document_confidence, 4),
            "needs_review": self.needs_review,
            "capped_by_errors": self.capped_by_errors,
            "line_item_confidence": (
                round(self.line_item_confidence, 4)
                if self.line_item_confidence is not None else None
            ),
            "missing_required": list(self.missing_required),
            "fields_needing_attention": self.fields_needing_attention(),
            "fields": {name: entry.to_dict() for name, entry in self.fields.items()},
        }


def score_field(
    field_name: str,
    extraction_score: float,
    margin: float,
    ocr_confidence: float,
    implicated_by: list[str] | None = None,
) -> FieldConfidence:
    """Confidence for one field.

    The margin term is what a single-score system cannot express. A value
    that scored 0.9 with a 0.89 runner-up was very nearly a different number,
    and no amount of confidence in the winner changes that — so a thin margin
    pulls the confidence down even when everything else is strong.
    """
    implicated = list(implicated_by or [])

    # Margin contributes but cannot dominate: a genuinely uncontested field
    # (margin 1.0) should not be scored above its own extraction quality.
    margin_factor = 0.75 + 0.25 * min(max(margin, 0.0), 1.0)
    confidence = extraction_score * margin_factor * max(0.2, min(1.0, ocr_confidence))

    if implicated:
        confidence *= IMPLICATED_FIELD_PENALTY

    return FieldConfidence(
        field=field_name,
        confidence=round(min(1.0, max(0.0, confidence)), 4),
        extraction_score=extraction_score,
        margin=margin,
        ocr_confidence=ocr_confidence,
        penalised_by=implicated,
    )


def score_line_items(line_confidences: list[float]) -> float | None:
    """Confidence across the line-item table.

    The *worst* line, not the mean. A table is only as trustworthy as its
    least trustworthy row: nineteen perfect rows and one misread amount is a
    wrong invoice, and averaging hides exactly that.
    """
    if not line_confidences:
        return None
    return round(min(line_confidences), 4)


def score_document(
    resolutions: dict,
    validation: ValidationReport,
    ocr_confidence_by_field: dict[str, float] | None = None,
    line_confidences: list[float] | None = None,
) -> ConfidenceReport:
    """Combine per-field confidence and validation into a document score.

    `resolutions` is the output of `field_scoring.resolve` — each entry
    carries the winning candidate's score and the margin over its runner-up.
    """
    ocr_confidence_by_field = ocr_confidence_by_field or {}
    implicated = validation.fields_with_findings()

    # Which finding implicated each field, so a reviewer sees the reason
    # rather than an unexplained low number.
    reasons_by_field: dict[str, list[str]] = {}
    for finding in validation.findings:
        for name in finding.fields:
            reasons_by_field.setdefault(name, []).append(finding.code)

    fields: dict[str, FieldConfidence] = {}
    for name, resolution in resolutions.items():
        winner = resolution.winner
        # `quality`, not `score`: the score ranks candidates against each
        # other and includes proximity and region priors, which say how
        # likely this reading was to *win* rather than how likely it is to be
        # *right*. Confidence built on the ranking score charges an ordinary
        # right-aligned summary amount for sitting where summary amounts sit.
        fields[name] = score_field(
            field_name=name,
            extraction_score=winner.quality or winner.score,
            margin=resolution.margin,
            ocr_confidence=ocr_confidence_by_field.get(name, winner.ocr_confidence),
            implicated_by=reasons_by_field.get(name) if name in implicated else None,
        )

    missing_required = [name for name in REQUIRED_FIELDS if name not in fields]

    # Weighted mean over every weighted field, with missing required fields
    # contributing zero. Without that term, an extraction that found nothing
    # would score the same as one that found everything.
    total_weight = 0.0
    weighted_sum = 0.0
    for name, entry in fields.items():
        weight = FIELD_WEIGHTS.get(name, DEFAULT_FIELD_WEIGHT)
        weighted_sum += entry.confidence * weight
        total_weight += weight
    for name in missing_required:
        total_weight += FIELD_WEIGHTS.get(name, DEFAULT_FIELD_WEIGHT)

    document = weighted_sum / total_weight if total_weight else 0.0

    warning_penalty = min(MAX_WARNING_PENALTY, WARNING_PENALTY * len(validation.warnings))
    document = max(0.0, document - warning_penalty)

    capped = bool(validation.errors)
    if capped:
        # Not a penalty but a ceiling: an invoice whose arithmetic does not
        # reconcile must not be auto-accepted however confident its fields
        # were, because the disagreement proves at least one of them is wrong.
        document = min(document, ERROR_CAP)

    line_confidence = score_line_items(line_confidences or [])

    report = ConfidenceReport(
        document_confidence=round(min(1.0, max(0.0, document)), 4),
        fields=fields,
        line_item_confidence=line_confidence,
        capped_by_errors=capped,
        missing_required=missing_required,
    )
    logger.info(
        "confidence.scored document=%.3f review=%s errors=%d warnings=%d attention=%s",
        report.document_confidence, report.needs_review,
        len(validation.errors), len(validation.warnings),
        report.fields_needing_attention(),
    )
    return report
