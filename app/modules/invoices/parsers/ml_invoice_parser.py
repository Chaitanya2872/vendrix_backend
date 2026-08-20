"""Deterministic parsing first, the trained model second — for the gaps only.

The two approaches fail in opposite places, so this composes them rather than
choosing between them:

  the regex parser  is exact when a label it knows appears verbatim, and
                    recovers structure (line-item tables, currency, payment
                    terms) the line classifier does not model at all. It goes
                    blank the moment OCR renders "Amount Payable" as
                    "AmountPayabIe", or the invoice uses a wording nobody put
                    in the alias list.
  the model         tolerates that damage and unfamiliar wordings, and gives a
                    per-field probability, but it only knows the header fields
                    it was trained on.

So the deterministic result stands, and the model is asked only about fields
that came back None. A model value never overwrites one the regex parser
found: where both are confident they agree anyway, and where they disagree the
exact match is the better bet.

If no trained artifact is present, `can_parse` returns False and dispatch falls
through to GenericInvoiceParser unchanged. An untrained checkout behaves
exactly as it did before this parser existed.
"""
from __future__ import annotations

import logging

from ..dto import ParsedInvoiceResult
from .base_invoice_parser import BaseInvoiceParser
from .generic_invoice_parser import GenericInvoiceParser
from .text_extraction import ExtractedDocument

logger = logging.getLogger(__name__)


def _assign(result: ParsedInvoiceResult, target: str, value: object) -> None:
    """Set a possibly-nested attribute ('vendor.name') on the result."""
    owner = result
    *path, attribute = target.split(".")
    for step in path:
        owner = getattr(owner, step)
    setattr(owner, attribute, value)


def _current(result: ParsedInvoiceResult, target: str) -> object:
    owner = result
    *path, attribute = target.split(".")
    for step in path:
        owner = getattr(owner, step)
    return getattr(owner, attribute)


class MlAssistedInvoiceParser(BaseInvoiceParser):
    name = "ml_assisted"
    version = "1.0"

    def __init__(self) -> None:
        self._deterministic = GenericInvoiceParser()

    def can_parse(self, text: str) -> bool:
        from ....ml.ocr import predict

        return bool(text and text.strip()) and predict.available()

    def parse(self, extracted: ExtractedDocument) -> ParsedInvoiceResult:
        from ....core.config import settings
        from ....ml.ocr import predict
        from ....ml.ocr.labels import SPEC_BY_LABEL

        result = self._deterministic.parse(extracted)

        found = predict.predict_fields(extracted.text)
        if not found:
            return result

        minimum = getattr(settings, "ocr_field_model_min_confidence", 0.35)
        filled: list[str] = []

        for label, field in found.items():
            target = SPEC_BY_LABEL[label].target
            if target is None or field.value is None:
                continue
            if field.confidence < minimum:
                continue
            if _current(result, target) is not None:
                continue  # deterministic extraction already has this field
            _assign(result, target, field.value)
            filled.append(f"{target} ({field.confidence:.2f})")

        if not filled:
            return result

        # The derived tax total and the arithmetic cross-checks were computed
        # against a sparser result; both have to be redone now that fields have
        # been added, or the invoice carries validation errors for values it
        # now has and skips checks that have become possible.
        result.validation_errors.clear()
        self._deterministic._derive_tax_amount_if_missing(result)
        reconciled = self._deterministic._reconcile_shifted_gst_summary(result)
        if reconciled:
            result.warnings = [warning for warning in result.warnings if not warning.startswith(
                "CGST + SGST + IGST does not closely match"
            )]
            reconciled_targets = {
                "subtotal", "taxable_amount", "cgst_amount", "sgst_amount",
                "tax_amount", "round_off", "total_amount",
            }
            filled = [entry for entry in filled if entry.split(" ", 1)[0] not in reconciled_targets]
        self._deterministic._validate(result)
        result.parsing_confidence = self._deterministic._score_confidence(result)

        result.parser_name = self.name
        result.parser_version = self.version
        # Provenance matters on a review screen: a value the model inferred
        # from a blurred scan deserves a closer look than one read verbatim.
        if filled:
            result.add_warning(
                "Some fields were recovered by the OCR field model rather than "
                f"read directly: {', '.join(filled)}."
            )
        logger.info(
            "invoice_parsing.ml_fields_filled count=%d fields=%s confidence=%.2f",
            len(filled), filled, result.parsing_confidence,
        )
        return result
