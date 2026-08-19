"""Confidence scoring.

This number decides whether an invoice is posted automatically or shown to a
human, so both failure directions are expensive: too high and wrong data
reaches the ledger, too low and the automation saves nobody any work. The
tests here pin the behaviours that keep it honest.
"""
from decimal import Decimal

import pytest

from app.modules.invoices.parsers.field_scoring import Candidate, Resolution
from app.modules.invoices.services import confidence_service
from app.modules.invoices.services.confidence_service import (
    REVIEW_THRESHOLD,
    score_document,
    score_field,
    score_line_items,
)
from app.modules.invoices.services.validation_service import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ValidationReport,
)
from app.modules.ocr.geometry import BoundingBox


def candidate(value="1000.00", score=0.95, ocr=1.0):
    result = Candidate(
        field="total_amount", raw_value=value, parsed_value=Decimal(value),
        relation="same-token", box=BoundingBox(0, 0, 100, 30), ocr_confidence=ocr,
    )
    result.score = score
    return result


def resolution(field_name, score=0.95, runner_up_score=None, ocr=1.0):
    winner = candidate(score=score, ocr=ocr)
    winner.field = field_name
    runner = None
    if runner_up_score is not None:
        runner = candidate(value="2000.00", score=runner_up_score, ocr=ocr)
        runner.field = field_name
    return Resolution(field=field_name, winner=winner, runner_up=runner)


def full_resolutions(**overrides):
    base = {
        "invoice_number": resolution("invoice_number"),
        "invoice_date": resolution("invoice_date"),
        "vendor_name": resolution("vendor_name"),
        "vendor_gstin": resolution("vendor_gstin"),
        "total_amount": resolution("total_amount"),
        "subtotal": resolution("subtotal"),
        "tax_amount": resolution("tax_amount"),
    }
    base.update(overrides)
    return base


def report_with(errors=0, warnings=0, fields=()):
    report = ValidationReport()
    for index in range(errors):
        report.add(f"ERROR_{index}", SEVERITY_ERROR, "Something does not reconcile at all.", *fields)
    for index in range(warnings):
        report.add(f"WARNING_{index}", SEVERITY_WARNING, "Something looks a little unusual.", *fields)
    return report


class TestFieldConfidence:
    def test_a_strong_uncontested_field_scores_high(self):
        result = score_field("total_amount", extraction_score=0.95, margin=1.0, ocr_confidence=0.99)

        assert result.confidence > 0.9
        assert not result.needs_attention

    def test_a_thin_margin_pulls_confidence_down(self):
        """What a single score cannot express: this value was very nearly a
        different number."""
        clear = score_field("total_amount", 0.90, margin=1.0, ocr_confidence=1.0)
        contested = score_field("total_amount", 0.90, margin=0.01, ocr_confidence=1.0)

        assert contested.confidence < clear.confidence

    def test_margin_cannot_push_a_field_above_its_extraction_quality(self):
        result = score_field("total_amount", extraction_score=0.60, margin=1.0, ocr_confidence=1.0)

        assert result.confidence <= 0.60

    def test_poor_ocr_confidence_drags_the_field_down(self):
        result = score_field("total_amount", 0.95, margin=1.0, ocr_confidence=0.30)

        assert result.confidence < 0.35
        assert result.needs_attention

    def test_a_field_implicated_by_validation_is_penalised_directly(self):
        """The reviewer's attention should land on the field at fault, not on
        every field equally."""
        clean = score_field("total_amount", 0.95, 1.0, 1.0)
        implicated = score_field("total_amount", 0.95, 1.0, 1.0,
                                 implicated_by=["TOTAL_DOES_NOT_RECONCILE"])

        assert implicated.confidence < clean.confidence * 0.6
        assert implicated.penalised_by == ["TOTAL_DOES_NOT_RECONCILE"]

    def test_confidence_stays_within_range(self):
        assert 0.0 <= score_field("x", 1.0, 1.0, 1.0).confidence <= 1.0
        assert 0.0 <= score_field("x", 0.0, 0.0, 0.0).confidence <= 1.0


class TestLineItemConfidence:
    def test_the_worst_row_decides_the_table(self):
        """Nineteen perfect rows and one misread amount is a wrong invoice;
        averaging hides exactly that."""
        assert score_line_items([0.99] * 19 + [0.30]) == pytest.approx(0.30)

    def test_no_line_items_yields_no_score_rather_than_zero(self):
        # A summary-only invoice has no line-item confidence; reporting 0.0
        # would drag the document score down for a document that is fine.
        assert score_line_items([]) is None


class TestDocumentConfidence:
    def test_a_clean_extraction_scores_high_enough_to_auto_accept(self):
        report = score_document(full_resolutions(), report_with())

        assert report.document_confidence >= REVIEW_THRESHOLD
        assert not report.needs_review

    def test_a_validation_error_caps_the_document_however_good_its_fields(self):
        """Arithmetic that does not reconcile proves at least one field is
        wrong, which per-field confidence provably cannot see."""
        report = score_document(full_resolutions(), report_with(errors=1))

        assert report.capped_by_errors
        assert report.document_confidence <= confidence_service.ERROR_CAP
        assert report.needs_review

    def test_warnings_cost_confidence_without_blocking(self):
        clean = score_document(full_resolutions(), report_with())
        warned = score_document(full_resolutions(), report_with(warnings=2))

        assert warned.document_confidence < clean.document_confidence
        assert not warned.capped_by_errors

    def test_the_warning_penalty_is_bounded(self):
        # Twenty small oddities should not score worse than a document with
        # a genuine arithmetic error.
        many = score_document(full_resolutions(), report_with(warnings=20))

        assert many.document_confidence > 0.4

    def test_missing_required_fields_lower_the_score(self):
        """Without counting the gap, an extraction that found nothing scores
        the same as one that found everything — both have no bad fields."""
        complete = score_document(full_resolutions(), report_with())
        partial = score_document(
            {"invoice_number": resolution("invoice_number")}, report_with()
        )

        assert partial.document_confidence < complete.document_confidence
        assert "total_amount" in partial.missing_required

    def test_the_total_matters_more_than_the_payment_terms(self):
        """Fields are weighted by consequence: posting the wrong total is a
        financial error, the wrong payment terms an inconvenience."""
        bad_total = score_document(
            full_resolutions(total_amount=resolution("total_amount", score=0.20)),
            report_with(),
        )
        bad_terms = score_document(
            {**full_resolutions(), "payment_terms": resolution("payment_terms", score=0.20)},
            report_with(),
        )

        assert bad_total.document_confidence < bad_terms.document_confidence

    def test_a_document_below_the_threshold_goes_to_review(self):
        weak = score_document(
            {name: resolution(name, score=0.45) for name in
             ("invoice_number", "invoice_date", "vendor_name", "total_amount")},
            report_with(),
        )

        assert weak.needs_review


class TestAttention:
    def test_weak_fields_are_named_for_the_reviewer(self):
        report = score_document(
            full_resolutions(total_amount=resolution("total_amount", score=0.25)),
            report_with(),
        )

        assert "total_amount" in report.fields_needing_attention()

    def test_a_strong_document_flags_nothing(self):
        report = score_document(full_resolutions(), report_with())

        assert report.fields_needing_attention() == []

    def test_a_field_can_need_attention_while_the_document_passes(self):
        """A single weak low-weight field should not send the whole invoice
        to review, but it should still be highlighted."""
        report = score_document(
            {**full_resolutions(), "place_of_supply": resolution("place_of_supply", score=0.20)},
            report_with(),
        )

        assert "place_of_supply" in report.fields_needing_attention()
        assert not report.needs_review

    def test_the_finding_that_penalised_a_field_is_recorded(self):
        report = score_document(
            full_resolutions(),
            report_with(errors=1, fields=("total_amount",)),
        )

        assert "ERROR_0" in report.fields["total_amount"].penalised_by


class TestSerialisation:
    def test_the_report_serialises_for_storage_and_the_api(self):
        import json

        payload = score_document(full_resolutions(), report_with(warnings=1)).to_dict()
        json.dumps(payload)

        assert "document_confidence" in payload
        assert "fields" in payload
        assert payload["fields"]["total_amount"]["confidence"] > 0
