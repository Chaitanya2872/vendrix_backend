"""Validation: catching extractions that are individually plausible and
collectively impossible.

This is the layer that finds what per-field confidence cannot. A misread
total sits next to a correct subtotal looking perfectly fine on its own; only
adding them up exposes it.
"""
from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.modules.invoices.dto import ParsedInvoiceResult, ParsedLineItem, ParsedParty
from app.modules.invoices.services import validation_service
from app.modules.invoices.services.validation_service import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    gstin_checksum_char,
    is_valid_gstin,
    validate,
)

TODAY = date(2026, 8, 19)

# Checksum-valid GSTINs from two different states, so intra- and inter-state
# scenarios can both be built from real numbers.
KARNATAKA = "29AAGCB7383J1Z4"
# A second Karnataka number, so an intra-state invoice can have two distinct
# parties in the same state — which is what "intra-state" means, and what a
# single shared GSTIN would (correctly) be flagged for.
KARNATAKA_BUYER = "29AAECS1234K1Z9"
MAHARASHTRA = "27AAPFU0939F1ZV"
GUJARAT = "24AAACC1206D1ZM"


def invoice(**overrides):
    """A clean, fully reconciling intra-state invoice."""
    defaults = dict(
        invoice_number="INV-2026-0042",
        invoice_date=date(2026, 8, 14),
        due_date=date(2026, 9, 13),
        subtotal=Decimal("19800.00"),
        taxable_amount=Decimal("19800.00"),
        cgst_amount=Decimal("1782.00"),
        sgst_amount=Decimal("1782.00"),
        tax_amount=Decimal("3564.00"),
        round_off=Decimal("-0.40"),
        total_amount=Decimal("23363.60"),
        vendor=ParsedParty(name="Alpha Steel Works", gstin=KARNATAKA),
        customer=ParsedParty(name="Beta Constructions", gstin=KARNATAKA_BUYER),
        line_items=[
            ParsedLineItem(description="Steel Fabrication", quantity=Decimal("10"),
                           unit_price=Decimal("1500.00"), taxable_value=Decimal("15000.00"),
                           hsn_sac="7308"),
            ParsedLineItem(description="Welding Services", quantity=Decimal("4"),
                           unit_price=Decimal("1200.00"), taxable_value=Decimal("4800.00"),
                           hsn_sac="9988"),
        ],
    )
    defaults.update(overrides)
    return ParsedInvoiceResult(**defaults)


def codes(report):
    return {finding.code for finding in report.findings}


class TestCleanInvoice:
    def test_a_fully_reconciling_invoice_produces_no_errors(self):
        report = validate(invoice(), today=TODAY)

        assert report.is_clean, f"unexpected errors: {[f.message for f in report.errors]}"

    def test_a_clean_invoice_produces_no_warnings_either(self):
        report = validate(invoice(), today=TODAY)

        assert report.warnings == [], [finding.message for finding in report.warnings]


class TestGstinChecksum:
    @pytest.mark.parametrize("gstin", [KARNATAKA, MAHARASHTRA, GUJARAT])
    def test_real_gstins_validate(self, gstin):
        assert is_valid_gstin(gstin)

    def test_a_single_altered_character_is_caught(self):
        """The whole point of a check digit: one OCR slip and the number no
        longer computes."""
        broken = KARNATAKA[:5] + "X" + KARNATAKA[6:]

        assert not is_valid_gstin(broken)

    def test_the_checksum_character_is_computed_correctly(self):
        assert gstin_checksum_char(KARNATAKA[:14]) == KARNATAKA[14]

    def test_a_checksum_failure_is_a_warning_not_an_error(self):
        """A failed check digit usually means OCR misread one character, not
        that the number is fake. Discarding it would throw away fourteen
        correct characters a reviewer could fix in seconds."""
        report = validate(invoice(vendor=ParsedParty(name="Alpha", gstin="29ABCDE1234F1Z5")),
                          today=TODAY)

        assert "GSTIN_CHECKSUM_FAILED" in codes(report)
        assert report.is_clean, "a checksum failure must not block acceptance on its own"

    def test_an_invalid_state_code_is_flagged(self):
        report = validate(invoice(vendor=ParsedParty(name="Alpha", gstin="88AAGCB7383J1Z4")),
                          today=TODAY)

        assert "GSTIN_BAD_STATE_CODE" in codes(report)

    def test_a_malformed_gstin_is_flagged(self):
        report = validate(invoice(vendor=ParsedParty(name="Alpha", gstin="NOTAGSTIN")), today=TODAY)

        assert "GSTIN_MALFORMED" in codes(report)

    def test_the_same_gstin_on_both_parties_is_an_error(self):
        # Not a checksum problem: both numbers are valid, they were just read
        # from the same block.
        report = validate(invoice(
            vendor=ParsedParty(name="Alpha", gstin=KARNATAKA),
            customer=ParsedParty(name="Beta", gstin=KARNATAKA),
        ), today=TODAY)

        assert "SAME_GSTIN_BOTH_PARTIES" in codes(report)
        assert not report.is_clean


class TestArithmetic:
    def test_a_total_that_does_not_reconcile_is_an_error(self):
        """The check that catches what nothing else can: every field read
        confidently, and they do not add up."""
        report = validate(invoice(total_amount=Decimal("25000.00")), today=TODAY)

        assert "TOTAL_DOES_NOT_RECONCILE" in codes(report)
        assert not report.is_clean

    def test_rounding_within_tolerance_is_accepted(self):
        # Invoices round per line and again at the total; exact equality is
        # the wrong test.
        report = validate(invoice(total_amount=Decimal("23365.00")), today=TODAY)

        assert "TOTAL_DOES_NOT_RECONCILE" not in codes(report)

    def test_tax_components_that_do_not_sum_are_an_error(self):
        report = validate(invoice(tax_amount=Decimal("5000.00"),
                                  total_amount=Decimal("24799.60")), today=TODAY)

        assert "TAX_COMPONENTS_MISMATCH" in codes(report)

    def test_unequal_cgst_and_sgst_is_an_error(self):
        report = validate(invoice(cgst_amount=Decimal("1782.00"),
                                  sgst_amount=Decimal("1500.00")), today=TODAY)

        assert "CGST_SGST_MISMATCH" in codes(report)

    def test_a_subtotal_greater_than_the_total_is_an_error(self):
        report = validate(invoice(subtotal=Decimal("50000.00"),
                                  taxable_amount=Decimal("50000.00")), today=TODAY)

        assert "SUBTOTAL_EXCEEDS_TOTAL" in codes(report)

    def test_a_negative_total_is_an_error(self):
        report = validate(invoice(total_amount=Decimal("-100.00"),
                                  subtotal=None, taxable_amount=None), today=TODAY)

        assert "NEGATIVE_TOTAL" in codes(report)

    def test_an_implausibly_large_round_off_is_flagged(self):
        """A round-off is by definition sub-unit; a big one means a different
        field was read into it."""
        report = validate(invoice(round_off=Decimal("500.00"),
                                  total_amount=Decimal("23864.00")), today=TODAY)

        assert "ROUND_OFF_TOO_LARGE" in codes(report)

    def test_paid_plus_due_must_equal_the_total(self):
        report = validate(invoice(amount_paid=Decimal("1000.00"),
                                  amount_due=Decimal("1000.00")), today=TODAY)

        assert "PAID_PLUS_DUE_MISMATCH" in codes(report)


class TestTaxStructure:
    def test_igst_alongside_cgst_is_a_contradiction(self):
        report = validate(invoice(igst_amount=Decimal("3564.00")), today=TODAY)

        assert "IGST_WITH_CGST_SGST" in codes(report)
        assert not report.is_clean

    def test_cgst_without_sgst_is_flagged(self):
        report = validate(invoice(sgst_amount=None, tax_amount=Decimal("1782.00"),
                                  total_amount=Decimal("21581.60")), today=TODAY)

        assert "SGST_MISSING" in codes(report)

    def test_a_valid_inter_state_invoice_passes(self):
        report = validate(invoice(
            vendor=ParsedParty(name="Alpha", gstin=KARNATAKA),
            customer=ParsedParty(name="Beta", gstin=MAHARASHTRA),
            cgst_amount=None, sgst_amount=None,
            igst_amount=Decimal("3564.00"),
        ), today=TODAY)

        assert report.is_clean, [finding.message for finding in report.errors]
        assert report.warnings == []

    def test_intra_state_parties_charged_igst_are_flagged(self):
        """Catches a misattributed party block: both GSTINs read perfectly,
        they were just assigned to the wrong sides."""
        report = validate(invoice(
            vendor=ParsedParty(name="Alpha", gstin=KARNATAKA),
            customer=ParsedParty(name="Beta", gstin=KARNATAKA_BUYER),
            cgst_amount=None, sgst_amount=None,
            igst_amount=Decimal("3564.00"),
        ), today=TODAY)

        assert "IGST_ON_INTRA_STATE_SUPPLY" in codes(report)

    def test_inter_state_parties_charged_cgst_are_flagged(self):
        report = validate(invoice(
            vendor=ParsedParty(name="Alpha", gstin=KARNATAKA),
            customer=ParsedParty(name="Beta", gstin=MAHARASHTRA),
        ), today=TODAY)

        assert "CGST_SGST_ON_INTER_STATE_SUPPLY" in codes(report)


class TestDates:
    def test_a_future_invoice_date_is_an_error(self):
        report = validate(invoice(invoice_date=TODAY + timedelta(days=30),
                                  due_date=TODAY + timedelta(days=60)), today=TODAY)

        assert "INVOICE_DATE_IN_FUTURE" in codes(report)

    def test_a_due_date_before_the_invoice_date_is_an_error(self):
        report = validate(invoice(invoice_date=date(2026, 8, 14),
                                  due_date=date(2026, 7, 1)), today=TODAY)

        assert "DUE_BEFORE_INVOICE" in codes(report)

    def test_a_decades_old_date_is_flagged_as_a_probable_misread(self):
        report = validate(invoice(invoice_date=date(1998, 8, 14), due_date=None), today=TODAY)

        assert "INVOICE_DATE_IMPLAUSIBLY_OLD" in codes(report)

    def test_a_po_dated_after_the_invoice_is_flagged(self):
        report = validate(invoice(purchase_order_date=date(2026, 9, 1)), today=TODAY)

        assert "PO_AFTER_INVOICE" in codes(report)

    def test_an_invoice_dated_today_is_fine(self):
        report = validate(invoice(invoice_date=TODAY, due_date=TODAY + timedelta(days=30)),
                          today=TODAY)

        assert "INVOICE_DATE_IN_FUTURE" not in codes(report)


class TestLineItems:
    def test_line_items_that_do_not_sum_to_the_taxable_value_are_an_error(self):
        """Catches a dropped row, which no per-line check can see."""
        report = validate(invoice(line_items=[
            ParsedLineItem(description="Steel Fabrication", quantity=Decimal("10"),
                           unit_price=Decimal("1500.00"), taxable_value=Decimal("15000.00")),
        ]), today=TODAY)

        assert "LINE_ITEMS_DO_NOT_SUM" in codes(report)

    def test_a_line_whose_arithmetic_is_wrong_is_flagged(self):
        report = validate(invoice(line_items=[
            ParsedLineItem(description="Steel Fabrication", quantity=Decimal("10"),
                           unit_price=Decimal("1500.00"), taxable_value=Decimal("14000.00")),
            ParsedLineItem(description="Welding", quantity=Decimal("4"),
                           unit_price=Decimal("1200.00"), taxable_value=Decimal("4800.00")),
        ]), today=TODAY)

        assert "LINE_ARITHMETIC_MISMATCH" in codes(report)

    def test_a_malformed_hsn_is_flagged(self):
        report = validate(invoice(line_items=[
            ParsedLineItem(description="Steel", quantity=Decimal("10"),
                           unit_price=Decimal("1500.00"), taxable_value=Decimal("15000.00"),
                           hsn_sac="73"),
            ParsedLineItem(description="Welding", quantity=Decimal("4"),
                           unit_price=Decimal("1200.00"), taxable_value=Decimal("4800.00"),
                           hsn_sac="9988"),
        ]), today=TODAY)

        assert "HSN_MALFORMED" in codes(report)

    def test_no_line_items_is_a_warning_not_an_error(self):
        # A summary-only invoice is unusual but real; it should not block.
        report = validate(invoice(line_items=[]), today=TODAY)

        assert "NO_LINE_ITEMS" in codes(report)
        assert report.is_clean


class TestMissingFields:
    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ({"invoice_number": None}, "MISSING_INVOICE_NUMBER"),
            ({"invoice_date": None}, "MISSING_INVOICE_DATE"),
            ({"total_amount": None}, "MISSING_TOTAL"),
            ({"vendor": ParsedParty()}, "MISSING_VENDOR"),
        ],
    )
    def test_each_required_field_is_reported_when_absent(self, override, expected):
        report = validate(invoice(**override), today=TODAY)

        assert expected in codes(report)
        assert not report.is_clean

    def test_an_empty_extraction_reports_every_missing_field_not_just_the_first(self):
        report = validate(ParsedInvoiceResult(), today=TODAY)

        assert {"MISSING_INVOICE_NUMBER", "MISSING_INVOICE_DATE",
                "MISSING_TOTAL", "MISSING_VENDOR"} <= codes(report)


class TestReportShape:
    def test_findings_name_the_fields_they_implicate(self):
        """Confidence is lowered on the implicated fields, not on the whole
        document — so the findings have to say which."""
        report = validate(invoice(total_amount=Decimal("25000.00")), today=TODAY)

        assert "total_amount" in report.fields_with_findings()

    def test_errors_and_warnings_are_separable(self):
        report = validate(invoice(total_amount=Decimal("25000.00"),
                                  round_off=Decimal("500.00")), today=TODAY)

        assert report.errors and report.warnings
        assert all(finding.severity == SEVERITY_ERROR for finding in report.errors)
        assert all(finding.severity == SEVERITY_WARNING for finding in report.warnings)

    def test_the_report_serialises_for_storage(self):
        import json

        json.dumps(validate(invoice(), today=TODAY).to_dict())

    def test_every_finding_carries_a_message_a_human_can_act_on(self):
        report = validate(invoice(total_amount=Decimal("25000.00")), today=TODAY)

        for finding in report.findings:
            assert finding.message and len(finding.message) > 20
            assert finding.code == finding.code.upper()
