"""The geometry-aware extraction path, end to end.

Invoices here are built as real PDFs at explicit coordinates, so each test
states a layout a vendor actually uses. The claim under test is that the same
code reads all of them without knowing any of them.
"""
from decimal import Decimal

import fitz
import pytest

from app.modules.invoices.services import structured_extraction_service as structured

# Checksum-valid GSTINs, so validation findings in these tests come from the
# extraction rather than from placeholder numbers.
KARNATAKA_SELLER = "29AAGCB7383J1Z4"
KARNATAKA_BUYER = "29AAECS1234K1Z9"
MAHARASHTRA_BUYER = "27AAPFU0939F1ZV"


def build_pdf(tmp_path, lines, name="invoice.pdf", pages=1):
    """Write a PDF from (text, x, y[, size]) tuples, monospaced so column
    positions in the source match column positions on the page."""
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        for entry in lines:
            text, x, y = entry[0], entry[1], entry[2]
            size = entry[3] if len(entry) > 3 else 10
            page.insert_text((x, y), text, fontsize=size, fontname="cour")
    path = tmp_path / name
    document.save(str(path))
    document.close()
    return path


STANDARD_LAYOUT = [
    ("ALPHA STEEL WORKS", 200, 60, 14),
    ("TAX INVOICE", 240, 85, 12),
    ("Invoice No: ASW/2026/0042", 50, 130),
    ("Invoice Date: 14/08/2026", 50, 150),
    ("Due Date: 13/09/2026", 50, 170),
    ("Vendor: Alpha Steel Works", 320, 130),
    (f"GSTIN: {KARNATAKA_SELLER}", 320, 150),
    ("Bill To: Beta Constructions Pvt Ltd", 50, 210),
    (f"GSTIN: {KARNATAKA_BUYER}", 50, 230),
    ("Description        HSN     Qty      Rate       Amount", 50, 290),
    ("Steel Fabrication  7308     10   1500.00    15000.00", 50, 315),
    ("Welding Services   9988      4   1200.00     4800.00", 50, 335),
    ("Site Supervision   9983      2    500.00     1000.00", 50, 355),
    ("Sub Total                              20800.00", 250, 410),
    ("CGST 9%                                 1872.00", 250, 430),
    ("SGST 9%                                 1872.00", 250, 450),
    ("Grand Total                            24544.00", 250, 480),
]


@pytest.fixture
def standard(tmp_path):
    return structured.extract(build_pdf(tmp_path, STANDARD_LAYOUT))


class TestHeaderFields:
    def test_the_invoice_number_is_extracted(self, standard):
        assert standard.parsed.invoice_number == "ASW/2026/0042"

    def test_dates_are_extracted_and_not_swapped(self, standard):
        from datetime import date

        assert standard.parsed.invoice_date == date(2026, 8, 14)
        assert standard.parsed.due_date == date(2026, 9, 13)

    def test_every_amount_is_extracted(self, standard):
        parsed = standard.parsed
        assert parsed.subtotal == Decimal("20800.00")
        assert parsed.cgst_amount == Decimal("1872.00")
        assert parsed.sgst_amount == Decimal("1872.00")
        assert parsed.total_amount == Decimal("24544.00")

    def test_the_tax_total_is_derived_when_the_invoice_states_only_its_parts(self, standard):
        assert standard.parsed.tax_amount == Decimal("3744.00")

    def test_the_grand_total_is_not_confused_with_the_subtotal(self, standard):
        assert standard.parsed.total_amount != standard.parsed.subtotal


class TestPartyAttribution:
    def test_each_party_gets_its_own_name(self, standard):
        assert standard.parsed.vendor.name == "Alpha Steel Works"
        assert standard.parsed.customer.name == "Beta Constructions Pvt Ltd"

    def test_each_party_gets_its_own_gstin(self, standard):
        """Two `GSTIN:` labels, identical wording, different owners. Only
        position can tell them apart."""
        assert standard.parsed.vendor.gstin == KARNATAKA_SELLER
        assert standard.parsed.customer.gstin == KARNATAKA_BUYER

    def test_the_gstins_are_not_both_assigned_to_one_party(self, standard):
        assert standard.parsed.vendor.gstin != standard.parsed.customer.gstin

    def test_attribution_survives_the_buyer_block_coming_first(self, tmp_path):
        # Reading order is not evidence: plenty of invoices put the buyer
        # above the seller.
        reordered = [
            ("TAX INVOICE", 240, 60, 12),
            ("Invoice No: ASW/2026/0042", 50, 110),
            ("Invoice Date: 14/08/2026", 50, 130),
            ("Bill To: Beta Constructions Pvt Ltd", 50, 180),
            (f"GSTIN: {KARNATAKA_BUYER}", 50, 200),
            ("Vendor: Alpha Steel Works", 50, 250),
            (f"GSTIN: {KARNATAKA_SELLER}", 50, 270),
            ("Grand Total                            24544.00", 250, 400),
        ]

        result = structured.extract(build_pdf(tmp_path, reordered))

        assert result.parsed.vendor.gstin == KARNATAKA_SELLER
        assert result.parsed.customer.gstin == KARNATAKA_BUYER


class TestLineItems:
    def test_every_row_is_recovered(self, standard):
        assert len(standard.parsed.line_items) == 3

    def test_each_column_lands_in_the_right_field(self, standard):
        first = standard.parsed.line_items[0]
        assert first.description == "Steel Fabrication"
        assert first.hsn_sac == "7308"
        assert first.quantity == Decimal("10")
        assert first.unit_price == Decimal("1500.00")
        assert first.total_amount == Decimal("15000.00")

    def test_the_summary_rows_are_not_taken_as_line_items(self, standard):
        """Vendors put the subtotal and taxes inside the table's ruling;
        taking them as items double-counts the invoice."""
        descriptions = [item.description or "" for item in standard.parsed.line_items]
        assert not any("Total" in text or "GST" in text for text in descriptions)

    def test_the_line_items_sum_to_the_subtotal(self, standard):
        total = sum(item.total_amount for item in standard.parsed.line_items)
        assert total == standard.parsed.subtotal


class TestLayoutIndependence:
    def test_a_two_column_header_is_read_correctly(self, tmp_path):
        two_column = [
            ("TAX INVOICE", 240, 60, 12),
            ("Invoice No: ASW/2026/0042", 50, 120),
            ("Invoice Date: 14/08/2026", 320, 120),
            ("Due Date: 13/09/2026", 50, 145),
            ("Vendor: Alpha Steel Works", 320, 145),
            ("Grand Total                            24544.00", 250, 400),
        ]

        result = structured.extract(build_pdf(tmp_path, two_column))

        from datetime import date
        assert result.parsed.invoice_number == "ASW/2026/0042"
        assert result.parsed.invoice_date == date(2026, 8, 14)
        assert result.parsed.due_date == date(2026, 9, 13)

    @pytest.mark.parametrize(
        ("number_label", "total_label"),
        [
            ("Invoice No", "Grand Total"),
            ("Bill No", "Amount Payable"),
            ("Tax Invoice No", "Net Payable"),
            ("Document No", "Total Amount"),
        ],
    )
    def test_different_vendor_wordings_all_extract(self, tmp_path, number_label, total_label):
        """No template: four vendors' wordings, one extractor."""
        lines = [
            ("TAX INVOICE", 240, 60, 12),
            (f"{number_label}: ASW/2026/0042", 50, 120),
            ("Invoice Date: 14/08/2026", 50, 145),
            (f"{total_label}                       24544.00", 250, 400),
        ]

        result = structured.extract(build_pdf(tmp_path, lines))

        assert result.parsed.invoice_number == "ASW/2026/0042"
        assert result.parsed.total_amount == Decimal("24544.00")

    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("14/08/2026", "2026-08-14"),
            ("14-08-2026", "2026-08-14"),
            ("2026-08-14", "2026-08-14"),
            ("14 Aug 2026", "2026-08-14"),
        ],
    )
    def test_different_date_formats_all_extract(self, tmp_path, written, expected):
        lines = [
            ("TAX INVOICE", 240, 60, 12),
            ("Invoice No: ASW/2026/0042", 50, 120),
            (f"Invoice Date: {written}", 50, 145),
            ("Grand Total                            24544.00", 250, 400),
        ]

        result = structured.extract(build_pdf(tmp_path, lines))

        assert result.parsed.invoice_date.isoformat() == expected

    def test_indian_lakh_grouping_is_read_correctly(self, tmp_path):
        lines = [
            ("TAX INVOICE", 240, 60, 12),
            ("Invoice No: ASW/2026/0042", 50, 120),
            ("Invoice Date: 14/08/2026", 50, 145),
            ("Grand Total                         1,92,407.04", 250, 400),
        ]

        result = structured.extract(build_pdf(tmp_path, lines))

        assert result.parsed.total_amount == Decimal("192407.04")


class TestValidationAndConfidence:
    def test_a_clean_invoice_validates_and_auto_accepts(self, standard):
        """A fully correct extraction must not be sent to manual review —
        that is the same as having no automation."""
        assert standard.validation.is_clean, [f.message for f in standard.validation.errors]
        assert not standard.confidence.needs_review
        assert standard.confidence.document_confidence > 0.8

    def test_a_total_that_does_not_reconcile_is_caught(self, tmp_path):
        broken = [entry for entry in STANDARD_LAYOUT
                  if not entry[0].startswith("Grand Total")]
        broken.append(("Grand Total                            99999.00", 250, 480))

        result = structured.extract(build_pdf(tmp_path, broken))

        assert "TOTAL_DOES_NOT_RECONCILE" in {f.code for f in result.validation.errors}
        assert result.confidence.needs_review

    def test_an_inter_state_mismatch_is_flagged(self, tmp_path):
        # Both parties charged CGST/SGST but in different states.
        lines = [entry for entry in STANDARD_LAYOUT
                 if KARNATAKA_BUYER not in entry[0]]
        lines.append((f"GSTIN: {MAHARASHTRA_BUYER}", 50, 230))

        result = structured.extract(build_pdf(tmp_path, lines))

        codes = {f.code for f in result.validation.findings}
        assert "CGST_SGST_ON_INTER_STATE_SUPPLY" in codes


class TestEvidence:
    def test_every_extracted_field_carries_its_source_location(self, standard):
        """The review screen highlights a field's source region; without a
        box it can only ask the user to hunt for it."""
        evidence = standard.evidence_dict()

        assert "total_amount" in evidence
        assert evidence["total_amount"]["box"] is not None
        assert evidence["total_amount"]["page_number"] == 1

    def test_evidence_records_why_the_value_was_chosen(self, standard):
        assert standard.evidence_dict()["invoice_number"]["reasons"]

    def test_evidence_serialises_for_storage(self, standard):
        import json

        json.dumps(standard.evidence_dict())


class TestMultiPage:
    def test_header_fields_are_taken_from_wherever_they_score_best(self, tmp_path):
        """Not "page one wins": a two-page invoice routinely carries its
        totals on page two, and a page-one preference takes a partial
        subtotal over the real grand total."""
        document = fitz.open()
        first = document.new_page()
        first.insert_text((50, 120), "Invoice No: ASW/2026/0042", fontsize=10, fontname="cour")
        first.insert_text((50, 145), "Invoice Date: 14/08/2026", fontsize=10, fontname="cour")
        first.insert_text((50, 200), "Description        HSN     Qty      Rate       Amount",
                          fontsize=10, fontname="cour")
        first.insert_text((50, 225), "Steel Fabrication  7308     10   1500.00    15000.00",
                          fontsize=10, fontname="cour")
        second = document.new_page()
        second.insert_text((250, 300), "Grand Total                            24544.00",
                           fontsize=10, fontname="cour")
        path = tmp_path / "two_page.pdf"
        document.save(str(path))
        document.close()

        result = structured.extract(path)

        assert result.parsed.invoice_number == "ASW/2026/0042"
        assert result.parsed.total_amount == Decimal("24544.00")
        assert result.ocr_document.page_count == 2

    def test_the_page_a_value_came_from_is_recorded(self, tmp_path):
        document = fitz.open()
        document.new_page().insert_text((50, 120), "Invoice No: ASW/2026/0042",
                                        fontsize=10, fontname="cour")
        document.new_page().insert_text((250, 300), "Grand Total       24544.00",
                                        fontsize=10, fontname="cour")
        path = tmp_path / "two_page.pdf"
        document.save(str(path))
        document.close()

        evidence = structured.extract(path).evidence_dict()

        assert evidence["invoice_number"]["page_number"] == 1
        assert evidence["total_amount"]["page_number"] == 2


class TestRouting:
    def test_a_text_pdf_is_never_sent_to_ocr(self, standard):
        assert standard.used_ocr is False
        assert all(entry["route"] == "native" for entry in standard.page_decisions)

    def test_office_formats_are_left_to_the_text_parser(self, tmp_path):
        # They keep exact cell structure of their own; forcing them through a
        # geometry pipeline built for pixels would be strictly worse.
        assert not structured.is_supported(tmp_path / "invoice.docx")
        assert not structured.is_supported(tmp_path / "invoice.xlsx")

    @pytest.mark.parametrize("name", ["a.pdf", "a.png", "a.jpg", "a.tiff"])
    def test_pdf_and_raster_formats_take_the_structured_route(self, name):
        assert structured.is_supported(name)
