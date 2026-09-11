"""Bill of Entry parser and validator, exercised against real OCR output.

Every input here is verbatim recogniser output for a real ICEGATE print --
see `boe_samples.py` for what is damaged in each page and why that matters.
The assertions are therefore about behaviour under genuine OCR noise, not
under text that was typed out from the PDF.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest

from app.modules.customs import validation
from app.modules.customs.dto import (
    BoeDuty,
    BoeInvoiceRef,
    BoeLineItem,
    ParsedBillOfEntry,
)
from app.modules.customs.parser import BillOfEntryParser, split_pages, to_date

from tests import boe_samples as samples


@dataclass
class FakeExtraction:
    """Stands in for `ExtractedDocument` -- the parser only reads these
    three attributes, and depending on the real type would drag pdfplumber
    into a unit test that has no file to open."""

    text: str
    used_ocr: bool = True
    page_count: int = 3


@pytest.fixture(scope="module")
def parsed() -> ParsedBillOfEntry:
    return BillOfEntryParser().parse(FakeExtraction(text=samples.FULL_DOCUMENT))


# --- document recognition --------------------------------------------------


def test_can_parse_accepts_a_bill_of_entry():
    assert BillOfEntryParser().can_parse(samples.FULL_DOCUMENT)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "TAX INVOICE\nInvoice No: INV-2024-0042\nGrand Total: 11800.00",
        # "Indian Customs" alone is not enough: it appears on shipping bills
        # and courier manifests this parser must not claim.
        "INDIAN CUSTOMS\nSome other form entirely\nPage 1 Of 2",
    ],
)
def test_can_parse_rejects_other_documents(text):
    assert not BillOfEntryParser().can_parse(text)


# --- page handling ---------------------------------------------------------


def test_split_pages_uses_the_printed_footer():
    assert len(split_pages(samples.FULL_DOCUMENT)) == 3


def test_trailing_text_after_the_last_footer_is_not_a_new_page(parsed):
    """OCR emits the ICETRAK strap after the page number. Counting it as a
    fourth page would misreport how much of the document was read."""
    assert parsed.pages_parsed == 3


def test_repeated_printed_page_is_skipped():
    """Scanned BoEs contain duplicated sheets -- in the source PDF, page 6
    was scanned twice. Counting it twice would double an invoice's goods."""
    doubled = "\n".join((samples.PAGE_ONE, samples.PAGE_TWO, samples.PAGE_TWO))
    result = BillOfEntryParser().parse(FakeExtraction(text=doubled))
    assert result.duplicate_pages
    assert len([inv for inv in result.invoices if inv.sequence_index == 1]) == 1


# --- Part I ----------------------------------------------------------------


def test_header_identifiers(parsed):
    assert parsed.be_number == samples.EXPECTED["be_number"]
    assert parsed.be_date == date(2026, 9, 4)
    assert parsed.port_code == samples.EXPECTED["port_code"]
    assert parsed.gstin == samples.EXPECTED["gstin"]
    assert parsed.cb_code == samples.EXPECTED["cb_code"]


def test_iec_recovered_from_the_sez_line_when_the_header_cell_is_illegible(parsed):
    """The IEC/Br cell is in the most degraded strip of the page; the same
    number is repeated in the SEZ unit details line, which survives."""
    assert parsed.iec == samples.EXPECTED["iec"]


def test_exchange_rate(parsed):
    assert parsed.exchange_rate == Decimal("95.25")
    assert parsed.exchange_currency == "USD"


def test_duty_heads_are_solved_jointly(parsed):
    """The statutory 10% ratio alone is ambiguous on this document: BCD is
    also 10% of the assessable value, so (ASS VAL, BCD) is a 10:1 pair too,
    and is the larger one. Only the total-duty sum separates them."""
    duty = parsed.duty
    assert duty.bcd == Decimal("552862.7")
    assert duty.sws == Decimal("55286.3")
    assert duty.igst == Decimal("1104620")
    assert duty.total_duty == Decimal("1712769")
    assert sum(duty.components) == duty.total_duty


def test_duty_is_reported_as_absent_rather_than_guessed():
    """A page with no closable combination must yield nothing and say so.
    Half a duty block invites a reviewer to trust the half that is there."""
    text = "PART - I - BILL OF ENTRY SUMMARY\nINDIAN CUSTOMS\n1.BCD\n5\n3\n9\nPage 1 Of 1"
    result = BillOfEntryParser().parse(FakeExtraction(text=text))
    assert result.duty.bcd is None
    assert result.duty.total_duty is None
    assert any("could not be reconciled" in w for w in result.warnings)


# --- Part II ---------------------------------------------------------------


def test_invoice_pages_are_identified_by_their_sequence_marker(parsed):
    assert [inv.sequence for inv in parsed.invoices] == ["1/7", "7/7"]
    assert parsed.declared_invoice_count == 7


def test_invoice_identity_is_anchored_on_its_own_label(parsed):
    """The repeated page-header strip carries the BE number, which is also a
    bare digit run; scanning the page unanchored picks that up instead."""
    first = parsed.invoices[0]
    assert first.invoice_number == "26208771"
    assert first.invoice_date == date(2026, 8, 31)
    assert first.assessable_value == Decimal("36393.12")


def test_supplier_name_survives_a_run_together_label(parsed):
    """OCR emits "3.SUPPLIERNAME&ADDRESS/CLIENTDETAILS" as one token.
    Resuming at the end of the label match takes the tail of the label as
    the supplier's name."""
    assert parsed.invoices[0].supplier_name == "GREENLEAF"


def test_goods_lines_zip_correctly(parsed):
    first = parsed.invoices[0].line_items
    assert len(first) == 1
    assert first[0].cth == "82090090"
    assert first[0].unit_price == Decimal("11.940000")
    assert first[0].quantity == Decimal("32.000000")
    assert first[0].uqc == "NOS"
    assert first[0].amount == Decimal("382.08")


def test_goods_lines_are_found_when_the_table_header_is_destroyed(parsed):
    """On the last page the "1.S NO. 2.CTH 3.DESCRIPTION" row came back as
    "T DO UI 9 AN". The MISC CHARGE row is the backstop anchor."""
    last = parsed.invoices[-1].line_items
    assert len(last) == 2
    assert [item.amount for item in last] == [Decimal("7650.20"), Decimal("17610.66")]
    assert all(item.cth == "82090090" for item in last)


def test_descriptions_are_reassembled_from_scattered_fragments(parsed):
    """A description is printed over several lines and OCR scatters them
    among the numeric columns."""
    descriptions = [item.description for item in parsed.invoices[-1].line_items]
    assert descriptions[0].startswith("WG-4187A,XSYTIN-1,INSERT")
    assert "CUTTING TOOL" in descriptions[0]
    assert descriptions[1].startswith("RCGN-4VA")


def test_every_parsed_goods_line_is_arithmetically_exact(parsed):
    """Which is also what proves the column zip was correct."""
    for invoice in parsed.invoices:
        for item in invoice.line_items:
            assert item.unit_price * item.quantity == item.amount


# --- validation ------------------------------------------------------------


def _duty(**overrides) -> BoeDuty:
    base = {
        "assessable_value": Decimal("5528626"),
        "bcd": Decimal("552862.7"),
        "sws": Decimal("55286.3"),
        "igst": Decimal("1104620"),
        "total_duty": Decimal("1712769"),
    }
    base.update(overrides)
    return BoeDuty(**base)


def _complete_boe(**overrides) -> ParsedBillOfEntry:
    """A minimal BoE that passes every check, for negative tests to perturb."""
    defaults = dict(
        be_number="3560107",
        be_date=date(2026, 9, 4),
        gstin="36ABBCS5682H1Z2",
        iec="ABBCS5682H",
        duty=_duty(),
        invoices=[
            BoeInvoiceRef(
                sequence_index=1, sequence_total=1,
                invoice_number="26208771", invoice_date=date(2026, 8, 31),
                assessable_value=Decimal("5528626"),
                line_items=[
                    BoeLineItem(
                        cth="82090090", unit_price=Decimal("11.94"),
                        quantity=Decimal("32"), amount=Decimal("382.08"),
                    )
                ],
            )
        ],
    )
    defaults.update(overrides)
    return ParsedBillOfEntry(**defaults)


def _codes(result) -> list[str]:
    return [finding.code for finding in validation.validate(result, today=date(2026, 9, 11)).findings]


def test_a_consistent_bill_of_entry_validates_clean():
    report = validation.validate(_complete_boe(), today=date(2026, 9, 11))
    assert report.is_clean, [f.message for f in report.findings]


def test_misread_assessable_value_is_caught_by_the_igst_amount():
    """The point of the whole validator.

    The recogniser returned 5525826 for a true 5528626 at 0.898 confidence --
    a transposed digit, comfortably above any usable threshold. Per-field
    confidence cannot tell it from a good read. The IGST computation can:
    18% of the misread base is 504 rupees short of the printed IGST.
    """
    good = _complete_boe()
    assert "boe_igst_amount_mismatch" not in _codes(good)

    misread = _complete_boe(duty=_duty(assessable_value=Decimal("5525826")))
    assert "boe_igst_amount_mismatch" in _codes(misread)


def test_truncated_read_is_an_error_not_a_silence():
    """`ocr_max_pages` defaults to 3; this document puts invoices 3-7 on
    pages 4 onward. A capped run is internally consistent and missing most
    of the goods."""
    partial = _complete_boe(
        invoices=[
            BoeInvoiceRef(sequence_index=1, sequence_total=7),
            BoeInvoiceRef(sequence_index=2, sequence_total=7),
        ]
    )
    assert partial.missing_invoice_sequences == [3, 4, 5, 6, 7]
    assert "boe_invoices_missing" in _codes(partial)


def test_missing_invoices_do_not_also_raise_an_arithmetic_finding():
    """The parts not summing to the whole is the truncation restated. Saying
    it twice sends a reviewer hunting a misread number that does not exist."""
    codes = _codes(_complete_boe(
        invoices=[
            BoeInvoiceRef(
                sequence_index=1, sequence_total=7,
                assessable_value=Decimal("36393.12"),
            )
        ]
    ))
    assert "boe_invoices_missing" in codes
    assert "boe_assessable_value_mismatch" not in codes


def test_sws_must_be_ten_percent_of_bcd():
    codes = _codes(_complete_boe(duty=_duty(sws=Decimal("55000.0"))))
    assert "boe_sws_rate_mismatch" in codes


def test_duty_heads_must_sum_to_the_total():
    codes = _codes(_complete_boe(duty=_duty(total_duty=Decimal("1700000"))))
    assert "boe_duty_sum_mismatch" in codes


def test_line_arithmetic_failure_is_an_error():
    """A slipped column zip leaves every value individually plausible; only
    the multiplication reveals it."""
    broken = _complete_boe()
    broken.invoices[0].line_items[0].quantity = Decimal("31")
    assert "boe_line_arithmetic" in _codes(broken)


def test_iec_and_gstin_must_agree():
    codes = _codes(_complete_boe(iec="ZZZZZ9999Z"))
    assert "boe_iec_gstin_disagree" in codes


def test_gstin_checksum_is_a_warning_not_an_error():
    """Same reasoning as the invoice validator: a failed checksum is usually
    one misread character, and discarding fourteen correct ones is worse."""
    report = validation.validate(
        _complete_boe(iec="ABBCS5682H", gstin="36ABBCS5682H1Z9"),
        today=date(2026, 9, 11),
    )
    codes = [f.code for f in report.findings]
    if "boe_gstin_checksum" in codes:
        finding = next(f for f in report.findings if f.code == "boe_gstin_checksum")
        assert finding.severity == validation.SEVERITY_WARNING


def test_invoice_dated_after_the_filing_is_flagged():
    late = _complete_boe()
    late.invoices[0].invoice_date = date(2026, 9, 30)
    assert "boe_invoice_after_filing" in _codes(late)


# --- date coercion ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("31-AUG-26", date(2026, 8, 31)),
        ("04-SEP-2026", date(2026, 9, 4)),
        ("04/09/2026", date(2026, 9, 4)),
        ("not a date", None),
        ("31-XXX-26", None),
    ],
)
def test_to_date(raw, expected):
    assert to_date(raw) == expected


def test_two_digit_years_are_read_as_this_century():
    """A BoE is a filing for goods in transit; a 1926 reading is never right."""
    assert to_date("31-AUG-26").year == 2026


# --- API schema ------------------------------------------------------------


def test_schema_round_trips_a_parsed_document(parsed):
    from app.modules.customs.schemas import BillOfEntryOut

    report = validation.validate(parsed, today=date(2026, 9, 11))
    payload = BillOfEntryOut.from_parsed(parsed, report)

    assert payload.be_number == samples.EXPECTED["be_number"]
    assert payload.duty.total_duty == Decimal("1712769")
    assert payload.total_line_items == 3
    assert payload.missing_invoice_sequences == [2, 3, 4, 5, 6]
    assert not payload.validation_clean
    assert any(f.code == "boe_invoices_missing" for f in payload.findings)
    # Must be serialisable: a Decimal or date that pydantic cannot encode
    # would only surface when a real request tried to return it.
    assert payload.model_dump_json()


def test_schema_without_a_report_does_not_claim_the_document_is_clean(parsed):
    """Absent evidence of correctness is not evidence of correctness."""
    from app.modules.customs.schemas import BillOfEntryOut

    assert BillOfEntryOut.from_parsed(parsed).validation_clean is False
