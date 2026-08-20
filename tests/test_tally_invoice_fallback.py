from datetime import date
from decimal import Decimal

from app.modules.invoices.dto import ParsedInvoiceResult
from app.modules.invoices.parsers import invoice_field_parser as fields
from app.modules.invoices.parsers.generic_invoice_parser import GenericInvoiceParser
from app.modules.invoices.parsers.invoice_table_parser import parse_line_items_from_ocr_text


TALLY_OCR_TEXT = """Tax Invoice
(ORIGINAL FOR RECIPIENT)
Invoice No.
Dated
KH/25-26/50
21-Apr-25
Keyhome Technologies LLP
1st floor, plot no.102, survey no.41/Part, and 42
GSTIN/UIN: 36AAUFK4104H2Z7
Consignee (Ship to)
IOTIQ INNOVATIONS PVT LTD
Level 7, Pardha Picasa, Durgam Cheruvu Road
GSTIN/UIN: 36AAECI9929F1Z9
Buyer (Bill to)
IOTIQ INNOVATIONS PVT LTD
Buyer's Order No.
2657
"""


def test_tally_header_values_survive_row_wise_ocr_order():
    assert fields.extract_invoice_number(TALLY_OCR_TEXT) == "KH/25-26/50"
    assert fields.extract_invoice_date(TALLY_OCR_TEXT) == date(2025, 4, 21)

    vendor, customer, _warnings = fields.extract_parties(TALLY_OCR_TEXT)
    assert vendor["name"] == "Keyhome Technologies LLP"
    assert vendor["gstin"] == "36AAUFK4104H2Z7"
    assert customer["name"] == "IOTIQ INNOVATIONS PVT LTD"
    assert customer["gstin"] == "36AAECI9929F1Z9"


def test_buyers_order_label_is_not_the_customer_name():
    text = "Buyer's Order No.\n2657"
    vendor, customer, _warnings = fields.extract_parties(text)
    assert vendor["name"] is None
    assert customer["name"] is None


def test_ocr_damaged_private_limited_suffix_is_normalised():
    assert fields._clean_party_name('IOTIQ INNOVATIONS PVT L"D State Name') == \
        "IOTIQ INNOVATIONS PVT LTD"


def test_shifted_gst_summary_is_reconciled_only_when_arithmetic_agrees():
    parsed = ParsedInvoiceResult(
        subtotal=Decimal("23064.48"),
        cgst_amount=Decimal("11532.24"),
        sgst_amount=Decimal("0.48"),
        tax_amount=Decimal("128136.00"),
        total_amount=Decimal("128136.00"),
    )

    assert GenericInvoiceParser._reconcile_shifted_gst_summary(parsed) is True
    assert parsed.subtotal == Decimal("128136.00")
    assert parsed.taxable_amount == Decimal("128136.00")
    assert parsed.cgst_amount == Decimal("11532.24")
    assert parsed.sgst_amount == Decimal("11532.24")
    assert parsed.tax_amount == Decimal("23064.48")
    assert parsed.round_off == Decimal("-0.48")
    assert parsed.total_amount == Decimal("151200.00")


def test_unrelated_amounts_are_not_rewritten():
    parsed = ParsedInvoiceResult(
        subtotal=Decimal("20000"),
        cgst_amount=Decimal("900"),
        sgst_amount=Decimal("900"),
        tax_amount=Decimal("1800"),
        total_amount=Decimal("21800"),
    )

    assert GenericInvoiceParser._reconcile_shifted_gst_summary(parsed) is False
    assert parsed.total_amount == Decimal("21800")


def test_flattened_tally_item_row_is_recovered_by_arithmetic():
    text = """Description of Goods
Rate per Disc. % Amount
1,28,136.00
85365090 200 Nos 756.00 640.68 Nos
1 Wifi_Dongle
11,532.24
Cgst
"""

    items = parse_line_items_from_ocr_text(text)

    assert len(items) == 1
    assert items[0].description == "Wifi_Dongle"
    assert items[0].hsn_sac == "85365090"
    assert items[0].quantity == Decimal("200")
    assert items[0].unit == "Nos"
    assert items[0].unit_price == Decimal("640.68")
    assert items[0].total_amount == Decimal("128136.00")


def test_flattened_row_is_rejected_when_amount_does_not_reconcile():
    text = """Description of Goods
99,999.00
85365090 200 Nos 756.00 640.68 Nos
1 Wifi_Dongle
Cgst
"""
    assert parse_line_items_from_ocr_text(text) == []


def test_identifier_and_date_can_share_a_line_with_other_columns():
    text = """Kiot Innovations Pvt Ltd Invoice No. Dated
Plot No 102,SY No 41 KIOT/26-27/560 12-Jun-26
Kavuri Hills Phase-1 Delivery Note
"""
    assert fields.extract_invoice_number(text) == "KIOT/26-27/560"
    assert fields.extract_invoice_date(text) == date(2026, 6, 12)


def test_tally_summary_pairs_values_on_either_side_of_labels():
    text = """8,325.00
749.25
CGST
SGST
749.25
0.50
ROUND OFF
9,824.00
26 NOS
Total
"""
    assert fields.extract_tally_summary(text) == {
        "total_amount": Decimal("9824.00"),
        "cgst_amount": Decimal("749.25"),
        "sgst_amount": Decimal("749.25"),
        "round_off": Decimal("0.50"),
        "tax_amount": Decimal("1498.50"),
        "subtotal": Decimal("8325.00"),
        "taxable_amount": Decimal("8325.00"),
    }


def test_column_stream_items_are_grouped_by_serial_and_arithmetic():
    text = """Descrlption of Goods
No.
1 FIRST ITEM
15 NOS
1,125.00
75.00 NOS
851770
180.00
2 NOS
2 SECOND ITEM
90.00 NOS
392390
Total
"""
    items = parse_line_items_from_ocr_text(text)
    assert [(item.description, item.quantity, item.unit_price, item.total_amount) for item in items] == [
        ("FIRST ITEM", Decimal("15"), Decimal("75.00"), Decimal("1125.00")),
        ("SECOND ITEM", Decimal("2"), Decimal("90.00"), Decimal("180.00")),
    ]
