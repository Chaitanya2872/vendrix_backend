"""The label lexicon.

Two things are being pinned here. First, that the wordings real vendors use
are recognised — the list is only as good as its coverage. Second, and more
important, that the *ordering and negatives* work: "Total Taxable Value"
contains three money labels and must resolve to exactly one of them, and
"Amount in Words" must resolve to none.
"""
import pytest

from app.modules.invoices.parsers import lexicon
from app.modules.invoices.parsers.lexicon import (
    CUSTOMER_LABELS,
    INVOICE_DATE,
    INVOICE_NUMBER,
    SUBTOTAL,
    TAXABLE_AMOUNT,
    TOTAL,
    best_match,
    match_any,
    match_label,
    normalise,
)


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Invoice No.", "invoice no"),
            ("INVOICE  NUMBER", "invoice number"),
            ("Invoice#", "invoice"),
            ("  Grand Total :  ", "grand total"),
            ("Sub-Total", "sub total"),
            ("GSTIN/UIN", "gstin uin"),
        ],
    )
    def test_punctuation_case_and_spacing_are_normalised_away(self, raw, expected):
        assert normalise(raw) == expected

    def test_accents_sprayed_on_by_noisy_ocr_are_stripped(self):
        assert normalise("Invóice Nö") == "invoice no"

    def test_empty_input_normalises_to_empty(self):
        assert normalise("") == ""
        assert normalise("   ") == ""


class TestExactAndVariantMatching:
    @pytest.mark.parametrize(
        "wording",
        [
            "Invoice No", "Invoice No.", "Invoice Number", "INVOICE #",
            "Inv No", "Bill No", "Tax Invoice No", "Document No", "Voucher No",
        ],
    )
    def test_every_common_wording_for_the_invoice_number_is_recognised(self, wording):
        assert match_label(wording, INVOICE_NUMBER) is not None

    @pytest.mark.parametrize(
        "wording",
        ["Invoice Date", "Bill Date", "Date of Invoice", "Issue Date", "Dated"],
    )
    def test_every_common_wording_for_the_invoice_date_is_recognised(self, wording):
        assert match_label(wording, INVOICE_DATE) is not None

    @pytest.mark.parametrize(
        "wording",
        ["Bill To", "Billed To", "Buyer", "Customer", "Sold To", "Invoice To", "Party Name"],
    )
    def test_every_common_wording_for_the_customer_is_recognised(self, wording):
        assert match_label(wording, CUSTOMER_LABELS) is not None

    def test_an_exact_match_scores_higher_than_a_containment_match(self):
        exact = match_label("Invoice No", INVOICE_NUMBER)
        contained = match_label("Supplier Invoice No and other detail", INVOICE_NUMBER)

        assert exact.exact is True
        assert exact.score > contained.score


class TestOcrTolerance:
    def test_a_capital_i_read_for_a_lowercase_l_still_matches(self):
        """The classic recognition error, and the one that silently empties a
        field on an otherwise good scan."""
        assert match_label("AmountPayabIe", TOTAL) is not None

    @pytest.mark.parametrize(
        "mangled",
        ["lnvoice No", "Invoive No", "Invoice N0", "1nvoice Number"],
    )
    def test_common_recognition_slips_still_match(self, mangled):
        assert match_label(mangled, INVOICE_NUMBER) is not None

    def test_a_genuinely_different_label_does_not_match_through_tolerance(self):
        # Tolerance must not become "matches everything": that turns a wrong
        # value into a confident one.
        assert match_label("Delivery Address", INVOICE_NUMBER) is None


class TestOrderingAndSpecificity:
    def test_total_taxable_value_resolves_to_the_taxable_field_not_the_total(self):
        """Contains 'total', 'taxable value' and 'value'. Only the most
        specific reading is correct, and ordering is what enforces it."""
        assert match_label("Total Taxable Value", TAXABLE_AMOUNT) is not None
        assert match_label("Total Taxable Value", TOTAL) is None

    def test_sub_total_is_not_read_as_the_grand_total(self):
        assert match_label("Sub Total", SUBTOTAL) is not None
        assert match_label("Sub Total", TOTAL) is None

    def test_total_tax_is_not_read_as_the_grand_total(self):
        assert match_label("Total Tax", TOTAL) is None

    def test_grand_total_resolves_to_the_total(self):
        assert best_match("Grand Total").field == "total_amount"

    def test_invoice_date_is_not_read_as_the_invoice_number(self):
        assert match_label("Invoice Date", INVOICE_NUMBER) is None

    def test_due_date_is_not_read_as_the_invoice_date(self):
        assert match_label("Due Date", INVOICE_DATE) is None
        assert best_match("Due Date").field == "due_date"

    def test_po_date_is_not_read_as_the_po_number(self):
        assert best_match("PO Date").field == "purchase_order_date"

    def test_ship_to_is_not_read_as_bill_to(self):
        assert match_label("Ship To", CUSTOMER_LABELS) is None
        assert best_match("Ship To").field == "ship_to_name"


class TestNegativeLabels:
    @pytest.mark.parametrize(
        "wording",
        ["Amount in Words", "Total Quantity", "Total Items", "Rupees Only", "Page"],
    )
    def test_a_negative_label_never_resolves_to_a_money_field(self, wording):
        """These all contain a money label. Reading one as an amount puts a
        page number or an item count into the invoice total."""
        matches = [match for match in match_any(wording) if match.field in lexicon.MONEY_FIELDS]
        assert matches == [], f"{wording!r} resolved to {[m.field for m in matches]}"

    def test_amount_chargeable_in_words_is_excluded(self):
        assert lexicon.is_negative_label("Amount Chargeable (in words)")

    def test_an_ordinary_total_is_not_flagged_negative(self):
        assert not lexicon.is_negative_label("Grand Total")


class TestGstFields:
    @pytest.mark.parametrize(
        ("wording", "expected"),
        [
            ("CGST", "cgst_amount"),
            ("C.GST", "cgst_amount"),
            ("Central GST", "cgst_amount"),
            ("SGST", "sgst_amount"),
            ("State GST", "sgst_amount"),
            ("UTGST", "sgst_amount"),
            ("IGST", "igst_amount"),
            ("Integrated GST", "igst_amount"),
            ("Compensation Cess", "cess_amount"),
        ],
    )
    def test_each_gst_component_resolves_to_its_own_field(self, wording, expected):
        assert best_match(wording).field == expected

    def test_a_rate_suffixed_component_still_resolves(self):
        # Vendors write "CGST @ 9%", "CGST 9%", "CGST(9%)" interchangeably.
        for wording in ("CGST @ 9%", "CGST 9%", "CGST(9%)"):
            assert best_match(wording).field == "cgst_amount", wording

    def test_gstin_is_not_confused_with_a_gst_amount(self):
        assert best_match("GSTIN").field == "gstin"

    def test_gst_reg_no_resolves_to_gstin(self):
        assert best_match("GST Reg No").field == "gstin"


class TestBankingAndFooter:
    @pytest.mark.parametrize(
        ("wording", "expected"),
        [
            ("Account No", "bank_account_number"),
            ("A/C No", "bank_account_number"),
            ("IFSC Code", "bank_ifsc"),
            ("SWIFT", "bank_ifsc"),
            ("UPI ID", "upi_id"),
            ("Bank Name", "bank_name"),
        ],
    )
    def test_banking_labels_resolve(self, wording, expected):
        assert best_match(wording).field == expected


class TestOtherHeaderFields:
    @pytest.mark.parametrize(
        ("wording", "expected"),
        [
            ("Place of Supply", "place_of_supply"),
            ("Payment Terms", "payment_terms"),
            ("Terms of Payment", "payment_terms"),
            ("E-Way Bill No", "delivery_reference"),
            ("Delivery Note", "delivery_reference"),
            ("IRN", "irn"),
            ("Ack No", "irn"),
            ("Reverse Charge", "reverse_charge"),
            ("Balance Due", "amount_due"),
            ("Amount Paid", "amount_paid"),
            ("Round Off", "round_off"),
            ("Freight", "freight_amount"),
        ],
    )
    def test_header_labels_resolve_to_their_field(self, wording, expected):
        assert best_match(wording).field == expected


class TestNonLabels:
    @pytest.mark.parametrize(
        "text",
        ["15000.00", "29ABCDE1234F1Z5", "14/08/2026", "Steel Fabrication Work", ""],
    )
    def test_values_and_prose_do_not_resolve_to_a_label(self, text):
        assert best_match(text) is None or best_match(text).score < 0.9


class TestCoverage:
    def test_every_field_has_at_least_one_label(self):
        for spec in lexicon.ALL_FIELDS:
            assert spec.labels, f"{spec.field} has no labels"

    def test_fields_are_ordered_most_specific_first(self):
        priorities = [spec.priority for spec in lexicon.ALL_FIELDS]
        assert priorities == sorted(priorities)

    def test_no_field_name_is_defined_twice(self):
        names = [spec.field for spec in lexicon.ALL_FIELDS]
        assert len(names) == len(set(names))

    def test_no_negative_disqualifies_its_own_field(self):
        """The bug this guards: `tax %` normalises to `tax`, which is a
        substring of every label `tax_amount` has — so the field disqualified
        itself and vanished from extraction entirely, with nothing failing."""
        for spec in lexicon.ALL_FIELDS:
            for label in spec.labels:
                assert match_label(label, spec) is not None, (
                    f"{spec.field} cannot match its own label {label!r} — "
                    f"check its negatives: {spec.negative}"
                )

    def test_a_subset_label_is_not_a_perfect_match(self):
        """Textbook token-set ratio scores any subset as 1.0, so `Invoice`
        matched `Invoice Total` perfectly and invoice numbers were extracted
        as invoice totals."""
        assert lexicon._similarity("invoice", "invoice total") < 0.85
        assert lexicon._similarity("total", "grand total") < 0.85

    def test_a_genuine_near_match_still_matches(self):
        # The fix must not make matching brittle: a mangled character or a
        # dropped letter still has nearly all its tokens in common.
        assert lexicon._similarity("invoice no", "lnvoice no") >= 0.85

    def test_every_money_field_carries_the_shared_negatives(self):
        """A money field that forgot the negatives will happily read
        "Amount in Words" as an amount."""
        for spec in lexicon.ALL_FIELDS:
            if spec.kind != "money":
                continue
            assert "amount in words" in spec.normalised_negatives, spec.field
