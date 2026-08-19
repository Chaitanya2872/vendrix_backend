from __future__ import annotations

from decimal import Decimal

from ..dto import ParsedInvoiceResult, ParsedParty
from .base_invoice_parser import BaseInvoiceParser
from .text_extraction import ExtractedDocument
from . import invoice_field_parser as fields
from .invoice_table_parser import parse_line_item_tables
from .money_utils import approx_equal

_TOLERANCE = Decimal("2.00")  # rounding tolerance for arithmetic validation


class GenericInvoiceParser(BaseInvoiceParser):
    """Format-agnostic parser that works purely from label matching and
    table-header detection. This is the only parser registered today; it is
    intentionally not tuned to any single vendor's layout."""

    name = "generic"
    version = "1.0"

    def can_parse(self, text: str) -> bool:
        # The generic parser is the fallback of last resort — it accepts
        # any document that has *some* text to work with.
        return bool(text and text.strip())

    def parse(self, extracted: ExtractedDocument) -> ParsedInvoiceResult:
        text = extracted.text
        result = ParsedInvoiceResult(parser_name=self.name, parser_version=self.version, used_ocr=extracted.used_ocr)

        result.invoice_number = fields.extract_invoice_number(text)
        result.invoice_date = fields.extract_invoice_date(text)
        result.due_date = fields.extract_due_date(text)
        result.purchase_order_number = fields.extract_po_number(text)
        result.purchase_order_date = fields.extract_po_date(text)
        result.currency = fields.extract_currency(text)
        result.place_of_supply = fields.extract_place_of_supply(text)
        result.payment_terms = fields.extract_payment_terms(text)

        amounts = fields.extract_header_amounts(text)
        for key, value in amounts.items():
            setattr(result, key, value)

        gst_breakdown = fields.extract_gst_breakdown(text)
        for key, value in gst_breakdown.items():
            setattr(result, key, value)

        vendor_dict, customer_dict, gstin_warnings = fields.extract_parties(text)
        result.vendor = ParsedParty(**vendor_dict)
        result.customer = ParsedParty(**customer_dict)
        for warning in gstin_warnings:
            result.add_warning(warning)

        line_items, table_warnings = parse_line_item_tables(extracted.tables_per_page)
        result.line_items = line_items
        for warning in table_warnings:
            result.add_warning(warning)

        self._derive_tax_amount_if_missing(result)
        self._validate(result)
        result.parsing_confidence = self._score_confidence(result)
        return result

    @staticmethod
    def _derive_tax_amount_if_missing(result: ParsedInvoiceResult) -> None:
        if result.tax_amount is None:
            components = [v for v in (result.cgst_amount, result.sgst_amount, result.igst_amount) if v is not None]
            if components:
                result.tax_amount = sum(components)

    @staticmethod
    def _validate(result: ParsedInvoiceResult) -> None:
        # Required fields
        if not result.invoice_number:
            result.add_validation_error("Invoice number could not be detected.")
        if not result.invoice_date:
            result.add_validation_error("Invoice date could not be detected.")
        if not result.vendor.name and not result.vendor.gstin:
            result.add_validation_error("Vendor could not be identified.")
        if result.total_amount is None:
            result.add_validation_error("Total amount could not be detected.")

        # Tax cross-check: CGST + SGST + IGST ≈ tax_amount
        components = [v for v in (result.cgst_amount, result.sgst_amount, result.igst_amount) if v is not None]
        if components and result.tax_amount is not None:
            if not approx_equal(sum(components), result.tax_amount, _TOLERANCE):
                result.add_warning(
                    "CGST + SGST + IGST does not closely match the extracted tax amount; please verify."
                )

        # Grand total cross-check: taxable_amount - discount + tax + round_off ≈ total
        if result.taxable_amount is not None and result.total_amount is not None:
            discount = result.discount_amount or Decimal("0")
            tax = result.tax_amount or Decimal("0")
            round_off = result.round_off or Decimal("0")
            expected_total = result.taxable_amount - discount + tax + round_off
            if not approx_equal(expected_total, result.total_amount, _TOLERANCE):
                result.add_warning(
                    "Taxable amount, tax and round-off do not closely reconcile to the total amount; please verify."
                )

        # Line item cross-check: quantity * unit_price ≈ taxable_value
        for item in result.line_items:
            if item.quantity is not None and item.unit_price is not None:
                expected = item.quantity * item.unit_price
                reference = item.taxable_value if item.taxable_value is not None else item.total_amount
                if reference is not None and not approx_equal(expected, reference, _TOLERANCE):
                    result.add_warning(
                        f"Line item '{item.description or 'unknown'}': quantity × unit price does not "
                        "closely match the line amount; please verify."
                    )

    @staticmethod
    def _score_confidence(result: ParsedInvoiceResult) -> float:
        score = 0
        if result.invoice_number:
            score += 15
        if result.invoice_date:
            score += 10
        if result.vendor.name:
            score += 10
        if result.vendor.gstin:
            score += 10
        if result.customer.name:
            score += 5
        if result.line_items:
            score += 15
        if result.subtotal is not None or result.taxable_amount is not None:
            score += 10
        if result.tax_amount is not None or any(
            v is not None for v in (result.cgst_amount, result.sgst_amount, result.igst_amount)
        ):
            score += 10
        if result.total_amount is not None:
            score += 10
        if not result.validation_errors:
            score += 5
        return round(min(score, 100) / 100, 2)
