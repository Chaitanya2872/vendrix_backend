"""Internal data-transfer objects for the invoice parsing pipeline.

These are deliberately plain dataclasses (not pydantic models) because they
never cross a process/API boundary as-is — invoice_parser_service.py maps
them onto the existing Invoice / InvoiceLineItem ORM models before anything
is persisted or returned to a client. Keeping them separate from schemas.py
avoids conflating "what the parser produced" with "what the API contract is".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class ParsedParty:
    """A vendor or customer extracted from the document. Any field may be
    None — the parser never fabricates a value it didn't find evidence for."""
    name: str | None = None
    address: str | None = None
    gstin: str | None = None
    pan: str | None = None
    email: str | None = None
    phone: str | None = None


@dataclass
class ParsedLineItem:
    description: str | None = None
    hsn_sac: str | None = None
    quantity: Decimal | None = None
    unit: str | None = None
    unit_price: Decimal | None = None
    discount: Decimal | None = None
    gst_rate: Decimal | None = None
    taxable_value: Decimal | None = None
    total_amount: Decimal | None = None
    raw_row: list[str] = field(default_factory=list)


@dataclass
class ParsedInvoiceResult:
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    purchase_order_number: str | None = None
    purchase_order_date: date | None = None

    currency: str | None = None

    subtotal: Decimal | None = None
    discount_amount: Decimal | None = None
    taxable_amount: Decimal | None = None

    cgst_amount: Decimal | None = None
    sgst_amount: Decimal | None = None
    igst_amount: Decimal | None = None
    tax_amount: Decimal | None = None
    round_off: Decimal | None = None
    total_amount: Decimal | None = None

    amount_paid: Decimal | None = None
    amount_due: Decimal | None = None

    payment_terms: str | None = None
    place_of_supply: str | None = None

    vendor: ParsedParty = field(default_factory=ParsedParty)
    customer: ParsedParty = field(default_factory=ParsedParty)

    line_items: list[ParsedLineItem] = field(default_factory=list)

    parser_name: str = "generic"
    parser_version: str = "1.0"
    used_ocr: bool = False
    parsing_confidence: float = 0.0

    warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add_validation_error(self, message: str) -> None:
        if message not in self.validation_errors:
            self.validation_errors.append(message)
