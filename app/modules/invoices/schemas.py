"""Invoice pydantic schemas — EXTENDED for invoice-document parsing.

The original file (kept intact below) defines InvoiceCreate/InvoiceUpdate,
consumed by `attach_crud` in invoices/router.py. This adds:

  * InvoiceLineItemSchema / InvoiceLineItemCreate — the new line-item entity
  * ParsedInvoiceResponse — the shape returned alongside the Document when
    a document upload triggers parsing (see documents/router.py changes)

Nothing below removes or narrows the original two classes.
"""
from datetime import date
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field


class InvoiceCreate(BaseModel):
    vendor_id: str; invoice_number: str; invoice_date: date; due_date: date | None = None; amount: float = Field(gt=0); tax_amount: float = Field(default=0, ge=0); document_id: str | None = None


class InvoiceUpdate(BaseModel):
    vendor_id: str | None = None; invoice_number: str | None = None; invoice_date: date | None = None; due_date: date | None = None; amount: float | None = Field(default=None, gt=0); tax_amount: float | None = Field(default=None, ge=0); document_id: str | None = None; status: str | None = None


# --------------------------------------------------------------------------
# New: line items
# --------------------------------------------------------------------------
class InvoiceLineItemSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    description: str | None = None
    hsn_sac: str | None = None
    quantity: Decimal | None = None
    unit: str | None = None
    unit_price: Decimal | None = None
    discount: Decimal | None = None
    gst_rate: Decimal | None = None
    taxable_value: Decimal | None = None
    total_amount: Decimal | None = None


# --------------------------------------------------------------------------
# New: parsed-invoice response, returned from the document upload flow
# --------------------------------------------------------------------------
class ParsedInvoiceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    status: str  # "PARSED" | "REVIEW_REQUIRED" | existing workflow statuses
    invoice_number: str | None = None
    invoice_date: date | None = None
    vendor_name: str | None = None
    vendor_gstin: str | None = None
    customer_name: str | None = None
    customer_gstin: str | None = None
    subtotal: Decimal | None = None
    cgst_amount: Decimal | None = None
    sgst_amount: Decimal | None = None
    igst_amount: Decimal | None = None
    tax_amount: Decimal | None = None
    total_amount: Decimal | None = None
    parsing_confidence: Decimal | None = None
    line_items: list[InvoiceLineItemSchema] = []


class InvoiceParsingWarnings(BaseModel):
    duplicate: bool = False
    existing_invoice_id: str | None = None
    warnings: list[str] = []
    validation_errors: list[str] = []
