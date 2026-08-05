from datetime import date
from pydantic import BaseModel, Field
class InvoiceCreate(BaseModel):
    vendor_id: str; invoice_number: str; invoice_date: date; due_date: date | None = None; amount: float = Field(gt=0); tax_amount: float = Field(default=0, ge=0); document_id: str | None = None
class InvoiceUpdate(BaseModel):
    vendor_id: str | None = None; invoice_number: str | None = None; invoice_date: date | None = None; due_date: date | None = None; amount: float | None = Field(default=None, gt=0); tax_amount: float | None = Field(default=None, ge=0); document_id: str | None = None; status: str | None = None
