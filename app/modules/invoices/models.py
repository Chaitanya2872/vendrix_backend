"""Invoice ORM model — EXTENDED for invoice-document parsing.

The existing Invoice model (inferred from invoices/schemas.py and
invoices/router.py, which use `attach_crud` + InvoiceCreate/InvoiceUpdate)
already has: id, vendor_id, invoice_number, invoice_date, due_date, amount,
tax_amount, document_id, status.

This file shows the ADDITIVE columns needed to support parsed-from-document
invoices, plus the new InvoiceLineItem entity (which did not exist before —
repository.py/models.py for both modules were empty). Merge these onto the
real Invoice class in-place; do not create a second Invoice table.

`document_id` already exists as the Invoice ↔ Document link (per
InvoiceCreate/InvoiceUpdate schemas) — that relationship is reused as-is,
per the "do not copy invoice data into the document table" rule. No new
Document columns are needed.
"""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import relationship

# NOTE: `Base`, common id/timestamp mixins, and the existing Invoice class
# itself live in whatever base module the rest of the app already uses
# (e.g. app.db.base / app.models). Import from there instead of redefining —
# shown here as a plain reference so the diff is self-contained to read.
from app.db.base import Base
from app.models import IdMixin


# --------------------------------------------------------------------------
# Additive columns to merge onto the EXISTING Invoice class.
# --------------------------------------------------------------------------
INVOICE_ADDITIVE_COLUMNS = """
    # --- vendor / customer detail (vendor_id FK is reused as-is) ---
    vendor_name = Column(String(255), nullable=True)
    vendor_address = Column(Text, nullable=True)
    vendor_gstin = Column(String(15), nullable=True, index=True)
    vendor_pan = Column(String(10), nullable=True)
    vendor_email = Column(String(255), nullable=True)
    vendor_phone = Column(String(32), nullable=True)

    customer_name = Column(String(255), nullable=True)
    customer_address = Column(Text, nullable=True)
    customer_gstin = Column(String(15), nullable=True, index=True)
    customer_pan = Column(String(10), nullable=True)
    customer_email = Column(String(255), nullable=True)
    customer_phone = Column(String(32), nullable=True)

    # --- header fields not already present ---
    purchase_order_number = Column(String(64), nullable=True)
    purchase_order_date = Column(Date, nullable=True)
    currency = Column(String(8), nullable=True, default="INR")
    subtotal = Column(Numeric(14, 2), nullable=True)
    discount_amount = Column(Numeric(14, 2), nullable=True)
    taxable_amount = Column(Numeric(14, 2), nullable=True)
    cgst_amount = Column(Numeric(14, 2), nullable=True)
    sgst_amount = Column(Numeric(14, 2), nullable=True)
    igst_amount = Column(Numeric(14, 2), nullable=True)
    round_off = Column(Numeric(14, 2), nullable=True)
    total_amount = Column(Numeric(14, 2), nullable=True)  # `amount` already exists; total_amount is the
                                                           # parsed grand total shown to the reviewer —
                                                           # reconcile/rename to `amount` if the team
                                                           # prefers a single canonical column.
    amount_paid = Column(Numeric(14, 2), nullable=True)
    amount_due = Column(Numeric(14, 2), nullable=True)
    payment_terms = Column(String(128), nullable=True)
    place_of_supply = Column(String(128), nullable=True)

    # --- parsing metadata ---
    parser_name = Column(String(64), nullable=True)
    parser_version = Column(String(16), nullable=True)
    used_ocr = Column(String(8), nullable=True)  # store as bool if the project uses native Boolean elsewhere
    parsing_confidence = Column(Numeric(4, 2), nullable=True)
    parsing_warnings = Column(JSONB, nullable=True)       # list[str]
    parsing_validation_errors = Column(JSONB, nullable=True)  # list[str]

    line_items = relationship(
        "InvoiceLineItem", back_populates="invoice", cascade="all, delete-orphan"
    )
"""
# `status` already exists on Invoice per InvoiceUpdate.status; this feature
# reuses it with two additional values: "PARSED" and "REVIEW_REQUIRED",
# alongside whatever values the workflow already defines (e.g. DRAFT,
# PENDING_APPROVAL — see invoices/router.py's submit_invoice). No new status
# enum/model is introduced.


class InvoiceLineItem(IdMixin, Base):
    """New entity — line items did not exist in the prior schema."""
    __tablename__ = "invoice_line_items"

    invoice_id = Column(String, ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True)

    description = Column(Text, nullable=True)
    hsn_sac = Column(String(16), nullable=True)
    quantity = Column(Numeric(14, 3), nullable=True)
    unit = Column(String(16), nullable=True)
    unit_price = Column(Numeric(14, 2), nullable=True)
    discount = Column(Numeric(14, 2), nullable=True)
    gst_rate = Column(Numeric(5, 2), nullable=True)
    taxable_value = Column(Numeric(14, 2), nullable=True)
    total_amount = Column(Numeric(14, 2), nullable=True)

    invoice = relationship("Invoice", back_populates="line_items")
