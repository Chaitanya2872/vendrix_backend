"""Invoice module service layer — EXTENDED for invoice-document parsing.

invoices/service.py and invoices/repository.py were empty in the current
codebase (the existing invoice CRUD goes through the generic `attach_crud`
helper directly from the router — see invoices/router.py). This file adds
the two functions the parsing pipeline needs, following the same
Session-based, no-ORM-leak-into-callers pattern used elsewhere
(documents/router.py commits/refreshes directly on the session it's given).

Only invoice_parser_service.py calls into this module for the parsing flow;
the router keeps using `attach_crud` for ordinary manual CRUD, so existing
behavior is untouched.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog, Document, Invoice
from app.modules.invoices.dto import ParsedInvoiceResult
from app.modules.invoices.models import InvoiceLineItem

# Secondary duplicate check tolerance: same vendor + invoice date + total
# amount within this many rupees is treated as "probably the same invoice"
# even without a matching GSTIN+number pair.
_DUPLICATE_AMOUNT_TOLERANCE = Decimal("1.00")
_DUPLICATE_DATE_WINDOW_DAYS = 0  # exact date match for the secondary rule


def find_probable_duplicate(db: Session, parsed: ParsedInvoiceResult) -> Invoice | None:
    """Primary rule: vendor GSTIN + invoice number.
    Secondary rule (only if the primary rule can't run, e.g. no GSTIN found):
    vendor name + invoice date + total amount.
    """
    if parsed.vendor.gstin and parsed.invoice_number:
        existing = db.scalars(
            select(Invoice).where(
                Invoice.vendor_gstin == parsed.vendor.gstin,
                Invoice.invoice_number == parsed.invoice_number,
            )
        ).first()
        if existing:
            return existing

    if parsed.vendor.name and parsed.invoice_date and parsed.total_amount is not None:
        candidates = db.scalars(
            select(Invoice).where(
                Invoice.vendor_name == parsed.vendor.name,
                Invoice.invoice_date == parsed.invoice_date,
            )
        ).all()
        for candidate in candidates:
            if candidate.total_amount is not None and abs(
                Decimal(candidate.total_amount) - parsed.total_amount
            ) <= _DUPLICATE_AMOUNT_TOLERANCE:
                return candidate

    return None


def create_from_parsed_document(
    db: Session,
    document: Document,
    parsed: ParsedInvoiceResult,
    status: str,
) -> Invoice:
    """Create an Invoice + its InvoiceLineItems from a parsed document,
    transactionally: if line-item creation fails, the whole thing rolls
    back and no partial invoice is left behind. The Document row itself is
    never touched or rolled back here — it was already committed by the
    upload endpoint and remains available regardless of parsing outcome.
    """
    try:
        invoice = Invoice(
            # Upload already records the selected vendor. Preserve that
            # ownership when creating the invoice; the column is required
            # and discarding it here made an otherwise successful parse fail.
            vendor_id=document.vendor_id,
            invoice_number=parsed.invoice_number,
            invoice_date=parsed.invoice_date,
            due_date=parsed.due_date,
            amount=parsed.total_amount or Decimal("0"),
            tax_amount=parsed.tax_amount or Decimal("0"),
            document_id=document.id,
            status=status,
            vendor_name=parsed.vendor.name,
            vendor_address=parsed.vendor.address,
            vendor_gstin=parsed.vendor.gstin,
            vendor_pan=parsed.vendor.pan,
            vendor_email=parsed.vendor.email,
            vendor_phone=parsed.vendor.phone,
            customer_name=parsed.customer.name,
            customer_address=parsed.customer.address,
            customer_gstin=parsed.customer.gstin,
            customer_pan=parsed.customer.pan,
            customer_email=parsed.customer.email,
            customer_phone=parsed.customer.phone,
            purchase_order_number=parsed.purchase_order_number,
            purchase_order_date=parsed.purchase_order_date,
            currency=parsed.currency or "INR",
            subtotal=parsed.subtotal,
            discount_amount=parsed.discount_amount,
            taxable_amount=parsed.taxable_amount,
            cgst_amount=parsed.cgst_amount,
            sgst_amount=parsed.sgst_amount,
            igst_amount=parsed.igst_amount,
            round_off=parsed.round_off,
            total_amount=parsed.total_amount,
            amount_paid=parsed.amount_paid,
            amount_due=parsed.amount_due,
            payment_terms=parsed.payment_terms,
            place_of_supply=parsed.place_of_supply,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            used_ocr=str(parsed.used_ocr),
            parsing_confidence=Decimal(str(parsed.parsing_confidence)),
            parsing_warnings=parsed.warnings,
            parsing_validation_errors=parsed.validation_errors,
        )
        db.add(invoice)
        db.flush()  # assign invoice.id without committing yet

        for item in parsed.line_items:
            db.add(
                InvoiceLineItem(
                    invoice_id=invoice.id,
                    description=item.description,
                    hsn_sac=item.hsn_sac,
                    quantity=item.quantity,
                    unit=item.unit,
                    unit_price=item.unit_price,
                    discount=item.discount,
                    gst_rate=item.gst_rate,
                    taxable_value=item.taxable_value,
                    total_amount=item.total_amount,
                )
            )

        db.add(
            AuditLog(
                actor_id=None,
                action="PARSE_INVOICE_DOCUMENT",
                resource_type="invoices",
                resource_id=invoice.id,
            )
        )
        db.commit()
        db.refresh(invoice)
        return invoice
    except Exception:
        db.rollback()  # invoice + line items roll back together; Document is unaffected
        raise
