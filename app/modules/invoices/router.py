from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.common.dependencies import current_user
from app.db.session import get_db
from app.models import Approval, AuditLog, Invoice, User
from app.modules.crud import attach_crud
from app.modules.invoices.schemas import InvoiceCreate, InvoiceUpdate
router = APIRouter(prefix="/invoices", tags=["invoices"])
attach_crud(router, Invoice, InvoiceCreate, InvoiceUpdate, write_roles=("ADMIN", "OPERATOR", "FINANCE"))

@router.post("/{invoice_id}/submit")
def submit_invoice(invoice_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    invoice = db.get(Invoice, invoice_id)
    if not invoice: raise HTTPException(404, "Invoice not found")
    if invoice.status != "DRAFT": raise HTTPException(409, "Only draft invoices can be submitted")
    invoice.status = "PENDING_APPROVAL"; approval = Approval(resource_type="invoice", resource_id=invoice.id, requested_by=user.id)
    db.add(approval); db.add(AuditLog(actor_id=user.id, action="SUBMIT", resource_type="invoices", resource_id=invoice.id)); db.commit(); db.refresh(approval)
    return {"invoice": invoice, "approval_id": approval.id}
