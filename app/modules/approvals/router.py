from datetime import datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import Approval, AuditLog, Invoice, User

router = APIRouter(prefix="/approvals", tags=["approvals"])
class Decision(BaseModel): decision: Literal["APPROVED", "REJECTED"]; comment: str | None = None

@router.get("")
def list_approvals(db: Session = Depends(get_db), _: User = Depends(require_roles("ADMIN", "APPROVER", "FINANCE"))):
    return db.scalars(select(Approval).order_by(Approval.created_at.desc())).all()

@router.post("/{approval_id}/decision")
def decide(approval_id: str, body: Decision, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "APPROVER", "FINANCE"))):
    approval = db.get(Approval, approval_id)
    if not approval: raise HTTPException(404, "Approval not found")
    if approval.status != "PENDING": raise HTTPException(409, "Approval is already decided")
    approval.status, approval.approver_id, approval.comment, approval.decided_at = body.decision, user.id, body.comment, datetime.now(timezone.utc)
    if approval.resource_type == "invoice":
        invoice = db.get(Invoice, approval.resource_id)
        if invoice: invoice.status = body.decision; db.add(AuditLog(actor_id=user.id, action=body.decision, resource_type="invoices", resource_id=invoice.id))
    db.commit(); db.refresh(approval); return approval
