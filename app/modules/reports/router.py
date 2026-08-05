from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.common.dependencies import require_roles
from app.db.session import get_db
from app.models import Approval, Invoice, Payment, User, Vehicle, Vendor
router = APIRouter(prefix="/reports", tags=["reports"])
@router.get("/summary")
def summary(db: Session = Depends(get_db), _: User = Depends(require_roles("ADMIN", "FINANCE", "APPROVER"))):
    return {"vendors": db.scalar(select(func.count()).select_from(Vendor)), "active_vehicles": db.scalar(select(func.count()).select_from(Vehicle).where(Vehicle.status == "ACTIVE")), "pending_approvals": db.scalar(select(func.count()).select_from(Approval).where(Approval.status == "PENDING")), "invoice_total": db.scalar(select(func.coalesce(func.sum(Invoice.amount), 0))), "paid_total": db.scalar(select(func.coalesce(func.sum(Payment.amount), 0)))}
