from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.common.dependencies import current_user
from app.db.session import get_db
from app.models import Delivery, Document, Invoice, User, Vendor
from app.modules.deliveries.router import serialize

router = APIRouter(prefix="/mobile", tags=["mobile"])


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), _: User = Depends(current_user)):
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    due = Invoice.due_date < now.date()
    pending = Invoice.status.in_(("PENDING", "PENDING_APPROVAL", "DRAFT"))
    active_delivery = Delivery.status.in_(("SCHEDULED", "ASSIGNED", "IN_TRANSIT"))
    upcoming = db.scalars(select(Delivery).where(Delivery.scheduled_at >= now).order_by(Delivery.scheduled_at).limit(5)).all()
    return {
        "vendors": db.scalar(select(func.count()).select_from(Vendor)) or 0,
        "active_vendors": db.scalar(select(func.count()).select_from(Vendor).where(Vendor.status == "ACTIVE")) or 0,
        "pending_invoices": db.scalar(select(func.count()).select_from(Invoice).where(pending)) or 0,
        "overdue_invoices": db.scalar(select(func.count()).select_from(Invoice).where(due, Invoice.status != "PAID")) or 0,
        "deliveries_today": db.scalar(select(func.count()).select_from(Delivery).where(Delivery.scheduled_at >= now, Delivery.scheduled_at < tomorrow)) or 0,
        "active_deliveries": db.scalar(select(func.count()).select_from(Delivery).where(active_delivery)) or 0,
        "documents_for_review": db.scalar(select(func.count()).select_from(Document).where(Document.status == "REVIEW_REQUIRED")) or 0,
        "upcoming_deliveries": [serialize(item, db) for item in upcoming],
    }
