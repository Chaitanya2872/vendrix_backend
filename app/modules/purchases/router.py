from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import AuditLog, Purchase, User, Vendor

router = APIRouter(prefix="/purchases", tags=["purchases"])


class PurchaseCreate(BaseModel):
    purchase_number: str = Field(min_length=1, max_length=80)
    vendor_id: str
    reference: str | None = Field(default=None, max_length=120)
    quantity: float = Field(default=0, ge=0)
    total_amount: float = Field(default=0, ge=0)
    expected_date: date | None = None
    status: str = "DRAFT"
    notes: str | None = None


class PurchaseUpdate(BaseModel):
    purchase_number: str | None = Field(default=None, min_length=1, max_length=80)
    vendor_id: str | None = None
    reference: str | None = Field(default=None, max_length=120)
    quantity: float | None = Field(default=None, ge=0)
    total_amount: float | None = Field(default=None, ge=0)
    expected_date: date | None = None
    status: str | None = None
    notes: str | None = None


def serialize(item: Purchase, db: Session) -> dict:
    vendor = db.get(Vendor, item.vendor_id)
    return {"id": item.id, "purchase_number": item.purchase_number, "vendor_id": item.vendor_id,
            "vendor_name": vendor.legal_name if vendor else None, "reference": item.reference,
            "quantity": item.quantity, "total_amount": item.total_amount, "expected_date": item.expected_date,
            "status": item.status, "notes": item.notes, "created_at": item.created_at}


@router.get("")
def list_purchases(limit: int = Query(50, le=200), offset: int = 0, status: str | None = None,
                   db: Session = Depends(get_db), _: User = Depends(current_user)):
    query = select(Purchase).order_by(Purchase.expected_date.asc().nulls_last(), Purchase.created_at.desc())
    count = select(func.count()).select_from(Purchase)
    if status:
        query = query.where(Purchase.status == status.upper()); count = count.where(Purchase.status == status.upper())
    items = db.scalars(query.offset(offset).limit(limit)).all()
    return {"items": [serialize(item, db) for item in items], "total": db.scalar(count) or 0, "limit": limit, "offset": offset}


@router.post("", status_code=201)
def create_purchase(body: PurchaseCreate, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "OPERATOR"))):
    if db.scalar(select(Purchase.id).where(Purchase.purchase_number == body.purchase_number)):
        raise HTTPException(409, "Purchase number already exists")
    if not db.get(Vendor, body.vendor_id): raise HTTPException(422, "Vendor not found")
    item = Purchase(**body.model_dump(), status=body.status.upper())
    db.add(item); db.flush(); db.add(AuditLog(actor_id=user.id, action="CREATE", resource_type="purchases", resource_id=item.id)); db.commit(); db.refresh(item)
    return serialize(item, db)


@router.get("/{purchase_id}")
def get_purchase(purchase_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    item = db.get(Purchase, purchase_id)
    if not item: raise HTTPException(404, "Purchase not found")
    return serialize(item, db)


@router.patch("/{purchase_id}")
def update_purchase(purchase_id: str, body: PurchaseUpdate, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "OPERATOR"))):
    item = db.get(Purchase, purchase_id)
    if not item: raise HTTPException(404, "Purchase not found")
    changes = body.model_dump(exclude_unset=True)
    if "purchase_number" in changes and changes["purchase_number"] != item.purchase_number and db.scalar(select(Purchase.id).where(Purchase.purchase_number == changes["purchase_number"])):
        raise HTTPException(409, "Purchase number already exists")
    if "vendor_id" in changes and changes["vendor_id"] and not db.get(Vendor, changes["vendor_id"]):
        raise HTTPException(422, "Vendor not found")
    for key, value in changes.items(): setattr(item, key, value.upper() if key == "status" and value else value)
    db.add(AuditLog(actor_id=user.id, action="UPDATE", resource_type="purchases", resource_id=item.id)); db.commit(); db.refresh(item)
    return serialize(item, db)
