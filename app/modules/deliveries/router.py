from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import AuditLog, Delivery, Driver, User, Vehicle, Vendor

router = APIRouter(prefix="/deliveries", tags=["deliveries"])


class DeliveryCreate(BaseModel):
    delivery_number: str = Field(min_length=1, max_length=80)
    vendor_id: str
    vehicle_id: str | None = None
    driver_id: str | None = None
    destination: str = Field(min_length=1, max_length=250)
    scheduled_at: datetime
    status: str = "SCHEDULED"
    notes: str | None = None


def serialize(delivery: Delivery, db: Session) -> dict:
    vendor = db.get(Vendor, delivery.vendor_id)
    vehicle = db.get(Vehicle, delivery.vehicle_id) if delivery.vehicle_id else None
    driver = db.get(Driver, delivery.driver_id) if delivery.driver_id else None
    return {
        "id": delivery.id, "delivery_number": delivery.delivery_number,
        "vendor_id": delivery.vendor_id, "vendor_name": vendor.legal_name if vendor else None,
        "vehicle_id": delivery.vehicle_id, "vehicle_number": vehicle.registration_number if vehicle else None,
        "driver_id": delivery.driver_id, "driver_name": driver.full_name if driver else None,
        "driver_phone": driver.phone if driver else None, "destination": delivery.destination,
        "scheduled_at": delivery.scheduled_at, "status": delivery.status, "notes": delivery.notes,
        "created_at": delivery.created_at,
    }


@router.get("")
def list_deliveries(
    limit: int = Query(50, le=200), offset: int = 0, status: str | None = None,
    db: Session = Depends(get_db), _: User = Depends(current_user),
):
    query = select(Delivery).order_by(Delivery.scheduled_at.asc())
    count_query = select(func.count()).select_from(Delivery)
    if status:
        query = query.where(Delivery.status == status.upper())
        count_query = count_query.where(Delivery.status == status.upper())
    items = db.scalars(query.offset(offset).limit(limit)).all()
    return {"items": [serialize(item, db) for item in items], "total": db.scalar(count_query), "limit": limit, "offset": offset}


@router.post("", status_code=201)
def create_delivery(body: DeliveryCreate, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "OPERATOR"))):
    if db.scalar(select(Delivery.id).where(Delivery.delivery_number == body.delivery_number)):
        raise HTTPException(409, "Delivery number already exists")
    if not db.get(Vendor, body.vendor_id):
        raise HTTPException(422, "Vendor not found")
    item = Delivery(**body.model_dump())
    db.add(item); db.flush()
    db.add(AuditLog(actor_id=user.id, action="CREATE", resource_type="deliveries", resource_id=item.id))
    db.commit(); db.refresh(item)
    return serialize(item, db)


@router.get("/{delivery_id}")
def get_delivery(delivery_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    item = db.get(Delivery, delivery_id)
    if not item: raise HTTPException(404, "Delivery not found")
    return serialize(item, db)
