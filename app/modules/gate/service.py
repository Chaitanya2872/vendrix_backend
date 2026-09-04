"""Gate-register rules: plate resolution, open-visit guarding, weighbridge maths.

Kept out of the router because every one of these is a statement about the
domain that has to hold no matter who is calling — the web form, the ANPR
camera or a future turnstile integration.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Delivery, Driver, Purchase, Vehicle, VehicleEntry, Vendor
from app.modules.documents.repository import next_document_number
from app.utils.validators import normalize_registration_number

OPEN_STATUS = "IN_PREMISES"
CLOSED_STATUS = "COMPLETED"
CANCELLED_STATUS = "CANCELLED"


def aware(value: datetime | None) -> datetime | None:
    """Treat a stored naive timestamp as UTC.

    SQLite hands back naive datetimes even for DateTime(timezone=True)
    columns, so comparing a freshly parsed request timestamp against a stored
    one raises TypeError on SQLite and works on PostgreSQL. Normalising both
    sides is the only version that behaves the same on each.
    """
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def next_entry_number(db: Session) -> str:
    """GE-2026-000001, from the same locked per-year counter the document
    numbers use — gate passes are read out over a radio, so they need the
    same gap-free, human-sayable shape."""
    return next_document_number(db, prefix="GE")


def resolve_vehicle(
    db: Session, registration_number: str | None, vehicle_id: str | None
) -> tuple[str, Vehicle | None]:
    """Settle on one plate and, when we know it, the fleet record behind it.

    A registered vehicle_id wins over typed text: if the guard picked the
    truck from the list, the list is the authority on how its plate is
    spelled. An unrecognised plate is not an error — see VehicleEntry.
    """
    if vehicle_id:
        vehicle = db.get(Vehicle, vehicle_id)
        if not vehicle:
            raise HTTPException(422, "Vehicle not found")
        return vehicle.registration_number, vehicle
    if not registration_number:
        raise HTTPException(422, "A registration number or a registered vehicle is required")
    plate = normalize_registration_number(registration_number)
    return plate, db.scalar(select(Vehicle).where(Vehicle.registration_number == plate))


def open_visit(db: Session, registration_number: str, exclude_id: str | None = None) -> VehicleEntry | None:
    """The visit this plate has not yet been signed out of, if any."""
    query = select(VehicleEntry).where(
        VehicleEntry.registration_number == registration_number,
        VehicleEntry.status == OPEN_STATUS,
        VehicleEntry.exit_at.is_(None),
    )
    if exclude_id:
        query = query.where(VehicleEntry.id != exclude_id)
    return db.scalars(query.order_by(VehicleEntry.entry_at.desc())).first()


def check_references(db: Session, values: dict) -> None:
    """Reject links to rows that are not there.

    Done up front and as a batch so a form with two bad ids does not need two
    round trips to discover that.
    """
    for field, model, label in (
        ("vendor_id", Vendor, "Vendor"),
        ("driver_id", Driver, "Driver"),
        ("purchase_id", Purchase, "Purchase"),
        ("delivery_id", Delivery, "Delivery"),
    ):
        identifier = values.get(field)
        if identifier and not db.get(model, identifier):
            raise HTTPException(422, f"{label} not found")


def apply_weights(entry: VehicleEntry, gross: float | None, tare: float | None) -> None:
    """Record whichever readings were taken and derive the net.

    Both readings rarely arrive together: an inward vehicle is weighed loaded
    on the way in and empty on the way out, so net_weight stays null until
    the second reading exists. Gross below tare means the two were entered
    the wrong way round — worth refusing, because a negative net silently
    understates a receipt.
    """
    if gross is not None:
        entry.gross_weight = gross
    if tare is not None:
        entry.tare_weight = tare
    if entry.gross_weight is None or entry.tare_weight is None:
        entry.net_weight = None
        return
    if float(entry.gross_weight) < float(entry.tare_weight):
        raise HTTPException(422, "Gross weight cannot be lower than tare weight")
    entry.net_weight = float(entry.gross_weight) - float(entry.tare_weight)


def close_visit(entry: VehicleEntry, exit_at: datetime | None, actor_id: str | None) -> None:
    """Sign a vehicle out of the site."""
    if entry.status == CANCELLED_STATUS:
        raise HTTPException(409, "This entry was cancelled and cannot be closed")
    if entry.exit_at is not None:
        raise HTTPException(409, "This vehicle has already been signed out")
    stamped = aware(exit_at) or datetime.now(timezone.utc)
    if stamped < aware(entry.entry_at):
        raise HTTPException(422, "Exit time cannot be earlier than entry time")
    entry.exit_at = stamped
    entry.status = CLOSED_STATUS
    entry.exit_recorded_by = actor_id


def duration_minutes(entry: VehicleEntry) -> int | None:
    """How long the vehicle was on site; null while it is still inside."""
    if entry.exit_at is None:
        return None
    return int((aware(entry.exit_at) - aware(entry.entry_at)).total_seconds() // 60)


def serialize(db: Session, entry: VehicleEntry) -> dict:
    """One visit, with the names a gate screen shows instead of ids."""
    vehicle = db.get(Vehicle, entry.vehicle_id) if entry.vehicle_id else None
    vendor = db.get(Vendor, entry.vendor_id) if entry.vendor_id else None
    driver = db.get(Driver, entry.driver_id) if entry.driver_id else None
    purchase = db.get(Purchase, entry.purchase_id) if entry.purchase_id else None
    delivery = db.get(Delivery, entry.delivery_id) if entry.delivery_id else None
    return {
        "id": entry.id,
        "entry_number": entry.entry_number,
        "direction": entry.direction,
        "status": entry.status,
        "purpose": entry.purpose,
        "registration_number": entry.registration_number,
        "vehicle_id": entry.vehicle_id,
        "vehicle_type": vehicle.vehicle_type if vehicle else None,
        # False for a plate the fleet registry has never seen. The gate screen
        # flags these rather than hiding them: an unregistered vehicle on site
        # is exactly what a security review wants to find.
        "vehicle_registered": vehicle is not None,
        "vendor_id": entry.vendor_id,
        "vendor_name": vendor.legal_name if vendor else None,
        "driver_id": entry.driver_id,
        "driver_name": entry.driver_name or (driver.full_name if driver else None),
        "driver_phone": entry.driver_phone or (driver.phone if driver else None),
        "purchase_id": entry.purchase_id,
        "purchase_number": purchase.purchase_number if purchase else None,
        "delivery_id": entry.delivery_id,
        "delivery_number": delivery.delivery_number if delivery else None,
        "gate": entry.gate,
        "material_description": entry.material_description,
        "document_reference": entry.document_reference,
        "gross_weight": float(entry.gross_weight) if entry.gross_weight is not None else None,
        "tare_weight": float(entry.tare_weight) if entry.tare_weight is not None else None,
        "net_weight": float(entry.net_weight) if entry.net_weight is not None else None,
        "entry_at": entry.entry_at,
        "exit_at": entry.exit_at,
        "duration_minutes": duration_minutes(entry),
        "capture_method": entry.capture_method,
        "recorded_by": entry.recorded_by,
        "exit_recorded_by": entry.exit_recorded_by,
        "remarks": entry.remarks,
        "created_at": entry.created_at,
    }
