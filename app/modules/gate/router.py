"""The gate register: inward and outward vehicle entries.

Mounted at /vehicle-entries. Reads are open to any authenticated user because
a stores clerk expecting a truck needs to see it arrive; writes are limited to
the roles that staff a gate.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import AuditLog, User, VehicleEntry
from app.modules.gate import service
from app.modules.gate.schemas import (
    DIRECTIONS,
    VehicleEntryCreate,
    VehicleEntryUpdate,
    VehicleExit,
)
from app.utils.validators import normalize_registration_number

router = APIRouter(prefix="/vehicle-entries", tags=["gate"])

GATE_ROLES = ("ADMIN", "OPERATOR", "SECURITY")


@router.get("")
def list_entries(
    direction: str | None = None,
    status: str | None = None,
    purpose: str | None = None,
    gate: str | None = None,
    vendor_id: str | None = None,
    vehicle_id: str | None = None,
    registration_number: str | None = None,
    # Windowed on entry_at: "what came through the gate yesterday" is the
    # question a shift handover asks, and an entry that has not been closed
    # yet still belongs in yesterday's list.
    entry_from: datetime | None = None,
    entry_to: datetime | None = None,
    q: str | None = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    filters = []
    if direction:
        filters.append(VehicleEntry.direction == direction.upper())
    if status:
        filters.append(VehicleEntry.status == status.upper())
    if purpose:
        filters.append(VehicleEntry.purpose == purpose.upper())
    if gate:
        filters.append(VehicleEntry.gate == gate)
    if vendor_id:
        filters.append(VehicleEntry.vendor_id == vendor_id)
    if vehicle_id:
        filters.append(VehicleEntry.vehicle_id == vehicle_id)
    if registration_number:
        # Matched on the raw text rather than through the normaliser: a guard
        # searching a partial plate would otherwise be rejected for length.
        plate = "".join(ch for ch in registration_number.upper() if ch.isalnum())
        filters.append(VehicleEntry.registration_number.contains(plate))
    if entry_from:
        filters.append(VehicleEntry.entry_at >= entry_from)
    if entry_to:
        filters.append(VehicleEntry.entry_at <= entry_to)
    if q:
        needle = f"%{q.strip().lower()}%"
        filters.append(
            or_(
                func.lower(VehicleEntry.entry_number).like(needle),
                func.lower(VehicleEntry.registration_number).like(needle),
                func.lower(func.coalesce(VehicleEntry.driver_name, "")).like(needle),
                func.lower(func.coalesce(VehicleEntry.document_reference, "")).like(needle),
                func.lower(func.coalesce(VehicleEntry.material_description, "")).like(needle),
            )
        )

    query = select(VehicleEntry).where(*filters).order_by(VehicleEntry.entry_at.desc())
    total = db.scalar(select(func.count()).select_from(VehicleEntry).where(*filters))
    items = db.scalars(query.offset(offset).limit(limit)).all()
    return {
        "items": [service.serialize(db, item) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/on-premises")
def on_premises(
    gate: str | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    """Every vehicle currently inside, oldest first.

    Its own endpoint rather than a filter preset because it is the question
    the gate screen refreshes on a loop, and because "oldest first" is the
    order that matters here — the vehicle that has been inside longest is the
    one somebody needs to chase.
    """
    filters = [VehicleEntry.status == service.OPEN_STATUS, VehicleEntry.exit_at.is_(None)]
    if gate:
        filters.append(VehicleEntry.gate == gate)
    items = db.scalars(select(VehicleEntry).where(*filters).order_by(VehicleEntry.entry_at.asc())).all()
    return {"items": [service.serialize(db, item) for item in items], "total": len(items)}


@router.get("/summary")
def summary(db: Session = Depends(get_db), _: User = Depends(current_user)):
    """Counts for the gate dashboard, scoped to the current UTC day."""
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today = VehicleEntry.entry_at >= day_start

    def count(*where) -> int:
        return db.scalar(select(func.count()).select_from(VehicleEntry).where(*where)) or 0

    return {
        "on_premises": count(VehicleEntry.status == service.OPEN_STATUS, VehicleEntry.exit_at.is_(None)),
        "inward_today": count(today, VehicleEntry.direction == "INWARD"),
        "outward_today": count(today, VehicleEntry.direction == "OUTWARD"),
        "completed_today": count(today, VehicleEntry.status == service.CLOSED_STATUS),
        # Vehicles nobody has onboarded. A non-zero number here is a
        # compliance finding, not a display detail.
        "unregistered_on_premises": count(
            VehicleEntry.status == service.OPEN_STATUS,
            VehicleEntry.exit_at.is_(None),
            VehicleEntry.vehicle_id.is_(None),
        ),
    }


@router.post("", status_code=201)
def create_entry(
    body: VehicleEntryCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*GATE_ROLES)),
):
    """Record a gate-in.

    The same endpoint serves both directions; `direction` says whether the
    visit is bringing material in or taking it out.
    """
    values = body.model_dump()
    plate, vehicle = service.resolve_vehicle(db, values.pop("registration_number", None), values.get("vehicle_id"))

    existing = service.open_visit(db, plate)
    if existing:
        raise HTTPException(
            409,
            f"{plate} is already on the premises under {existing.entry_number}; sign it out before recording a new entry",
        )

    service.check_references(db, values)
    gross = values.pop("gross_weight", None)
    tare = values.pop("tare_weight", None)
    entry_at = service.aware(values.pop("entry_at", None)) or datetime.now(timezone.utc)

    entry = VehicleEntry(
        **values,
        entry_number=service.next_entry_number(db),
        registration_number=plate,
        entry_at=entry_at,
        status=service.OPEN_STATUS,
        recorded_by=user.id,
    )
    # A plate the fleet registry knows brings its vendor with it, so the guard
    # does not have to pick one the system already knows the answer to.
    if vehicle:
        entry.vehicle_id = vehicle.id
        entry.vendor_id = entry.vendor_id or vehicle.vendor_id
    service.apply_weights(entry, gross, tare)

    db.add(entry)
    db.flush()
    db.add(
        AuditLog(
            actor_id=user.id,
            action="GATE_IN",
            resource_type="vehicle_entries",
            resource_id=entry.id,
            details={"registration_number": plate, "direction": entry.direction},
        )
    )
    db.commit()
    db.refresh(entry)
    return service.serialize(db, entry)


@router.get("/{entry_id}")
def get_entry(entry_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    entry = db.get(VehicleEntry, entry_id)
    if not entry:
        raise HTTPException(404, "Vehicle entry not found")
    return service.serialize(db, entry)


@router.patch("/{entry_id}")
def update_entry(
    entry_id: str,
    body: VehicleEntryUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*GATE_ROLES)),
):
    """Correct a visit — a mistyped challan number, a vendor picked late, or
    a cancellation of an entry that was opened twice."""
    entry = db.get(VehicleEntry, entry_id)
    if not entry:
        raise HTTPException(404, "Vehicle entry not found")

    changes = body.model_dump(exclude_unset=True)
    service.check_references(db, changes)
    # Weights go through apply_weights so the net is recomputed from whatever
    # the two readings now are, rather than being set field by field.
    gross = changes.pop("gross_weight", None)
    tare = changes.pop("tare_weight", None)

    if "vehicle_id" in changes and changes["vehicle_id"]:
        plate, vehicle = service.resolve_vehicle(db, None, changes["vehicle_id"])
        conflict = service.open_visit(db, plate, exclude_id=entry.id)
        if conflict and entry.exit_at is None:
            raise HTTPException(409, f"{plate} is already on the premises under {conflict.entry_number}")
        entry.registration_number = plate
        changes.setdefault("vendor_id", entry.vendor_id or (vehicle.vendor_id if vehicle else None))
    if "entry_at" in changes and changes["entry_at"]:
        stamped = service.aware(changes["entry_at"])
        if entry.exit_at is not None and stamped > service.aware(entry.exit_at):
            raise HTTPException(422, "Entry time cannot be later than exit time")
        changes["entry_at"] = stamped

    for key, value in changes.items():
        setattr(entry, key, value)
    service.apply_weights(entry, gross, tare)

    db.add(
        AuditLog(
            actor_id=user.id,
            action="UPDATE",
            resource_type="vehicle_entries",
            resource_id=entry.id,
            details={"fields": sorted(changes)},
        )
    )
    db.commit()
    db.refresh(entry)
    return service.serialize(db, entry)


@router.post("/{entry_id}/exit")
def record_exit(
    entry_id: str,
    body: VehicleExit,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*GATE_ROLES)),
):
    """Sign a vehicle out, closing the visit.

    Separate from PATCH so the guard who let the vehicle out is recorded
    against the act of letting it out, and so a double sign-out is a 409
    rather than a silently overwritten timestamp.
    """
    entry = db.get(VehicleEntry, entry_id)
    if not entry:
        raise HTTPException(404, "Vehicle entry not found")

    service.close_visit(entry, body.exit_at, user.id)
    service.apply_weights(entry, body.gross_weight, body.tare_weight)
    if body.remarks:
        entry.remarks = f"{entry.remarks}\n{body.remarks}" if entry.remarks else body.remarks

    db.add(
        AuditLog(
            actor_id=user.id,
            action="GATE_OUT",
            resource_type="vehicle_entries",
            resource_id=entry.id,
            details={
                "registration_number": entry.registration_number,
                "duration_minutes": service.duration_minutes(entry),
            },
        )
    )
    db.commit()
    db.refresh(entry)
    return service.serialize(db, entry)


@router.get("/vehicle/{registration_number}/history")
def vehicle_history(
    registration_number: str,
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    """Every recorded visit for one plate, newest first.

    The gate screen calls this the moment a plate is typed or read by the
    camera: whether this truck has been here before, and whether it left last
    time, is what the guard decides on.
    """
    plate = normalize_registration_number(registration_number)
    items = db.scalars(
        select(VehicleEntry)
        .where(VehicleEntry.registration_number == plate)
        .order_by(VehicleEntry.entry_at.desc())
        .limit(limit)
    ).all()
    open_entry = next((item for item in items if item.status == service.OPEN_STATUS and item.exit_at is None), None)
    return {
        "registration_number": plate,
        "on_premises": open_entry is not None,
        "open_entry": service.serialize(db, open_entry) if open_entry else None,
        "items": [service.serialize(db, item) for item in items],
        "total": len(items),
        "directions": list(DIRECTIONS),
    }
