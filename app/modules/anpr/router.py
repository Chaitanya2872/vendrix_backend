from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.common.dependencies import current_user
from app.db.session import get_db
from app.models import Vehicle, Vendor, User
from app.utils.validators import normalize_registration_number
from app.core.config import settings
from app.workers.vision import crop_plate, decode_image, read_text, registration_candidates
from app.modules.gate import service as gate_service
router = APIRouter(prefix="/anpr", tags=["ANPR"])


def gate_context(db: Session, plate: str) -> dict:
    """Whether this plate is currently inside, and under which entry.

    A plate read at the barrier only answers half the guard's question;
    the other half is whether the vehicle is arriving or leaving. Returning
    the open visit here lets one camera event drive either a gate-in or a
    gate-out without a second round trip."""
    open_entry = gate_service.open_visit(db, plate)
    return {
        "on_premises": open_entry is not None,
        "open_entry": gate_service.serialize(db, open_entry) if open_entry else None,
        "suggested_action": "GATE_OUT" if open_entry else "GATE_IN",
    }
@router.post("/lookup")
def lookup(registration_number: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    plate = normalize_registration_number(registration_number); vehicle = db.scalar(select(Vehicle).where(Vehicle.registration_number == plate))
    if not vehicle: raise HTTPException(404, "Vehicle not found in internal registry")
    return {"registration_number": plate, "vehicle": vehicle, "vendor": db.get(Vendor, vehicle.vendor_id), **gate_context(db, plate)}

@router.post("/recognize")
def recognize_plate(file: UploadFile = File(...), db: Session = Depends(get_db), _: User = Depends(current_user)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}: raise HTTPException(415, "Upload a vehicle image")
    image = crop_plate(decode_image(file.file.read()), settings.plate_detector_model_path)
    candidates = registration_candidates(read_text(image))
    if not candidates: raise HTTPException(422, "No registration number could be read")
    plate = normalize_registration_number(candidates[0]); vehicle = db.scalar(select(Vehicle).where(Vehicle.registration_number == plate))
    return {"registration_number": plate, "matched": vehicle is not None, "vehicle": vehicle, "vendor": db.get(Vendor, vehicle.vendor_id) if vehicle else None, "candidates": candidates, **gate_context(db, plate)}
