from datetime import date
from pydantic import BaseModel

class VehicleCreate(BaseModel):
    vendor_id: str; registration_number: str; vehicle_type: str; make: str | None = None; model: str | None = None; rc_expiry: date | None = None; insurance_expiry: date | None = None; status: str = "ACTIVE"
class VehicleUpdate(BaseModel):
    vendor_id: str | None = None; registration_number: str | None = None; vehicle_type: str | None = None; make: str | None = None; model: str | None = None; rc_expiry: date | None = None; insurance_expiry: date | None = None; status: str | None = None
