from datetime import date
from pydantic import BaseModel
class DriverCreate(BaseModel):
    vendor_id: str; vehicle_id: str | None = None; full_name: str; phone: str; license_number: str; license_expiry: date | None = None; status: str = "ACTIVE"
class DriverUpdate(BaseModel):
    vendor_id: str | None = None; vehicle_id: str | None = None; full_name: str | None = None; phone: str | None = None; license_number: str | None = None; license_expiry: date | None = None; status: str | None = None
