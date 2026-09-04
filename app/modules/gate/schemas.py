"""Request bodies for the gate register.

Directions, purposes and statuses are constrained here rather than in the
database because they are API vocabulary: a bad value is a client bug worth
a 422, not a row worth storing.
"""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

DIRECTIONS = ("INWARD", "OUTWARD")
PURPOSES = ("DELIVERY", "PICKUP", "SERVICE", "TRANSFER", "VISITOR", "OTHER")
STATUSES = ("IN_PREMISES", "COMPLETED", "CANCELLED")
CAPTURE_METHODS = ("MANUAL", "ANPR")


def _upper(value: str | None) -> str | None:
    return value.upper().strip() if value else value


class VehicleEntryCreate(BaseModel):
    direction: str
    registration_number: str = Field(min_length=4, max_length=20)
    purpose: str = "DELIVERY"
    vehicle_id: str | None = None
    vendor_id: str | None = None
    driver_id: str | None = None
    driver_name: str | None = Field(default=None, max_length=150)
    driver_phone: str | None = Field(default=None, max_length=30)
    purchase_id: str | None = None
    delivery_id: str | None = None
    gate: str | None = Field(default=None, max_length=60)
    material_description: str | None = None
    document_reference: str | None = Field(default=None, max_length=120)
    gross_weight: float | None = Field(default=None, ge=0)
    tare_weight: float | None = Field(default=None, ge=0)
    # Optional so a guard can back-date a visit they wrote on paper during a
    # network outage; defaults to the moment the request is handled.
    entry_at: datetime | None = None
    capture_method: str = "MANUAL"
    remarks: str | None = None

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str) -> str:
        if _upper(value) not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
        return _upper(value)

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str) -> str:
        if _upper(value) not in PURPOSES:
            raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}")
        return _upper(value)

    @field_validator("capture_method")
    @classmethod
    def _capture(cls, value: str) -> str:
        if _upper(value) not in CAPTURE_METHODS:
            raise ValueError(f"capture_method must be one of {', '.join(CAPTURE_METHODS)}")
        return _upper(value)


class VehicleEntryUpdate(BaseModel):
    """Corrections to an open or closed visit.

    Deliberately excludes `exit_at` and `entry_number`: closing a visit goes
    through the exit endpoint so the guard who closed it is recorded, and the
    number is the system's own identifier.
    """

    direction: str | None = None
    purpose: str | None = None
    status: str | None = None
    vendor_id: str | None = None
    vehicle_id: str | None = None
    driver_id: str | None = None
    driver_name: str | None = Field(default=None, max_length=150)
    driver_phone: str | None = Field(default=None, max_length=30)
    purchase_id: str | None = None
    delivery_id: str | None = None
    gate: str | None = Field(default=None, max_length=60)
    material_description: str | None = None
    document_reference: str | None = Field(default=None, max_length=120)
    gross_weight: float | None = Field(default=None, ge=0)
    tare_weight: float | None = Field(default=None, ge=0)
    entry_at: datetime | None = None
    remarks: str | None = None

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str | None) -> str | None:
        if value and _upper(value) not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
        return _upper(value)

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str | None) -> str | None:
        if value and _upper(value) not in PURPOSES:
            raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}")
        return _upper(value)

    @field_validator("status")
    @classmethod
    def _status(cls, value: str | None) -> str | None:
        if value and _upper(value) not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        return _upper(value)


class VehicleExit(BaseModel):
    """Closing half of a visit."""

    exit_at: datetime | None = None
    gross_weight: float | None = Field(default=None, ge=0)
    tare_weight: float | None = Field(default=None, ge=0)
    remarks: str | None = None
