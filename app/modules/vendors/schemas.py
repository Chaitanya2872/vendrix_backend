from typing import Any
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class VendorCreate(BaseModel):
    vendor_code: str = Field(min_length=2, max_length=30)
    legal_name: str = Field(min_length=2, max_length=200)
    gstin: str | None = Field(default=None, max_length=15)
    category: str | None = None
    status: str = "DRAFT"
    phone: str | None = None
    email: EmailStr | None = None
    address: dict[str, Any] | None = None
    bank_details: dict[str, Any] | None = None


class VendorUpdate(BaseModel):
    legal_name: str | None = Field(default=None, min_length=2, max_length=200)
    gstin: str | None = Field(default=None, max_length=15)
    category: str | None = None
    status: str | None = None
    phone: str | None = None
    email: EmailStr | None = None
    address: dict[str, Any] | None = None
    bank_details: dict[str, Any] | None = None


class VendorRead(VendorCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
