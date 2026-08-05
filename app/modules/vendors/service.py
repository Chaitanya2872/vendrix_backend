from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.models import AuditLog, Vendor
from app.modules.vendors.repository import VendorRepository
from app.modules.vendors.schemas import VendorCreate, VendorUpdate


class VendorService:
    def __init__(self, repository: VendorRepository | None = None): self.repository = repository or VendorRepository()

    def create(self, db: Session, data: VendorCreate, actor_id: str) -> Vendor:
        if self.repository.by_code(db, data.vendor_code): raise HTTPException(409, "Vendor code already exists")
        vendor = self.repository.save(db, Vendor(**data.model_dump()))
        db.add(AuditLog(actor_id=actor_id, action="CREATE", resource_type="vendors", resource_id=vendor.id))
        try: db.commit()
        except IntegrityError: db.rollback(); raise HTTPException(409, "GSTIN is already registered")
        db.refresh(vendor); return vendor

    def update(self, db: Session, vendor_id: str, data: VendorUpdate, actor_id: str) -> Vendor:
        vendor = self.get(db, vendor_id); changes = data.model_dump(exclude_unset=True)
        for key, value in changes.items(): setattr(vendor, key, value)
        db.add(AuditLog(actor_id=actor_id, action="UPDATE", resource_type="vendors", resource_id=vendor.id, details={"fields": list(changes)}))
        try: db.commit()
        except IntegrityError: db.rollback(); raise HTTPException(409, "GSTIN is already registered")
        db.refresh(vendor); return vendor

    def get(self, db: Session, vendor_id: str) -> Vendor:
        vendor = self.repository.get(db, vendor_id)
        if not vendor: raise HTTPException(404, "Vendor not found")
        return vendor
