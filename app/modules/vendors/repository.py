from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models import Vendor


class VendorRepository:
    def list(self, db: Session, *, limit: int, offset: int) -> list[Vendor]:
        return list(db.scalars(select(Vendor).order_by(Vendor.created_at.desc()).offset(offset).limit(limit)))

    def get(self, db: Session, vendor_id: str) -> Vendor | None:
        return db.get(Vendor, vendor_id)

    def by_code(self, db: Session, code: str) -> Vendor | None:
        return db.scalar(select(Vendor).where(Vendor.vendor_code == code))

    def save(self, db: Session, vendor: Vendor) -> Vendor:
        db.add(vendor); db.flush(); return vendor
