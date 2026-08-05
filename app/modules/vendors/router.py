from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session
from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import AuditLog, User
from app.modules.vendors.repository import VendorRepository
from app.modules.vendors.schemas import VendorCreate, VendorRead, VendorUpdate
from app.modules.vendors.service import VendorService

router = APIRouter(prefix="/vendors", tags=["vendors"])
service = VendorService()

@router.get("", response_model=list[VendorRead])
def list_vendors(limit: int = Query(50, le=200), offset: int = 0, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return VendorRepository().list(db, limit=limit, offset=offset)

@router.post("", response_model=VendorRead, status_code=201)
def create_vendor(body: VendorCreate, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "OPERATOR"))):
    return service.create(db, body, user.id)

@router.get("/{vendor_id}", response_model=VendorRead)
def get_vendor(vendor_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return service.get(db, vendor_id)

@router.patch("/{vendor_id}", response_model=VendorRead)
def update_vendor(vendor_id: str, body: VendorUpdate, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN", "OPERATOR"))):
    return service.update(db, vendor_id, body, user.id)

@router.delete("/{vendor_id}", status_code=204)
def delete_vendor(vendor_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN"))):
    vendor = service.get(db, vendor_id); db.add(AuditLog(actor_id=user.id, action="DELETE", resource_type="vendors", resource_id=vendor.id)); db.delete(vendor); db.commit(); return Response(status_code=204)
