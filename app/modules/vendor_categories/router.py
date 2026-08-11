from fastapi import APIRouter
from app.models import VendorCategory
from app.modules.crud import attach_crud
from app.modules.vendor_categories.schemas import VendorCategoryCreate, VendorCategoryUpdate
router = APIRouter(prefix="/vendor-categories", tags=["vendor-categories"])
attach_crud(router, VendorCategory, VendorCategoryCreate, VendorCategoryUpdate, write_roles=("ADMIN",))
