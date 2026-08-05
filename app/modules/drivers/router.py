from fastapi import APIRouter
from app.models import Driver
from app.modules.crud import attach_crud
from app.modules.drivers.schemas import DriverCreate, DriverUpdate
router = APIRouter(prefix="/drivers", tags=["drivers"])
attach_crud(router, Driver, DriverCreate, DriverUpdate, write_roles=("ADMIN", "OPERATOR"))
