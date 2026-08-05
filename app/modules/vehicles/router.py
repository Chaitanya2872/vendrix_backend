from fastapi import APIRouter
from app.models import Vehicle
from app.modules.crud import attach_crud
from app.modules.vehicles.schemas import VehicleCreate, VehicleUpdate
from app.utils.validators import normalize_registration_number
router = APIRouter(prefix="/vehicles", tags=["vehicles"])
attach_crud(router, Vehicle, VehicleCreate, VehicleUpdate, write_roles=("ADMIN", "OPERATOR"), transform=lambda v: {**v, **({"registration_number": normalize_registration_number(v["registration_number"])} if v.get("registration_number") else {})})
