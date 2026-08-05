from fastapi import APIRouter
from app.models import Payment
from app.modules.crud import attach_crud
from app.modules.payments.schemas import PaymentCreate, PaymentUpdate
router = APIRouter(prefix="/payments", tags=["payments"])
attach_crud(router, Payment, PaymentCreate, PaymentUpdate, write_roles=("ADMIN", "FINANCE"))
