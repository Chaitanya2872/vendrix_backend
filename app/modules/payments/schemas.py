from datetime import date
from pydantic import BaseModel, Field
class PaymentCreate(BaseModel):
    invoice_id: str; amount: float = Field(gt=0); paid_on: date; reference: str; status: str = "PENDING"
class PaymentUpdate(BaseModel):
    amount: float | None = Field(default=None, gt=0); paid_on: date | None = None; reference: str | None = None; status: str | None = None
