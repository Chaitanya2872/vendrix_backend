"""FastAPI application composition; domain behavior lives in app.modules."""
import sys
from pathlib import Path

# Supports `python main.py` when launched from the app directory. Production
# servers should use `uvicorn app.main:app` from the backend project root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import hash_password
from app.db.base import Base
from app.db.session import engine
from app.models import User
from app.modules.anpr.router import router as anpr_router
from app.modules.approvals.router import router as approvals_router
from app.modules.audit.router import router as audit_router
from app.modules.auth.router import router as auth_router
from app.modules.documents.router import router as documents_router
from app.modules.deliveries.router import router as deliveries_router
from app.modules.drivers.router import router as drivers_router
from app.modules.invoices.router import router as invoices_router
from app.modules.mobile.router import router as mobile_router
from app.modules.purchases.router import router as purchases_router
from app.modules.payments.router import router as payments_router
from app.modules.reports.router import router as reports_router
from app.modules.users.router import router as users_router
from app.modules.vehicles.router import router as vehicles_router
from app.modules.vendors.router import router as vendors_router

app = FastAPI(title=settings.app_name, version="0.1.0", openapi_url="/api/v1/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3})(:\d+)?", allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for router in (auth_router, users_router, vendors_router, vehicles_router, drivers_router, invoices_router, payments_router, approvals_router, documents_router, purchases_router, deliveries_router, anpr_router, reports_router, mobile_router, audit_router):
    app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def initialize() -> None:
    Base.metadata.create_all(engine)
    # The app intentionally supports existing lightweight SQLite deployments.
    # Add the optional purchase link when upgrading a database created before
    # purchase orders were introduced (new databases receive it via metadata).
    inspector = inspect(engine)
    if "deliveries" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("deliveries")}
        with engine.begin() as connection:
            if "purchase_id" not in columns:
                connection.execute(text("ALTER TABLE deliveries ADD COLUMN purchase_id VARCHAR(36)"))
            if "recipient" not in columns:
                connection.execute(text("ALTER TABLE deliveries ADD COLUMN recipient VARCHAR(150)"))
    with Session(engine) as db:
        if not db.scalar(select(User.id).limit(1)):
            db.add(User(email="admin@iotiq.example.com", full_name="System Administrator", password_hash=hash_password("Admin@123"), role="ADMIN"))
            db.commit()


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vendor-management-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
