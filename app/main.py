"""FastAPI application composition; domain behavior lives in app.modules."""
import logging
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
from app.models import User, VendorCategory
from app.modules.anpr.router import router as anpr_router
from app.modules.approvals.router import router as approvals_router
from app.modules.audit.router import router as audit_router
from app.modules.auth.router import router as auth_router
from app.modules.documents.router import router as documents_router
from app.modules.deliveries.router import router as deliveries_router
from app.modules.drivers.router import router as drivers_router
from app.modules.invoices.extraction_router import router as invoice_extraction_router
from app.modules.invoices.router import router as invoices_router
from app.modules.mobile.router import router as mobile_router
from app.modules.purchases.router import router as purchases_router
from app.modules.payments.router import router as payments_router
from app.modules.reports.router import router as reports_router
from app.modules.users.router import router as users_router
from app.modules.vehicles.router import router as vehicles_router
from app.modules.vendors.router import router as vendors_router
from app.modules.vendor_categories.router import router as vendor_categories_router

app = FastAPI(title=settings.app_name, version="0.1.0", openapi_url="/api/v1/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3})(:\d+)?", allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for router in (auth_router, users_router, vendors_router, vendor_categories_router, vehicles_router, drivers_router, invoices_router, invoice_extraction_router, payments_router, approvals_router, documents_router, purchases_router, deliveries_router, anpr_router, reports_router, mobile_router, audit_router):
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
    # Keep existing lightweight databases compatible with invoice parsing.
    if "invoices" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("invoices")}
        invoice_columns = {
            "vendor_name": "VARCHAR(255)", "vendor_address": "TEXT",
            "vendor_gstin": "VARCHAR(15)", "vendor_pan": "VARCHAR(10)",
            "vendor_email": "VARCHAR(255)", "vendor_phone": "VARCHAR(32)",
            "customer_name": "VARCHAR(255)", "customer_address": "TEXT",
            "customer_gstin": "VARCHAR(15)", "customer_pan": "VARCHAR(10)",
            "customer_email": "VARCHAR(255)", "customer_phone": "VARCHAR(32)",
            "purchase_order_number": "VARCHAR(64)", "purchase_order_date": "DATE",
            "currency": "VARCHAR(8)", "subtotal": "NUMERIC(14, 2)",
            "discount_amount": "NUMERIC(14, 2)", "taxable_amount": "NUMERIC(14, 2)",
            "cgst_amount": "NUMERIC(14, 2)", "sgst_amount": "NUMERIC(14, 2)",
            "igst_amount": "NUMERIC(14, 2)", "round_off": "NUMERIC(14, 2)",
            "total_amount": "NUMERIC(14, 2)", "amount_paid": "NUMERIC(14, 2)",
            "amount_due": "NUMERIC(14, 2)", "payment_terms": "VARCHAR(128)",
            "place_of_supply": "VARCHAR(128)", "parser_name": "VARCHAR(64)",
            "parser_version": "VARCHAR(16)", "used_ocr": "VARCHAR(8)",
            "parsing_confidence": "NUMERIC(4, 2)", "parsing_warnings": "JSON",
            "parsing_validation_errors": "JSON",
        }
        with engine.begin() as connection:
            for name, sql_type in invoice_columns.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE invoices ADD COLUMN {name} {sql_type}"))
    # Keep existing lightweight databases compatible with the OCR pipeline.
    # The unique index on document_number is created separately because
    # SQLite cannot add a UNIQUE column to a populated table.
    if "documents" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("documents")}
        document_columns = {
            "document_number": "VARCHAR(24)", "sha256": "VARCHAR(64)",
            "size_bytes": "INTEGER", "file_format": "VARCHAR(10)",
            "page_count": "INTEGER",
            # Vendor ownership and compliance expiry, added when the Documents
            # page gained vendor and expiry filtering. Both nullable: existing
            # rows have neither and inventing values would be a lie.
            "vendor_id": "VARCHAR(36)", "expires_on": "DATE",
        }
        indexes = {index["name"] for index in inspector.get_indexes("documents")}
        with engine.begin() as connection:
            for name, sql_type in document_columns.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}"))
            if "ix_documents_document_number" not in indexes:
                connection.execute(text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_document_number "
                    "ON documents (document_number)"
                ))
            if "ix_documents_sha256" not in indexes:
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_sha256 ON documents (sha256)"
                ))
            if "ix_documents_vendor_id" not in indexes:
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_vendor_id ON documents (vendor_id)"
                ))

    # Load the extraction models now, off the request path. Both are a fixed
    # per-process cost — the OCR engine's runtime session, and the field
    # model's joblib artifact — and paying either inside the first document
    # upload is what made extraction look slow even for documents that parse
    # in under a second. A daemon thread, so a slow or missing model never
    # delays startup or holds the process open at shutdown.
    if settings.ocr_warm_up_on_startup:
        import threading

        threading.Thread(target=_warm_up_extraction, name="extraction-warm-up", daemon=True).start()

    with Session(engine) as db:
        if not db.scalar(select(User.id).limit(1)):
            db.add(User(email="admin@iotiq.example.com", full_name="System Administrator", password_hash=hash_password("Admin@123"), role="ADMIN"))
            db.commit()
        if not db.scalar(select(VendorCategory.id).limit(1)):
            default_categories = ["Transport & Logistics", "Construction", "Equipment Rental", "Materials Supplier", "Professional Services", "Maintenance", "Office Supplies", "Other"]
            db.add_all(VendorCategory(name=name) for name in default_categories)
            db.commit()


def _warm_up_extraction() -> None:
    """Preload everything the extraction pipeline builds lazily.

    Both loaders are idempotent and internally cached, so this is safe to run
    concurrently with a request that got there first. Neither is allowed to
    raise: a deployment with no OCR models or no trained field model still
    serves every other route, and both paths already degrade gracefully.
    """
    from app.modules.ocr import engine as ocr_engine

    ocr_engine.warm_up()
    try:
        from app.ml.ocr import predict

        predict.load_model()
    except Exception:
        logging.getLogger(__name__).warning("ocr_model.warm_up_failed", exc_info=True)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vendor-management-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
