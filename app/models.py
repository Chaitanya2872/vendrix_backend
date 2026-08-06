from datetime import datetime, date, timezone
from uuid import uuid4
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


class IdMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class User(IdMixin, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(150))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30), default="OPERATOR")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Vendor(IdMixin, Base):
    __tablename__ = "vendors"
    vendor_code: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    legal_name: Mapped[str] = mapped_column(String(200), index=True)
    gstin: Mapped[str | None] = mapped_column(String(15), unique=True, nullable=True)
    category: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="DRAFT")
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    email: Mapped[str | None] = mapped_column(String(254), nullable=True)
    address: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bank_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Vehicle(IdMixin, Base):
    __tablename__ = "vehicles"
    vendor_id: Mapped[str] = mapped_column(ForeignKey("vendors.id"), index=True)
    registration_number: Mapped[str] = mapped_column(String(15), unique=True, index=True)
    vehicle_type: Mapped[str] = mapped_column(String(80))
    make: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    rc_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    insurance_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ACTIVE")


class Driver(IdMixin, Base):
    __tablename__ = "drivers"
    vendor_id: Mapped[str] = mapped_column(ForeignKey("vendors.id"), index=True)
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True)
    full_name: Mapped[str] = mapped_column(String(150))
    phone: Mapped[str] = mapped_column(String(30))
    license_number: Mapped[str] = mapped_column(String(40), unique=True)
    license_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ACTIVE")


class Invoice(IdMixin, Base):
    __tablename__ = "invoices"
    vendor_id: Mapped[str] = mapped_column(ForeignKey("vendors.id"), index=True)
    invoice_number: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    invoice_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    tax_amount: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(30), default="DRAFT")
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)


class Purchase(IdMixin, Base):
    __tablename__ = "purchases"
    purchase_number: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    vendor_id: Mapped[str] = mapped_column(ForeignKey("vendors.id"), index=True)
    reference: Mapped[str | None] = mapped_column(String(120), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, default=0)
    total_amount: Mapped[float] = mapped_column(Float, default=0)
    expected_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="DRAFT")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Delivery(IdMixin, Base):
    __tablename__ = "deliveries"
    delivery_number: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    vendor_id: Mapped[str] = mapped_column(ForeignKey("vendors.id"), index=True)
    purchase_id: Mapped[str | None] = mapped_column(ForeignKey("purchases.id"), nullable=True, index=True)
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True)
    driver_id: Mapped[str | None] = mapped_column(ForeignKey("drivers.id"), nullable=True)
    destination: Mapped[str] = mapped_column(String(250))
    recipient: Mapped[str | None] = mapped_column(String(150), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(30), default="SCHEDULED")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Payment(IdMixin, Base):
    __tablename__ = "payments"
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.id"), index=True)
    amount: Mapped[float] = mapped_column(Float)
    paid_on: Mapped[date] = mapped_column(Date)
    reference: Mapped[str] = mapped_column(String(100), unique=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING")


class Approval(IdMixin, Base):
    __tablename__ = "approvals"
    resource_type: Mapped[str] = mapped_column(String(40), index=True)
    resource_id: Mapped[str] = mapped_column(String(36), index=True)
    requested_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    approver_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING")
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Document(IdMixin, Base):
    __tablename__ = "documents"
    filename: Mapped[str] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(String(400), unique=True)
    content_type: Mapped[str] = mapped_column(String(100))
    document_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="UPLOADED")
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    extracted_fields: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    review_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(IdMixin, Base):
    __tablename__ = "audit_logs"
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    resource_type: Mapped[str] = mapped_column(String(40))
    resource_id: Mapped[str] = mapped_column(String(36))
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
