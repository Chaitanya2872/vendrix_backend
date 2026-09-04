from datetime import datetime, date, timezone
from uuid import uuid4
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
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


class VendorCategory(IdMixin, Base):
    __tablename__ = "vendor_categories"
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)


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
    vendor_id: Mapped[str | None] = mapped_column(ForeignKey("vendors.id"), nullable=True, index=True)
    invoice_number: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    invoice_date: Mapped[date] = mapped_column(Date)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    tax_amount: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(30), default="DRAFT")
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    vendor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    vendor_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    vendor_gstin: Mapped[str | None] = mapped_column(String(15), nullable=True, index=True)
    vendor_pan: Mapped[str | None] = mapped_column(String(10), nullable=True)
    vendor_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    vendor_phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    customer_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    customer_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_gstin: Mapped[str | None] = mapped_column(String(15), nullable=True, index=True)
    customer_pan: Mapped[str | None] = mapped_column(String(10), nullable=True)
    customer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    customer_phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    purchase_order_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    purchase_order_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True, default="INR")
    subtotal: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    discount_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    taxable_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    cgst_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    sgst_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    igst_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    round_off: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    total_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    amount_paid: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    amount_due: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    payment_terms: Mapped[str | None] = mapped_column(String(128), nullable=True)
    place_of_supply: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    used_ocr: Mapped[str | None] = mapped_column(String(8), nullable=True)
    parsing_confidence: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    parsing_warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    parsing_validation_errors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    line_items: Mapped[list["InvoiceLineItem"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )


class InvoiceLineItem(IdMixin, Base):
    __tablename__ = "invoice_line_items"

    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hsn_sac: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quantity: Mapped[float | None] = mapped_column(Numeric(14, 3), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    unit_price: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    discount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    gst_rate: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    taxable_value: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    total_amount: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)

    invoice: Mapped[Invoice] = relationship(back_populates="line_items")


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
    # Which vendor this document belongs to. Nullable because the upload form
    # has always allowed a document with no vendor chosen, and documents
    # predating this column have none — the Documents list surfaces those as
    # "Unassigned" rather than hiding them.
    vendor_id: Mapped[str | None] = mapped_column(ForeignKey("vendors.id"), nullable=True, index=True)
    # Compliance expiry the uploader entered. Kept on the document rather than
    # derived from extracted_fields: a certificate's expiry is a fact about
    # the file, and the expiry filters must work for types nothing parses.
    expires_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    extracted_fields: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    review_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Human-facing identifier (DOC-2026-000001). Nullable because documents
    # uploaded before the OCR pipeline existed have none, and backfilling
    # them would invent numbers nobody ever saw.
    document_number: Mapped[str | None] = mapped_column(String(24), unique=True, nullable=True, index=True)
    # Content hash, so re-uploading the same file is recognised instead of
    # paying for OCR twice. Indexed because it is looked up on every upload.
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Format as *detected from the bytes*, not as claimed by the extension.
    file_format: Mapped[str | None] = mapped_column(String(10), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AuditLog(IdMixin, Base):
    __tablename__ = "audit_logs"
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    resource_type: Mapped[str] = mapped_column(String(40))
    resource_id: Mapped[str] = mapped_column(String(36))
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class VehicleEntry(IdMixin, Base):
    """One vehicle visit across the gate — the inward/outward gate pass.

    `direction` describes what the visit is *for*, not which way the barrier
    moved: INWARD is a vehicle bringing material on to the site, OUTWARD is
    one taking material off it. Both kinds of visit arrive and both leave, so
    a single record with `entry_at` / `exit_at` covers each of them without
    the two-rows-per-visit bookkeeping that makes "which vehicles are inside
    right now" an expensive question. While `exit_at` is null the vehicle is
    on the premises; that is the whole of the state machine.

    A visit is deliberately not required to reference a registered Vehicle.
    Gate security cannot refuse a truck because nobody onboarded it, so
    `registration_number` is always captured as text and `vehicle_id` is
    filled in when the plate matches the fleet registry.
    """

    __tablename__ = "vehicle_entries"

    entry_number: Mapped[str] = mapped_column(String(24), unique=True, index=True)
    direction: Mapped[str] = mapped_column(String(10), index=True)
    status: Mapped[str] = mapped_column(String(20), default="IN_PREMISES", index=True)
    purpose: Mapped[str] = mapped_column(String(30), default="DELIVERY")

    # Normalised plate text, always present. Indexed because the open-visit
    # check runs on it for every single gate-in.
    registration_number: Mapped[str] = mapped_column(String(15), index=True)
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True, index=True)
    vendor_id: Mapped[str | None] = mapped_column(ForeignKey("vendors.id"), nullable=True, index=True)
    driver_id: Mapped[str | None] = mapped_column(ForeignKey("drivers.id"), nullable=True)
    # Kept alongside driver_id: a substitute driver with no record in the
    # system is the common case at a gate, and the name on the pass is what
    # an incident investigation needs months later even if the Driver row
    # has since been reassigned or deleted.
    driver_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    driver_phone: Mapped[str | None] = mapped_column(String(30), nullable=True)

    purchase_id: Mapped[str | None] = mapped_column(ForeignKey("purchases.id"), nullable=True, index=True)
    delivery_id: Mapped[str | None] = mapped_column(ForeignKey("deliveries.id"), nullable=True, index=True)

    gate: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    material_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Whatever paperwork the driver presented: invoice, delivery challan, or
    # gate pass number issued elsewhere.
    document_reference: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # Weighbridge readings. Stored as measured rather than as a signed delta
    # so a disputed net weight can be traced back to the two readings that
    # produced it. `net_weight` is derived on write (gross - tare).
    gross_weight: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    tare_weight: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)
    net_weight: Mapped[float | None] = mapped_column(Numeric(12, 3), nullable=True)

    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    exit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    # MANUAL or ANPR — which one wrote the plate. Worth keeping: it is the
    # only way to measure whether the camera is earning its place.
    capture_method: Mapped[str] = mapped_column(String(20), default="MANUAL")

    recorded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    exit_recorded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    remarks: Mapped[str | None] = mapped_column(Text, nullable=True)
