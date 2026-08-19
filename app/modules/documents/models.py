"""Tables owned by the document-processing pipeline.

`Document` itself lives in app/models.py with the rest of the core schema and
is not duplicated here. These two tables are additions the pipeline needs:
one to track a processing run, one to hand out human-facing document numbers.

Run state lives in its own table rather than as columns on `Document`
because a document can be processed more than once — a reprocess after a
model upgrade, a retry after a transient failure — and squashing that into
the document row destroys the history exactly when someone is asking "why did
this one come out wrong last week".
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models import IdMixin, now
from app.modules.documents.stages import STAGE_UPLOADED, STATUS_UPLOADED


class DocumentProcessingRun(IdMixin, Base):
    """One attempt at processing one document through the pipeline."""

    __tablename__ = "document_processing_runs"

    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    status: Mapped[str] = mapped_column(String(30), default=STATUS_UPLOADED, index=True)
    current_stage: Mapped[str] = mapped_column(String(40), default=STAGE_UPLOADED)
    progress: Mapped[int] = mapped_column(Integer, default=0)

    attempt: Mapped[int] = mapped_column(Integer, default=1)

    # Per-stage wall-clock in seconds, {stage_name: seconds}. The only honest
    # basis for tuning the progress weights above — and the first thing worth
    # looking at when a deployment is slower than expected.
    stage_timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Machine-readable failure discriminator (see file_utils.FileValidationError
    # and the OCR exceptions) plus the human message. Split for the same reason
    # they are split there: a UI should branch on the code, show the message.
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Summary of what came out, so a list view can show confidence and page
    # count without loading the full extraction result.
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    used_ocr: Mapped[bool | None] = mapped_column(nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)

    document = relationship("Document", backref="processing_runs")

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class DocumentNumberSequence(Base):
    """Per-year counter behind the DOC-2026-000001 document numbers.

    A real database sequence would be simpler, but the project supports both
    SQLite (single-file deployments) and PostgreSQL, and sequences are neither
    portable nor resettable per year. A counter row taken with `SELECT … FOR
    UPDATE` is portable, gives gap-free numbers, and serialises correctly
    under concurrent uploads — which `SELECT max(...)+1` does not.
    """

    __tablename__ = "document_number_sequences"
    __table_args__ = (UniqueConstraint("prefix", "year", name="uq_document_number_prefix_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prefix: Mapped[str] = mapped_column(String(10), default="DOC")
    year: Mapped[int] = mapped_column(Integer)
    last_value: Mapped[int] = mapped_column(Integer, default=0)


class FieldCorrectionRecord(IdMixin, Base):
    """What a reviewer changed, and what it was before.

    Kept as its own record rather than merged silently into the extraction:
    the difference between what was read and what was right is the only real
    training signal this system will ever get. The ML package's README says
    the same thing from the other side — "the first real correction data from
    the review screen is worth more than another thousand synthetic
    documents" — and this table is where that data accumulates.
    """

    __tablename__ = "invoice_field_corrections"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    invoice_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    corrected_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # {field: {"was": <extracted>, "now": <corrected>}}
    corrections: Mapped[dict | None] = mapped_column(JSON, nullable=True)
