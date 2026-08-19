"""Persistence for document processing runs and document numbers.

Kept separate from the service layer because both of these have concurrency
requirements that are easy to lose track of when mixed with pipeline logic:
numbers must not be handed out twice, and progress updates must land in their
own transaction so a poller can actually see them.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Document
from app.modules.documents.models import DocumentNumberSequence, DocumentProcessingRun
from app.modules.documents.stages import (
    STAGE_UPLOADED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
)

logger = logging.getLogger(__name__)


def next_document_number(db: Session, prefix: str = "DOC", year: int | None = None) -> str:
    """Allocate the next `DOC-2026-000001`.

    The counter row is locked for the duration of the surrounding
    transaction, so two simultaneous uploads queue rather than both reading
    the same value. `SELECT max(number) + 1` is the obvious alternative and is
    wrong under exactly the concurrency this endpoint will see.

    Note this leaves a gap if the caller's transaction later rolls back. That
    is the correct trade: gap-free numbering would require holding the lock
    across the whole upload including the disk write, serialising every
    upload in the system behind one another.
    """
    year = year or datetime.now(timezone.utc).year

    statement = select(DocumentNumberSequence).where(
        DocumentNumberSequence.prefix == prefix,
        DocumentNumberSequence.year == year,
    )
    # SQLite has no row locks — its whole-database write lock gives the same
    # guarantee — so the FOR UPDATE is requested only where it is meaningful.
    if db.bind is not None and db.bind.dialect.name != "sqlite":
        statement = statement.with_for_update()

    counter = db.scalars(statement).first()
    if counter is None:
        counter = DocumentNumberSequence(prefix=prefix, year=year, last_value=0)
        db.add(counter)
        try:
            db.flush()
        except IntegrityError:
            # Another uploader created the row for this year between our
            # SELECT and INSERT. Theirs is as good as ours; take it.
            db.rollback()
            counter = db.scalars(statement).one()

    counter.last_value += 1
    db.flush()
    return f"{prefix}-{year}-{counter.last_value:06d}"


def find_document(db: Session, identifier: str) -> Document | None:
    """Look a document up by either its human number or its internal id.

    The API hands out `DOC-2026-000001`, so that is what clients send back.
    Accepting the UUID too costs one branch and saves every internal caller
    from having to convert.
    """
    if identifier.upper().startswith("DOC-"):
        return db.scalars(
            select(Document).where(func.upper(Document.document_number) == identifier.upper())
        ).first()
    return db.get(Document, identifier)


def find_by_checksum(db: Session, sha256: str, owner_id: str | None = None) -> Document | None:
    """Find a previously uploaded identical file.

    Re-uploading the same invoice is common — a user refreshes, or a mail
    ingest replays — and OCR is expensive enough that recognising it is worth
    a query. Scoped to the owner when given, so one tenant's upload never
    reveals the existence of another's.
    """
    statement = select(Document).where(Document.sha256 == sha256)
    if owner_id is not None:
        statement = statement.where(Document.owner_id == owner_id)
    return db.scalars(statement.order_by(Document.created_at.desc())).first()


def create_run(db: Session, document_id: str) -> DocumentProcessingRun:
    """Open a new processing run, numbering it after any earlier attempts."""
    previous = db.scalar(
        select(func.count(DocumentProcessingRun.id)).where(
            DocumentProcessingRun.document_id == document_id
        )
    ) or 0
    run = DocumentProcessingRun(
        document_id=document_id,
        status=STATUS_UPLOADED,
        current_stage=STAGE_UPLOADED,
        progress=0,
        attempt=previous + 1,
        stage_timings={},
    )
    db.add(run)
    db.flush()
    return run


def latest_run(db: Session, document_id: str) -> DocumentProcessingRun | None:
    return db.scalars(
        select(DocumentProcessingRun)
        .where(DocumentProcessingRun.document_id == document_id)
        .order_by(DocumentProcessingRun.attempt.desc())
        .limit(1)
    ).first()


def get_run(db: Session, run_id: str) -> DocumentProcessingRun | None:
    return db.get(DocumentProcessingRun, run_id)


def mark_started(db: Session, run: DocumentProcessingRun) -> None:
    run.started_at = datetime.now(timezone.utc)
    run.status = STATUS_PROCESSING
    db.commit()


def save_progress(
    db: Session,
    run: DocumentProcessingRun,
    stage: str,
    progress: int,
    status: str,
    stage_timings: dict | None = None,
) -> None:
    """Publish progress immediately.

    Committed on its own rather than batched with the pipeline's work: a
    progress value that only becomes visible when the job ends tells the user
    nothing during the one period they are actually watching. The cost is one
    small write per stage, against a job measured in minutes.
    """
    run.current_stage = stage
    run.progress = progress
    run.status = status
    if stage_timings is not None:
        # Reassigned rather than mutated: SQLAlchemy does not track in-place
        # changes to a plain JSON dict, so a mutation would never be written.
        run.stage_timings = dict(stage_timings)
    db.commit()


def mark_finished(
    db: Session,
    run: DocumentProcessingRun,
    status: str,
    stage: str,
    progress: int,
    stage_timings: dict | None = None,
    page_count: int | None = None,
    used_ocr: bool | None = None,
    ocr_confidence: float | None = None,
    extraction_confidence: float | None = None,
) -> None:
    run.status = status
    run.current_stage = stage
    run.progress = progress
    run.finished_at = datetime.now(timezone.utc)
    if stage_timings is not None:
        run.stage_timings = dict(stage_timings)
    if page_count is not None:
        run.page_count = page_count
    if used_ocr is not None:
        run.used_ocr = used_ocr
    if ocr_confidence is not None:
        run.ocr_confidence = ocr_confidence
    if extraction_confidence is not None:
        run.extraction_confidence = extraction_confidence
    db.commit()


def mark_failed(
    db: Session,
    run: DocumentProcessingRun,
    code: str,
    message: str,
    stage: str | None = None,
) -> None:
    """Record a failure against the run.

    Rolls back first: the failure may well have left the session in a broken
    state, and a commit attempted on top of that would lose the error record
    itself — leaving a run stuck in PROCESSING forever, which is the one
    outcome a polling client cannot recover from.
    """
    db.rollback()
    run.status = STATUS_FAILED
    if stage:
        run.current_stage = stage
    run.error_code = code
    run.error_message = message[:2000]
    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    logger.warning(
        "pipeline.run_failed run_id=%s document_id=%s stage=%s code=%s",
        run.id, run.document_id, run.current_stage, code,
    )
