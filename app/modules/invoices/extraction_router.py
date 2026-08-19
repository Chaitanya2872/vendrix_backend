"""Invoice extraction endpoints: upload a document, watch it process.

Mounted onto the existing invoices router, so these live at
`/api/v1/invoices/...` alongside the manual CRUD. The brief writes them as
`/api/invoices/...`; the `/v1` is this project's existing convention and
every other route already carries it, so consistency wins over matching the
brief's prose literally.

The upload endpoint does the minimum that must happen synchronously —
validate, store, register, queue — and nothing that takes measurable time.
OCR is minutes per document; doing it inline would hold the connection open
long past any sensible client timeout and lose the work when it fires.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.common.dependencies import current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import AuditLog, Document, Invoice, User
from app.modules.documents import repository as document_repository
from app.modules.documents.models import FieldCorrectionRecord
from app.modules.documents.service import status_payload
from app.modules.documents.stages import STATUS_PROCESSING, is_terminal
from app.modules.invoices.schemas_extraction import ProcessingStatus, UploadAccepted
from app.utils import file_utils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invoices", tags=["invoice-extraction"])

# The formats the brief names, plus WEBP which the existing upload path
# already accepted and which costs nothing to keep.
INVOICE_FORMATS = frozenset(
    {file_utils.PDF, file_utils.JPEG, file_utils.PNG, file_utils.TIFF, file_utils.WEBP}
)


@router.post("/upload", response_model=UploadAccepted, status_code=201)
async def upload_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    reprocess_duplicates: bool = Form(
        default=False,
        description="Process the file even if an identical one was uploaded before.",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> UploadAccepted:
    payload = await file.read()

    try:
        validated = file_utils.validate_upload(
            payload=payload,
            filename=file.filename or "",
            declared_media_type=file.content_type,
            max_bytes=settings.max_upload_size_mb * 1024 * 1024,
            allowed_formats=INVOICE_FORMATS,
        )
    except file_utils.FileValidationError as exc:
        # 415 for "we don't handle this kind of file", 413 for size, 422 for
        # a file that is the right kind but broken. A client retrying blindly
        # on 4xx should be able to tell which of those is worth retrying.
        status_code = {
            "FILE_TOO_LARGE": 413,
            "UNSUPPORTED_EXTENSION": 415,
            "UNSUPPORTED_FORMAT": 415,
        }.get(exc.code, 422)
        logger.info("invoice_upload.rejected code=%s filename=%s", exc.code, file.filename)
        raise HTTPException(status_code, detail={"code": exc.code, "message": exc.message})

    existing = document_repository.find_by_checksum(db, validated.sha256, owner_id=user.id)
    if existing is not None and not reprocess_duplicates:
        # Not an error: the user gets the earlier document rather than a
        # second identical one, and is spared paying for OCR twice.
        logger.info(
            "invoice_upload.duplicate document=%s sha256=%s",
            existing.document_number or existing.id, validated.sha256[:12],
        )
        run = document_repository.latest_run(db, existing.id)
        return UploadAccepted(
            document_id=existing.document_number or existing.id,
            filename=existing.filename,
            status=run.status if run else "UPLOADED",
            file_format=existing.file_format or validated.file_format,
            size_bytes=existing.size_bytes or validated.size_bytes,
            page_count=existing.page_count,
            duplicate_of=existing.document_number or existing.id,
        )

    document_number = document_repository.next_document_number(db)
    object_key = file_utils.safe_storage_key("invoices", document_number, file.filename or "invoice")

    stored_path = Path(settings.storage_path) / object_key
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(payload)

    document = Document(
        filename=file.filename or stored_path.name,
        object_key=object_key,
        content_type=validated.media_type,
        document_type="INVOICE",
        status="UPLOADED",
        owner_id=user.id,
        document_number=document_number,
        sha256=validated.sha256,
        size_bytes=validated.size_bytes,
        file_format=validated.file_format,
        page_count=validated.page_count,
    )
    db.add(document)
    try:
        db.flush()
    except IntegrityError:
        # Same file, same second, two requests: the loser cleans up its own
        # stored copy rather than leaving an orphan on disk.
        db.rollback()
        stored_path.unlink(missing_ok=True)
        raise HTTPException(409, detail={"code": "UPLOAD_CONFLICT", "message": "This upload conflicted with another; retry."})

    run = document_repository.create_run(db, document.id)
    db.add(AuditLog(actor_id=user.id, action="UPLOAD_INVOICE", resource_type="documents", resource_id=document.id))
    db.commit()

    _enqueue(background_tasks, document.id, run.id)

    logger.info(
        "invoice_upload.accepted document=%s format=%s size=%s pages=%s",
        document_number, validated.file_format, validated.size_bytes, validated.page_count,
    )
    return UploadAccepted(
        document_id=document_number,
        filename=document.filename,
        status=STATUS_PROCESSING,
        file_format=validated.file_format,
        size_bytes=validated.size_bytes,
        page_count=validated.page_count,
    )


@router.get("/{document_id}/status", response_model=ProcessingStatus)
def processing_status(
    document_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> ProcessingStatus:
    """Progress of the most recent processing run for a document.

    This is a polling endpoint — clients hit it every second or two for the
    minutes a document takes — so it does two indexed reads and nothing else.
    """
    document = document_repository.find_document(db, document_id)
    if document is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Document not found"})
    run = document_repository.latest_run(db, document.id)
    return ProcessingStatus(**status_payload(document, run))


@router.post("/{document_id}/reprocess", response_model=ProcessingStatus, status_code=202)
def reprocess(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> ProcessingStatus:
    """Run the pipeline again over an already-uploaded file.

    Needed after a model or lexicon upgrade, and after a transient failure
    that exhausted its retries. A new run row is created rather than the old
    one being overwritten, so the previous outcome stays auditable.
    """
    document = document_repository.find_document(db, document_id)
    if document is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Document not found"})

    current = document_repository.latest_run(db, document.id)
    if current is not None and not is_terminal(current.status):
        raise HTTPException(
            409,
            detail={"code": "ALREADY_PROCESSING", "message": "This document is still being processed."},
        )

    stored_path = Path(settings.storage_path) / document.object_key
    if not stored_path.exists():
        raise HTTPException(
            410,
            detail={"code": "FILE_MISSING", "message": "The stored file is no longer available."},
        )

    run = document_repository.create_run(db, document.id)
    db.add(AuditLog(actor_id=user.id, action="REPROCESS_INVOICE", resource_type="documents", resource_id=document.id))
    db.commit()

    _enqueue(background_tasks, document.id, run.id)
    return ProcessingStatus(**status_payload(document, run))


@router.get("/{document_id}/result")
def extraction_result(
    document_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> dict:
    """Everything the review screen needs, in one request.

    Deliberately one call rather than three: the review screen is useless
    until it has all of the fields, the evidence and the confidence together,
    so splitting them would only add two round trips to the same wait.
    """
    document = document_repository.find_document(db, document_id)
    if document is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Document not found"})

    run = document_repository.latest_run(db, document.id)
    fields = dict(document.extracted_fields or {})

    # Evidence and confidence live alongside the values in the same JSON
    # column; lift them out so the client is not parsing a field bag.
    evidence = fields.pop("evidence", None)
    confidence = fields.pop("confidence", None)

    invoice = db.scalars(
        select(Invoice).where(Invoice.document_id == document.id)
    ).first()

    return {
        "document_id": document.document_number or document.id,
        "filename": document.filename,
        "status": run.status if run else document.status,
        "page_count": document.page_count or (run.page_count if run else None),
        "used_ocr": run.used_ocr if run else None,
        "fields": fields,
        "evidence": evidence,
        "confidence": confidence,
        "invoice_id": invoice.id if invoice else None,
        "reviewed_at": document.review_confirmed_at.isoformat() if document.review_confirmed_at else None,
    }


@router.get("/{document_id}/pages/{page_number}")
def page_image(
    document_id: str,
    page_number: int,
    dpi: int = 150,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    """One page rendered as a PNG, for the review screen to draw boxes over.

    Rendered on demand rather than stored: a cached render per page per
    document would multiply storage for something regenerable in
    milliseconds, and the review screen is the only consumer.

    `dpi` defaults well below the extraction DPI — the browser is displaying
    this at a few hundred pixels wide, and rendering at 300 would send
    several megabytes to draw a thumbnail. Evidence boxes are in extraction-
    DPI pixel space, so the response reports the scale to apply.
    """
    from fastapi.responses import Response

    document = document_repository.find_document(db, document_id)
    if document is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Document not found"})

    stored_file = Path(settings.storage_path) / document.object_key
    if not stored_file.exists():
        raise HTTPException(410, detail={"code": "FILE_MISSING", "message": "The stored file is gone."})

    dpi = max(72, min(dpi, 300))

    try:
        png = _render_page_png(stored_file, page_number, dpi)
    except IndexError:
        raise HTTPException(404, detail={"code": "NO_SUCH_PAGE", "message": f"Page {page_number} does not exist."})
    except Exception as exc:
        logger.warning("page_image.render_failed document=%s page=%s error=%s", document_id, page_number, exc)
        raise HTTPException(422, detail={"code": "RENDER_FAILED", "message": str(exc)})

    return Response(
        content=png,
        media_type="image/png",
        headers={
            # The rendered page is a pure function of the file and the DPI,
            # and the file never changes once uploaded.
            "Cache-Control": "private, max-age=86400",
            "X-Render-Dpi": str(dpi),
            "X-Extraction-Dpi": str(settings.ocr_render_dpi),
        },
    )


def _render_page_png(path: Path, page_number: int, dpi: int) -> bytes:
    """Render page `page_number` (1-based) of a PDF or raster file to PNG."""
    import cv2
    import numpy as np

    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(str(path)) as document:
            if not 1 <= page_number <= document.page_count:
                raise IndexError(page_number)
            return document.load_page(page_number - 1).get_pixmap(dpi=dpi, alpha=False).tobytes("png")

    # Raster: multi-frame TIFF has real pages; everything else has one.
    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        frames = list(ImageSequence.Iterator(image))
        if not 1 <= page_number <= len(frames):
            raise IndexError(page_number)
        frame = frames[page_number - 1].convert("RGB")
        array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    encoded, buffer = cv2.imencode(".png", array)
    if not encoded:
        raise ValueError("Unable to encode page image")
    return buffer.tobytes()


class FieldCorrections(BaseModel):
    fields: dict[str, Any]
    confirm: bool = True


@router.post("/{document_id}/corrections")
def submit_corrections(
    document_id: str,
    body: FieldCorrections,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> dict:
    """Record a reviewer's corrections and update the invoice.

    Corrections are kept as their own record, not merged silently into the
    extraction: the difference between what was read and what was right is
    the only real training signal this system will ever get, and it is worth
    more than any amount of synthetic data.
    """
    document = document_repository.find_document(db, document_id)
    if document is None:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Document not found"})

    original = dict(document.extracted_fields or {})
    corrections = {
        name: value for name, value in body.fields.items()
        if original.get(name) != value
    }

    updated = dict(original)
    updated.update(body.fields)
    document.extracted_fields = updated
    if body.confirm:
        document.status = "CONFIRMED"
        document.review_confirmed_at = datetime.now(timezone.utc)

    invoice = db.scalars(select(Invoice).where(Invoice.document_id == document.id)).first()
    if invoice is not None:
        _apply_to_invoice(invoice, body.fields)
        if body.confirm:
            invoice.status = "PARSED"

    if corrections:
        db.add(FieldCorrectionRecord(
            document_id=document.id,
            invoice_id=invoice.id if invoice else None,
            corrected_by=user.id,
            corrections={
                name: {"was": original.get(name), "now": value}
                for name, value in corrections.items()
            },
        ))

    db.add(AuditLog(actor_id=user.id, action="REVIEW_INVOICE",
                    resource_type="documents", resource_id=document.id))
    db.commit()

    logger.info(
        "invoice_review.saved document=%s corrected=%s confirmed=%s",
        document_id, sorted(corrections), body.confirm,
    )
    return {
        "document_id": document.document_number or document.id,
        "status": document.status,
        "corrected_fields": sorted(corrections),
    }


def _apply_to_invoice(invoice: Invoice, fields: dict) -> None:
    """Push corrected values onto the Invoice row.

    Only the columns the review form actually edits, and only where a value
    was supplied — a field the reviewer left alone must not be blanked by an
    absent key.
    """
    from datetime import date as date_type
    from decimal import Decimal, InvalidOperation

    def as_decimal(value):
        try:
            return Decimal(str(value)) if value is not None else None
        except (InvalidOperation, TypeError):
            return None

    def as_date(value):
        if not value:
            return None
        if isinstance(value, date_type):
            return value
        try:
            return date_type.fromisoformat(str(value))
        except ValueError:
            return None

    simple = ("invoice_number", "vendor_name", "vendor_gstin",
              "customer_name", "customer_gstin")
    for name in simple:
        if name in fields:
            setattr(invoice, name, fields[name])

    for name in ("invoice_date", "due_date"):
        if name in fields:
            parsed = as_date(fields[name])
            if parsed is not None:
                setattr(invoice, name, parsed)

    for name in ("subtotal", "taxable_amount", "cgst_amount", "sgst_amount",
                 "igst_amount", "tax_amount", "round_off", "total_amount"):
        if name in fields:
            setattr(invoice, name, as_decimal(fields[name]))

    if "total_amount" in fields:
        amount = as_decimal(fields["total_amount"])
        if amount is not None:
            invoice.amount = float(amount)


def _enqueue(background_tasks: BackgroundTasks, document_id: str, run_id: str) -> None:
    """Hand the work to Celery, or to FastAPI's background tasks when no
    broker is configured.

    The in-process fallback is what makes a single-container on-premise
    install work without Redis. It is not a queue — it dies with the process
    — which is why the Celery path is preferred whenever a broker is
    available, and why a failed enqueue falls back rather than 500ing.
    """
    from app.workers.document_tasks import run_invoice_pipeline, run_invoice_pipeline_now

    if settings.celery_enabled:
        try:
            run_invoice_pipeline.delay(document_id, run_id)
            return
        except Exception:
            logger.warning("invoice_upload.queue_unavailable falling_back_to_inline document_id=%s", document_id)
    background_tasks.add_task(run_invoice_pipeline_now, document_id, run_id)
