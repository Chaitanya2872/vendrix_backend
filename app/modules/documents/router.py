"""Documents router — EXTENDED to trigger invoice parsing on upload.

Only the `upload` endpoint changes. Everything else (list/get/download/
preview/delete/review) is unchanged from the original file. The diff is
scoped to:

  1. After an INVOICE-typed document is persisted, enqueue invoice parsing
     using the *same* dispatch mechanism already used for
     `process_vendor_document` (Celery if enabled, background task
     otherwise) — no new queue/worker is introduced.
  2. The synchronous response still returns the Document immediately
     (matching current behavior); parsed invoice data becomes available via
     GET /documents/{id} or GET /invoices/{id} once processing completes,
     same as the existing `process_vendor_document` flow already implies
     for other document-derived data.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import shutil
from io import BytesIO
from uuid import uuid4
from typing import Any
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.common.dependencies import current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import AuditLog, Document, User, Vendor
from app.modules.documents.schemas import DocumentListItem
from app.modules.invoices.services.invoice_parser_service import should_parse_as_invoice

router = APIRouter(prefix="/documents", tags=["documents"])
class DocumentReview(BaseModel): fields: dict[str, Any]

@router.get("", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()

@router.post("", status_code=201)
def upload(
    document_type: str,
    background_tasks: BackgroundTasks,
    vendor_id: str | None = None,
    expires_on: date | None = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    extension = Path(file.filename or "").suffix.lower()
    supported_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".docx", ".xlsx"}
    if extension not in supported_extensions:
        raise HTTPException(415, "Supported files: PDF, JPG, PNG, WEBP, DOCX, XLSX")
    # An unknown vendor is rejected rather than silently stored: a document
    # filed against a vendor that does not exist is invisible to every vendor
    # filter, which reads to the user as a lost upload.
    if vendor_id and not db.get(Vendor, vendor_id):
        raise HTTPException(404, "Vendor not found")
    key = f"{user.id}/{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}_{file.filename}"; path = Path(settings.storage_path) / key; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output: shutil.copyfileobj(file.file, output)
    document = Document(filename=file.filename, object_key=key, content_type=file.content_type, document_type=document_type.upper(), owner_id=user.id, vendor_id=vendor_id or None, expires_on=expires_on, size_bytes=path.stat().st_size)
    db.add(document); db.flush(); db.add(AuditLog(actor_id=user.id, action="UPLOAD", resource_type="documents", resource_id=document.id)); db.commit(); db.refresh(document)
    # Exactly one extraction task runs per upload. Both write to
    # Document.extracted_fields, so running the generic vendor extractor
    # alongside the invoice parser would have them overwrite each other —
    # the invoice fields the review UI needs would be replaced by a raw
    # text dump, depending on which finished last.
    from app.workers.document_tasks import (
        process_invoice_document, process_invoice_document_now,
        process_vendor_document, process_vendor_document_now,
    )
    if should_parse_as_invoice(document):
        queued_task, inline_task = process_invoice_document, process_invoice_document_now
    else:
        queued_task, inline_task = process_vendor_document, process_vendor_document_now

    if settings.celery_enabled:
        try:
            queued_task.delay(document.id)
        except Exception:
            # Keep the request fast if the queue is temporarily unavailable.
            background_tasks.add_task(inline_task, document.id)
    else:
        # Local development: send the response first, then process in-process.
        background_tasks.add_task(inline_task, document.id)

    return document

@router.get("/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    document = db.get(Document, document_id)
    if not document: raise HTTPException(404, "Document not found")
    return document

@router.get("/{document_id}/download")
def download_document(document_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    document = db.get(Document, document_id)
    if not document: raise HTTPException(404, "Document not found")
    stored_file = Path(settings.storage_path) / document.object_key
    if not stored_file.exists(): raise HTTPException(404, "Stored file not found")
    return FileResponse(stored_file, media_type=document.content_type or "application/octet-stream", filename=document.filename, content_disposition_type="inline")

@router.get("/{document_id}/preview")
def document_preview(document_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    document = db.get(Document, document_id)
    if not document: raise HTTPException(404, "Document not found")
    stored_file = Path(settings.storage_path) / document.object_key
    if not stored_file.exists(): raise HTTPException(404, "Stored file not found")
    extension = stored_file.suffix.lower()
    if extension == ".docx":
        from docx import Document as WordDocument
        lines = [paragraph.text for paragraph in WordDocument(stored_file).paragraphs if paragraph.text.strip()]
        return {"type": "document", "lines": lines}
    if extension == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(BytesIO(stored_file.read_bytes()), read_only=True, data_only=True)
        sheet = workbook.active
        rows = [["" if value is None else str(value) for value in row] for row in sheet.iter_rows(values_only=True, max_row=100, max_col=20)]
        return {"type": "spreadsheet", "sheet": sheet.title, "rows": rows}
    return {"type": "binary"}

@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    document = db.get(Document, document_id)
    if not document: raise HTTPException(404, "Document not found")
    stored_file = Path(settings.storage_path) / document.object_key
    if stored_file.exists(): stored_file.unlink()
    db.add(AuditLog(actor_id=user.id, action="DELETE", resource_type="documents", resource_id=document.id)); db.delete(document); db.commit()

@router.post("/{document_id}/review")
def review(document_id: str, body: DocumentReview, db: Session = Depends(get_db), user: User = Depends(current_user)):
    document = db.get(Document, document_id)
    if not document: raise HTTPException(404, "Document not found")
    document.extracted_fields, document.status, document.review_confirmed_at = body.fields, "CONFIRMED", datetime.now(timezone.utc)
    db.add(AuditLog(actor_id=user.id, action="CONFIRM_EXTRACTION", resource_type="documents", resource_id=document.id)); db.commit(); db.refresh(document); return document
