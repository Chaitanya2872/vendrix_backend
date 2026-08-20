import logging
from pathlib import Path
from io import BytesIO
from sqlalchemy import select
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Document
from app.modules.invoices.services.invoice_parser_service import process_document as parse_invoice_document
from app.workers.celery_app import celery_app
from app.modules.ocr import service as ocr_service

logger = logging.getLogger(__name__)


def read_page_text(raw: bytes) -> str:
    """OCR one page image. Routed through the OCR service rather than
    `workers.vision.read_text`, which wraps the same engine in a fixed
    denoise-and-CLAHE pass tuned for number-plate crops — several seconds on
    a full page, for no gain on a document scan."""
    return ocr_service.recognize_image_bytes(raw).text


def extract_document_fields(document_type: str, text: str) -> dict:
    """Deterministic first-pass extraction; replace/extend with domain ML rules."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"document_type": document_type, "text": text[:10000], "preview_lines": lines[:20]}


def read_document_text(path: Path) -> str:
    """Plain text out of any format the upload endpoint accepts.

    OCR is the last resort, not the first: a PDF with a text layer is read
    from that layer, and only a PDF with no usable text is rasterised. Pages
    are OCR'd up to `ocr_max_pages` — the same ceiling the invoice path uses,
    for the same reason. A scanned back-matter annexure costs seconds per
    page and adds nothing a reviewer of this document will read.
    """
    extension = path.suffix.lower()
    if extension == ".pdf":
        import fitz
        with fitz.open(path) as pdf:
            text = "\n".join(page.get_text() for page in pdf)
            if text.strip():
                return text
            limit = settings.ocr_max_pages if settings.ocr_max_pages > 0 else len(pdf)
            pages = []
            for index in range(min(len(pdf), limit)):
                pixmap = pdf[index].get_pixmap(dpi=settings.ocr_render_dpi, alpha=False)
                pages.append(read_page_text(pixmap.tobytes("png")))
            return "\n".join(pages)
    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        return read_page_text(path.read_bytes())
    if extension == ".docx":
        from docx import Document as WordDocument
        return "\n".join(paragraph.text for paragraph in WordDocument(path).paragraphs)
    if extension == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(BytesIO(path.read_bytes()), read_only=True, data_only=True)
        return "\n".join(
            " | ".join(str(value) for value in row if value is not None)
            for sheet in workbook.worksheets
            for row in sheet.iter_rows(values_only=True)
        )
    raise ValueError(f"Unsupported document extension: {extension}")


def _record_extraction_failure(db, document: Document, message: str) -> dict:
    """Mark a document as needing a human because extraction could not run.

    Leaving it at UPLOADED is what makes the Documents UI show an extraction
    that never ends: from the client's side "no fields yet" and "still
    working" are the same state unless the server distinguishes them.
    """
    document.extracted_fields = {"warnings": [message], "validation_errors": []}
    document.status = "REVIEW_REQUIRED"
    db.commit()
    logger.warning("document_extraction.failed document_id=%s reason=%s", document.id, message)
    return {"document_id": document.id, "status": "failed", "error": message}


def process_vendor_document_now(document_id: str) -> dict:
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if not document:
            return {"document_id": document_id, "status": "not_found"}
        try:
            path = Path(settings.storage_path) / document.object_key
            if not path.exists():
                raise FileNotFoundError(path)
            fields = extract_document_fields(document.document_type, read_document_text(path))
        except Exception as exc:
            # Never propagates. On the BackgroundTasks path this runs after the
            # response has already been sent, so raising reaches no caller and
            # throws away the one chance to record what went wrong where the
            # user will look for it.
            logger.exception("document_extraction.unhandled_error document_id=%s", document_id)
            db.rollback()
            return _record_extraction_failure(db, document, f"This document could not be read: {exc}")
        document.extracted_fields, document.status = fields, "REVIEW_REQUIRED"
        db.commit()
        return {"document_id": document_id, "status": "review_required", "fields": fields}


@celery_app.task(bind=True, max_retries=2)
def process_vendor_document(self, document_id: str) -> dict:
    """Queued entry point.

    No `autoretry_for=(Exception,)`: `process_vendor_document_now` records its
    own failures and returns, so a blanket retry would only re-run documents
    that already failed deterministically — three times the OCR cost for the
    same answer. Retries are for the case where the task dies before it can
    record anything.
    """
    return process_vendor_document_now(document_id)


def process_invoice_document_now(document_id: str) -> None:
    """Invoice parsing for BackgroundTasks / local dev. Never raises: a bad
    invoice document must not take down the upload flow it is queued from."""
    with SessionLocal() as db:
        try:
            document = db.get(Document, document_id)
            if not document:
                logger.warning("invoice_parsing.document_missing document_id=%s", document_id)
                return
            parse_invoice_document(db, document)
        except Exception:
            logger.exception("invoice_parsing.unhandled_error document_id=%s", document_id)


@celery_app.task
def process_invoice_document(document_id: str) -> None:
    return process_invoice_document_now(document_id)


def run_invoice_pipeline_now(document_id: str, run_id: str | None = None) -> dict:
    """Run the staged invoice pipeline for BackgroundTasks / single-container
    deployments. Never raises: a bad document must not take down the request
    that queued it, and the failure is already recorded on the run row."""
    from app.modules.invoices.services.invoice_pipeline import process

    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if not document:
            logger.warning("pipeline.document_missing document_id=%s", document_id)
            return {"success": False, "code": "DOCUMENT_MISSING"}
        return process(db, document, run_id=run_id)


@celery_app.task(bind=True, max_retries=2)
def run_invoice_pipeline(self, document_id: str, run_id: str | None = None) -> dict:
    """Queued entry point.

    No `autoretry_for=(Exception,)`: `process()` handles its own failures and
    records them, so a blanket retry would re-run a document that already
    failed deterministically — three times the OCR cost for the same answer.
    Retries are reserved for the case where the task itself dies before
    `process()` can record anything.
    """
    return run_invoice_pipeline_now(document_id, run_id)
