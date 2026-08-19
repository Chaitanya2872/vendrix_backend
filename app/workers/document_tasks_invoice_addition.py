"""ADDITIVE content for app/workers/document_tasks.py.

The existing file already defines `process_vendor_document` (Celery task)
and `process_vendor_document_now` (the synchronous/background-task version
called when Celery isn't enabled or the queue is unavailable — see
documents/router.py). This mirrors that exact pattern for invoice parsing
instead of introducing a new queueing mechanism.

Merge the two functions below into app/workers/document_tasks.py alongside
the existing ones; do not create a separate worker module.
"""
from __future__ import annotations

import logging

from app.db.session import SessionLocal  # same session factory process_vendor_document_now already uses
from app.models import Document
from app.modules.invoices.services.invoice_parser_service import process_document

logger = logging.getLogger(__name__)

# from app.workers.celery_app import celery_app  # reuse the existing Celery app instance


def process_invoice_document_now(document_id: str) -> None:
    """Synchronous invoice-parsing entry point for BackgroundTasks / local dev,
    mirroring process_vendor_document_now's session-management style."""
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("invoice_parsing.document_missing document_id=%s", document_id)
            return
        process_document(db, document)
    except Exception:
        logger.exception("invoice_parsing.unhandled_error document_id=%s", document_id)
    finally:
        db.close()


# @celery_app.task(name="documents.process_invoice_document")
def process_invoice_document(document_id: str) -> None:
    """Celery task wrapper — decorate with the project's existing
    `celery_app.task` the same way `process_vendor_document` is decorated."""
    process_invoice_document_now(document_id)
