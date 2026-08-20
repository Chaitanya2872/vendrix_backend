"""Getting extraction to actually run, and noticing when it did not.

Queueing a task is not the same as it being executed, and the difference is
invisible from the request that queued it. `delay()` raises only when the
*broker* is unreachable — with Redis up and no worker consuming, the call
succeeds, the task sits in the queue, and the document stays at UPLOADED
forever while the UI waits on an extraction nobody is performing. That is the
worst failure this module has: silent, permanent, and indistinguishable from
slowness.

Three defences, in order of when they apply:

1. `dispatch_extraction` asks whether a worker is actually listening before
   handing work to the queue, and runs in-process when none is.
2. `claim_for_processing` makes re-running safe, so a retry can never have two
   workers extracting the same document at once.
3. `recover_stuck_documents` sweeps anything already stranded, so a fix
   deploys rather than needing every affected document re-uploaded.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Document
from app.modules.documents.stages import (
    STATUS_PROCESSING,
    STATUS_UPLOADED,
)
# Imported at module level, not inside the function that uses it. As well as
# the predicate, this import registers the invoice ORM models, and
# `Invoice.line_items` names `InvoiceLineItem` as a string that SQLAlchemy can
# only resolve once that class has been defined. Made lazy, every mapper
# configuration in the process fails with "failed to locate a name
# ('InvoiceLineItem')" — including on requests that never touch an invoice.
from app.modules.invoices.services.invoice_parser_service import should_parse_as_invoice

logger = logging.getLogger(__name__)

# Whether a worker answered, and when we last asked. Cached because the check
# is a broker round trip and would otherwise be paid on every upload; short,
# because a worker dying should not stay unnoticed for long.
_worker_probe: tuple[float, bool] | None = None
WORKER_PROBE_TTL_SECONDS = 30.0
WORKER_PROBE_TIMEOUT_SECONDS = 1.0


def celery_workers_available(force: bool = False) -> bool:
    """Is any Celery worker listening right now?

    Never raises: every failure mode here — broker down, ping unsupported,
    transport error — means the same thing to the caller, which is "do not
    rely on the queue".
    """
    global _worker_probe

    if not settings.celery_enabled:
        return False

    now = time.monotonic()
    if not force and _worker_probe is not None and now - _worker_probe[0] < WORKER_PROBE_TTL_SECONDS:
        return _worker_probe[1]

    available = False
    try:
        from app.workers.celery_app import celery_app

        replies = celery_app.control.ping(timeout=WORKER_PROBE_TIMEOUT_SECONDS)
        available = bool(replies)
    except Exception as exc:
        logger.warning("documents.worker_probe_failed error=%s", exc)

    if _worker_probe is None or _worker_probe[1] != available:
        logger.info("documents.worker_availability available=%s", available)
    _worker_probe = (now, available)
    return available


def _tasks_for(document: Document):
    """The queued and in-process entry points for this document's type."""
    # The worker module stays lazy: it pulls in Celery and the OCR stack, and
    # nothing here needs either until a document is actually dispatched.
    from app.workers.document_tasks import (
        process_invoice_document, process_invoice_document_now,
        process_vendor_document, process_vendor_document_now,
    )
    if should_parse_as_invoice(document):
        return process_invoice_document, process_invoice_document_now
    return process_vendor_document, process_vendor_document_now


def dispatch_extraction(document: Document, background_tasks=None) -> str:
    """Start extraction for `document`, by whichever route will actually run.

    Returns "celery" or "inline" so the caller can log what happened. When
    `background_tasks` is given the in-process route defers until after the
    response; without it the work runs synchronously, which is what a sweep
    on its own thread wants.
    """
    queued_task, inline_task = _tasks_for(document)
    document_id = document.id

    if celery_workers_available():
        try:
            queued_task.delay(document_id)
            return "celery"
        except Exception as exc:
            # The probe passed a moment ago, so this is the queue failing
            # underneath us rather than a configuration problem. Fall through.
            logger.warning("documents.enqueue_failed document_id=%s error=%s", document_id, exc)
            celery_workers_available(force=True)

    if background_tasks is not None:
        background_tasks.add_task(inline_task, document_id)
    else:
        inline_task(document_id)
    return "inline"


def claim_for_processing(db: Session, document_id: str, from_statuses: tuple[str, ...]) -> bool:
    """Atomically move a document into PROCESSING, if it is in one of
    `from_statuses`.

    A conditional UPDATE rather than read-then-write: a retry from the UI and
    a recovery sweep can arrive at the same moment, and two extractions of one
    document would race to write `extracted_fields` — the loser's results
    silently replacing the winner's.
    """
    result = db.execute(
        update(Document)
        .where(Document.id == document_id, Document.status.in_(from_statuses))
        .values(status=STATUS_PROCESSING)
    )
    db.commit()
    return result.rowcount == 1


def recover_stuck_documents(limit: int | None = None) -> int:
    """Re-run extraction for documents that were queued but never processed.

    Two cases, both meaning "no worker did this":

      UPLOADED   — accepted, never picked up.
      PROCESSING — picked up, then the worker died mid-document. Only counted
                   once it is old enough that it cannot still be running.

    Runs synchronously, so callers put it on a background thread. Bounded by
    `limit`: a queue that has been broken for a week should not turn the next
    restart into an hour of OCR before the service is usable.
    """
    from app.db.session import SessionLocal

    cap = limit if limit is not None else settings.documents_recovery_limit
    if cap <= 0:
        return 0

    stale_before = datetime.now(timezone.utc) - timedelta(minutes=settings.documents_stuck_after_minutes)
    recovered = 0

    with SessionLocal() as db:
        candidates = db.scalars(
            select(Document)
            .where(Document.status.in_((STATUS_UPLOADED, STATUS_PROCESSING)))
            .where(Document.created_at < stale_before)
            .order_by(Document.created_at.desc())
            .limit(cap)
        ).all()

        if not candidates:
            return 0

        logger.info("documents.recovery_started count=%d", len(candidates))
        for document in candidates:
            # Re-claim from whichever state it was stranded in. If something
            # else got there first the claim fails and this one is skipped.
            if not claim_for_processing(db, document.id, (STATUS_UPLOADED, STATUS_PROCESSING)):
                continue
            try:
                dispatch_extraction(document)
                recovered += 1
            except Exception:
                logger.exception("documents.recovery_failed document_id=%s", document.id)

    logger.info("documents.recovery_finished recovered=%d", recovered)
    return recovered
