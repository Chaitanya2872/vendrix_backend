"""`ProcessingTracker` — the handle pipeline stages report progress through.

Every stage of the pipeline needs to say "I have started", "I am 3 pages of
10 through" and "I am done", and none of them should have to know about
sessions, commits, percentage arithmetic or the run row. They get a tracker
and call `stage()` and `substep()` on it.

The tracker is also where stage timings are collected, which is what makes
the progress weights in stages.py maintainable rather than guesses that rot.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from app.modules.documents import repository
from app.modules.documents.models import DocumentProcessingRun
from app.modules.documents.stages import (
    STAGE_DONE,
    STATUS_COMPLETED,
    STATUS_FAILED,
    is_terminal,
    progress_for,
    status_for_stage,
)

logger = logging.getLogger(__name__)


class ProcessingTracker:
    """Publishes pipeline progress for one run."""

    def __init__(self, db: Session, run: DocumentProcessingRun, document_number: str | None = None) -> None:
        self._db = db
        self._run = run
        self._document_number = document_number or run.document_id
        self._timings: dict[str, float] = dict(run.stage_timings or {})
        self._current_stage: str | None = None
        self._stage_started_at: float | None = None

    @property
    def run(self) -> DocumentProcessingRun:
        return self._run

    @property
    def document_id(self) -> str:
        return self._run.document_id

    def start(self) -> None:
        repository.mark_started(self._db, self._run)
        logger.info(
            "pipeline.started document=%s run=%s attempt=%s",
            self._document_number, self._run.id, self._run.attempt,
        )

    def stage(self, name: str) -> None:
        """Enter a stage. Closes the timing of whichever stage was open."""
        self._close_timing()
        self._current_stage = name
        self._stage_started_at = time.monotonic()
        repository.save_progress(
            self._db, self._run, stage=name, progress=progress_for(name),
            status=status_for_stage(name), stage_timings=self._timings,
        )
        logger.info("pipeline.stage document=%s stage=%s", self._document_number, name)

    def substep(self, completed: int, total: int) -> None:
        """Report intra-stage progress, e.g. page 3 of 10 through OCR.

        Silently ignored outside a stage rather than raising: a stage that
        reports sub-progress it forgot to open should not take down a running
        document over a progress bar.
        """
        if self._current_stage is None or total <= 0:
            return
        repository.save_progress(
            self._db, self._run,
            stage=self._current_stage,
            progress=progress_for(self._current_stage, completed / total),
            status=status_for_stage(self._current_stage),
            stage_timings=self._timings,
        )

    @contextmanager
    def stage_scope(self, name: str) -> Iterator["ProcessingTracker"]:
        """Run a block as a stage, closing its timing even if it raises."""
        self.stage(name)
        try:
            yield self
        finally:
            self._close_timing()
            self._persist_timings()

    def finish(
        self,
        status: str = STATUS_COMPLETED,
        page_count: int | None = None,
        used_ocr: bool | None = None,
        ocr_confidence: float | None = None,
        extraction_confidence: float | None = None,
    ) -> None:
        if not is_terminal(status):
            raise ValueError(f"{status!r} is not a terminal status")
        self._close_timing()
        repository.mark_finished(
            self._db, self._run,
            status=status, stage=STAGE_DONE, progress=100,
            stage_timings=self._timings,
            page_count=page_count, used_ocr=used_ocr,
            ocr_confidence=ocr_confidence, extraction_confidence=extraction_confidence,
        )
        logger.info(
            "pipeline.finished document=%s status=%s seconds=%.1f timings=%s",
            self._document_number, status, sum(self._timings.values()), self._timings,
        )

    def fail(self, code: str, message: str) -> None:
        self._close_timing()
        repository.mark_failed(
            self._db, self._run, code=code, message=message, stage=self._current_stage
        )

    def _persist_timings(self) -> None:
        """Write accumulated timings without touching stage or status.

        `_close_timing` only updates memory, and every other write happens on
        the *next* stage transition. A stage that raised has no next
        transition, so without this the measurement of the stage that failed
        — the one actually worth having — is the one that gets lost.
        """
        self._run.stage_timings = dict(self._timings)
        try:
            self._db.commit()
        except Exception:
            # The failing block may have left the session unusable. Timings
            # are diagnostic; losing them must never mask the real error.
            self._db.rollback()
            logger.debug("pipeline.timings_not_persisted stage=%s", self._current_stage)

    def _close_timing(self) -> None:
        if self._current_stage is not None and self._stage_started_at is not None:
            elapsed = time.monotonic() - self._stage_started_at
            self._timings[self._current_stage] = round(
                self._timings.get(self._current_stage, 0.0) + elapsed, 3
            )
        self._stage_started_at = None


def status_payload(document, run: DocumentProcessingRun | None) -> dict:
    """The GET /status response body.

    Built here rather than in the router because the pipeline's own logging
    and the Celery task result use the same shape, and three copies of it
    would drift.
    """
    from app.modules.documents.stages import STATUS_UPLOADED, stage_label

    if run is None:
        # A document with no run has been accepted but not yet picked up.
        return {
            "document_id": document.document_number or document.id,
            "filename": document.filename,
            "status": STATUS_UPLOADED,
            "progress": 0,
            "current_stage": "UPLOADED",
            "stage_label": stage_label("UPLOADED"),
        }

    payload = {
        "document_id": document.document_number or document.id,
        "filename": document.filename,
        "status": run.status,
        "progress": run.progress,
        "current_stage": run.current_stage,
        "stage_label": stage_label(run.current_stage),
        "attempt": run.attempt,
        "page_count": run.page_count,
        "used_ocr": run.used_ocr,
    }
    if run.status == STATUS_FAILED:
        payload["error"] = {"code": run.error_code, "message": run.error_message}
    if run.ocr_confidence is not None:
        payload["ocr_confidence"] = float(run.ocr_confidence)
    if run.extraction_confidence is not None:
        payload["extraction_confidence"] = float(run.extraction_confidence)
    if run.duration_seconds is not None:
        payload["duration_seconds"] = round(run.duration_seconds, 1)
    return payload
