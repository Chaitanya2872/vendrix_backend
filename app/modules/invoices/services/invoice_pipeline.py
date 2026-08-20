"""The invoice processing pipeline: one document, start to finish.

This module owns the *sequence*. Each stage's actual work lives in its own
service, and this file's job is to run them in order, report progress, and
make sure a failure anywhere lands as a recorded error on the run rather than
a document stuck in PROCESSING forever — the one state a polling client
cannot recover from on its own.

Stages are swapped in as the phases land. Where a stage's dedicated service
does not exist yet, the pipeline calls the existing deterministic path and
reports the stage honestly; the stage boundary is real from the start, so
adding the service later is a substitution rather than a restructure.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Document
from app.modules.documents import repository as document_repository
from app.modules.documents.service import ProcessingTracker
from app.modules.documents.stages import (
    STAGE_FIELDS,
    STAGE_LAYOUT,
    STAGE_LINE_ITEMS,
    STAGE_OCR,
    STAGE_PDF_ANALYSIS,
    STAGE_PERSISTING,
    STAGE_TABLE,
    STAGE_VALIDATING,
    STAGE_VALIDATION,
    STATUS_COMPLETED,
    STATUS_REVIEW_REQUIRED,
)
from app.modules.invoices import service as invoice_service
from app.modules.invoices.dto import ParsedInvoiceResult
from app.modules.invoices.parsers import select_parser
from app.modules.invoices.parsers.text_extraction import (
    UnsupportedDocumentError,
    extract_document,
)
from app.modules.invoices.services import structured_extraction_service
from app.modules.invoices.services.invoice_parser_service import (
    REVIEW_REQUIRED_THRESHOLD,
    as_extracted_fields,
)
from app.modules.ocr.exceptions import OcrEngineUnavailable, OcrError

logger = logging.getLogger(__name__)

# Maps the extraction layer's internal stage names onto the pipeline's
# vocabulary. `extract_document` reports what it is doing but predates the
# stage machine; translating here keeps that module free of pipeline
# concepts it has no other reason to know about.
_EXTRACTION_STAGE_MAP = {
    "reading": STAGE_PDF_ANALYSIS,
    "extracting_text": STAGE_OCR,
    "detecting_fields": STAGE_FIELDS,
    "saving": STAGE_PERSISTING,
}


def process(db: Session, document: Document, run_id: str | None = None) -> dict:
    """Run one document through the pipeline.

    Never raises. Every failure path records itself on the run and returns a
    result dict; a background task that raised would leave the run row saying
    PROCESSING with nothing to explain it.
    """
    run = (
        document_repository.get_run(db, run_id)
        if run_id
        else document_repository.latest_run(db, document.id)
    )
    if run is None:
        run = document_repository.create_run(db, document.id)
        db.commit()

    tracker = ProcessingTracker(db, run, document_number=document.document_number)
    tracker.start()

    try:
        return _run_stages(db, document, tracker)
    except OcrEngineUnavailable as exc:
        # Operational, not per-document: worth distinguishing so an operator
        # reading failures can tell "this scan is bad" from "OCR is down".
        logger.exception("pipeline.ocr_unavailable document_id=%s", document.id)
        tracker.fail("OCR_UNAVAILABLE", str(exc))
        return {"success": False, "error": str(exc), "code": "OCR_UNAVAILABLE"}
    except Exception as exc:
        logger.exception("pipeline.unhandled_error document_id=%s", document.id)
        tracker.fail("PIPELINE_ERROR", str(exc))
        return {"success": False, "error": str(exc), "code": "PIPELINE_ERROR"}


def _run_stages(db: Session, document: Document, tracker: ProcessingTracker) -> dict:
    stored_file = Path(settings.storage_path) / document.object_key

    tracker.stage(STAGE_VALIDATION)
    if not stored_file.exists():
        tracker.fail("FILE_MISSING", "The stored file could not be found.")
        return {"success": False, "code": "FILE_MISSING"}

    tracker.stage(STAGE_PDF_ANALYSIS)

    structured = _try_structured(stored_file, tracker)
    if structured is not None:
        return _persist(db, document, tracker, structured.parsed,
                        page_count=structured.ocr_document.page_count,
                        used_ocr=structured.used_ocr,
                        ocr_confidence=structured.ocr_document.mean_confidence,
                        evidence=structured.evidence_dict(),
                        confidence=structured.confidence)

    try:
        extracted = extract_document(
            str(stored_file),
            on_stage=lambda name: tracker.stage(_EXTRACTION_STAGE_MAP.get(name, STAGE_OCR)),
        )
    except UnsupportedDocumentError as exc:
        tracker.fail("UNREADABLE_DOCUMENT", str(exc))
        return {"success": False, "code": "UNREADABLE_DOCUMENT", "error": str(exc)}

    if not extracted.text.strip() and not any(extracted.tables_per_page):
        # Not a crash: a blank scan is a real thing a user can upload, and the
        # right response is "we found nothing, please check", not an error the
        # UI renders as a bug.
        tracker.stage(STAGE_VALIDATING)
        document.status = "REVIEW_REQUIRED"
        document.extracted_fields = {
            "parsing_confidence": 0.0,
            "warnings": ["No readable text was found in this document."],
            "validation_errors": [],
        }
        db.commit()
        tracker.finish(
            STATUS_REVIEW_REQUIRED, page_count=extracted.page_count, used_ocr=extracted.used_ocr
        )
        return {"success": True, "code": "NO_TEXT_FOUND", "review_required": True}

    # Layout, table and line-item extraction are reported as distinct stages
    # even where the current parser does them in one pass: the boundaries are
    # what the phased services slot into, and a progress bar that only ever
    # showed the stages already refactored would go backwards on each release.
    tracker.stage(STAGE_LAYOUT)
    tracker.stage(STAGE_TABLE)

    tracker.stage(STAGE_FIELDS)
    parser = select_parser(extracted.text)
    parsed: ParsedInvoiceResult = parser.parse(extracted)

    tracker.stage(STAGE_LINE_ITEMS)
    tracker.stage(STAGE_VALIDATING)

    return _persist(
        db, document, tracker, parsed,
        page_count=extracted.page_count,
        used_ocr=extracted.used_ocr,
    )


def _try_structured(stored_file: Path, tracker: ProcessingTracker):
    """Run the geometry-aware path, or return None to fall back.

    Returning None rather than raising is deliberate: the text parser is a
    complete, working extractor, and a document the structured path cannot
    handle — an Office file, a page whose OCR engine is unavailable — should
    still be extracted rather than failed. The fallback is silent to the user
    and loud in the log, because a deployment where *every* document falls
    back is a broken deployment that would otherwise look merely mediocre.
    """
    if not structured_extraction_service.is_supported(stored_file):
        return None

    stage_map = {
        "reading": STAGE_PDF_ANALYSIS,
        "layout": STAGE_LAYOUT,
        "tables": STAGE_TABLE,
        "fields": STAGE_FIELDS,
        "validating": STAGE_VALIDATING,
    }

    def on_stage(name: str) -> None:
        tracker.stage(stage_map.get(name, STAGE_FIELDS))

    def on_page(completed: int, total: int) -> None:
        # OCR is minutes per page; without this the bar sits still for the
        # entire wait, which is indistinguishable from a hung worker.
        tracker.stage(STAGE_OCR)
        tracker.substep(completed, total)

    try:
        structured = structured_extraction_service.extract(
            stored_file, on_page=on_page, on_stage=on_stage
        )
        parsed = structured.parsed
        has_supplier = bool(parsed.vendor.name or parsed.vendor.gstin)
        if not parsed.invoice_number or parsed.total_amount is None or not has_supplier:
            logger.warning(
                "pipeline.structured_extraction_incomplete file=%s confidence=%.3f; "
                "falling back to text parsing",
                stored_file.name, structured.confidence.document_confidence,
            )
            return None
        return structured
    except OcrEngineUnavailable:
        raise  # operational: worth failing loudly rather than degrading silently
    except (OcrError, Exception) as exc:
        logger.warning(
            "pipeline.structured_extraction_failed file=%s error=%s; falling back to text parsing",
            stored_file.name, exc, exc_info=True,
        )
        return None


def _persist(
    db: Session,
    document: Document,
    tracker: ProcessingTracker,
    parsed: ParsedInvoiceResult,
    page_count: int,
    used_ocr: bool,
    ocr_confidence: float | None = None,
    evidence: dict | None = None,
    confidence=None,
) -> dict:
    """Store the extraction and finish the run. Shared by both paths."""
    tracker.stage(STAGE_PERSISTING)

    # Written before anything else can fail: the review UI reads its fields
    # from here, and a duplicate check or a constraint violation downstream
    # should not cost the user an extraction that already succeeded.
    fields = as_extracted_fields(parsed)
    if evidence:
        # Where each value came from, so the review screen can highlight the
        # source region on the page rather than asking the user to hunt.
        fields["evidence"] = evidence
    if confidence is not None:
        fields["confidence"] = confidence.to_dict()
    document.extracted_fields = fields
    document.status = "REVIEW_REQUIRED"
    db.commit()

    if not parsed.invoice_number:
        # `invoices.invoice_number` is NOT NULL and unique, so there is no
        # such thing as an invoice row without one. A blank page, or a scan
        # whose number OCR could not recover, therefore stops here: the
        # extraction is already saved on the document for a human to complete,
        # and attempting the insert would fail the whole run over a field the
        # reviewer was going to fill in anyway.
        logger.info(
            "pipeline.no_invoice_number document_id=%s; extraction kept for review",
            document.id,
        )
        tracker.finish(
            STATUS_REVIEW_REQUIRED,
            page_count=page_count, used_ocr=used_ocr, ocr_confidence=ocr_confidence,
            extraction_confidence=parsed.parsing_confidence,
        )
        return {
            "success": True,
            "code": "NO_INVOICE_NUMBER",
            "review_required": True,
            "parsing_confidence": parsed.parsing_confidence,
            "validation_errors": parsed.validation_errors,
        }

    duplicate = invoice_service.find_probable_duplicate(db, parsed)
    if duplicate is not None:
        logger.info(
            "pipeline.duplicate_detected document_id=%s existing_invoice_id=%s",
            document.id, duplicate.id,
        )
        tracker.finish(
            STATUS_REVIEW_REQUIRED,
            page_count=page_count, used_ocr=used_ocr, ocr_confidence=ocr_confidence,
            extraction_confidence=parsed.parsing_confidence,
        )
        return {
            "success": True,
            "duplicate": True,
            "existing_invoice_id": duplicate.id,
            "parsing_confidence": parsed.parsing_confidence,
        }

    # The confidence report's own judgement wins when it exists: it accounts
    # for validation errors capping the score, which a bare threshold on the
    # number cannot see.
    if confidence is not None:
        clean = not confidence.needs_review
    else:
        clean = (parsed.parsing_confidence >= REVIEW_REQUIRED_THRESHOLD
                 and not parsed.validation_errors)

    invoice = invoice_service.create_from_parsed_document(
        db=db, document=document, parsed=parsed, status="PARSED" if clean else "REVIEW_REQUIRED"
    )

    tracker.finish(
        STATUS_COMPLETED if clean else STATUS_REVIEW_REQUIRED,
        page_count=page_count, used_ocr=used_ocr, ocr_confidence=ocr_confidence,
        extraction_confidence=parsed.parsing_confidence,
    )
    return {
        "success": True,
        "duplicate": False,
        "invoice_id": invoice.id,
        "review_required": not clean,
        "parsing_confidence": parsed.parsing_confidence,
        "warnings": parsed.warnings,
        "validation_errors": parsed.validation_errors,
    }
