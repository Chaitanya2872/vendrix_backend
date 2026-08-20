"""Orchestrates the deterministic invoice parsing pipeline and hands the
result to the existing invoice service/repository for persistence.

This module is the ONLY place that knows about both the Documents module
(the `Document` ORM entity) and the Invoices module (the `Invoice` /
`InvoiceLineItem` entities, via invoice_service). Neither module needs to
know about the parser package directly:

    documents/router.py  --calls-->  invoice_parser_service.process_document
    invoice_parser_service            --calls-->  invoices/service.py

Adapt the import paths below (`app.db.session`, `app.models`, etc.) to match
wherever this actually lives in the target repo — they mirror the imports
already used in documents/router.py and invoices/router.py.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Document
from app.modules.invoices import service as invoice_service
from app.modules.invoices.dto import ParsedInvoiceResult
from app.modules.invoices.parsers import select_parser
from app.modules.invoices.parsers.text_extraction import (
    extract_document,
    UnsupportedDocumentError,
)

logger = logging.getLogger(__name__)

# Below this confidence, the created invoice is left in a review-required
# state instead of being marked ready for use. Configurable via settings so
# it can be tuned per-deployment without a code change.
REVIEW_REQUIRED_THRESHOLD = getattr(settings, "invoice_parsing_confidence_threshold", 0.80)

# Document types that should trigger invoice parsing. Reuses whatever
# document_type convention already exists on the Document model — this list
# is intentionally a single source of truth so callers don't hard-code the
# string in multiple places.
INVOICE_DOCUMENT_TYPES = {"INVOICE"}


def should_parse_as_invoice(document: Document) -> bool:
    return (document.document_type or "").upper() in INVOICE_DOCUMENT_TYPES


def _plain(value: Any) -> Any:
    """Coerce parser values into something JSON-serialisable, since
    Document.extracted_fields is a JSON column."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def as_extracted_fields(parsed: ParsedInvoiceResult) -> dict:
    """Flatten a ParsedInvoiceResult into the JSON shape stored on
    Document.extracted_fields and rendered by the document preview UI.

    Nested parties are flattened to vendor_*/customer_* keys because the
    review form edits one field per input; the full relational structure
    lives on the Invoice row created alongside this.
    """
    return {
        "invoice_number": parsed.invoice_number,
        "invoice_date": _plain(parsed.invoice_date),
        "due_date": _plain(parsed.due_date),
        "vendor_name": parsed.vendor.name,
        "vendor_gstin": parsed.vendor.gstin,
        "customer_name": parsed.customer.name,
        "customer_gstin": parsed.customer.gstin,
        "subtotal": _plain(parsed.subtotal),
        "cgst_amount": _plain(parsed.cgst_amount),
        "sgst_amount": _plain(parsed.sgst_amount),
        "igst_amount": _plain(parsed.igst_amount),
        "tax_amount": _plain(parsed.tax_amount),
        "total_amount": _plain(parsed.total_amount),
        "line_items": [
            {
                "description": item.description,
                "hsn_sac": item.hsn_sac,
                "quantity": _plain(item.quantity),
                "unit": item.unit,
                "unit_price": _plain(item.unit_price),
                "gst_rate": _plain(item.gst_rate),
                "taxable_value": _plain(item.taxable_value),
                "total_amount": _plain(item.total_amount),
            }
            for item in parsed.line_items
        ],
        "parsing_confidence": parsed.parsing_confidence,
        "warnings": list(parsed.warnings),
        "validation_errors": list(parsed.validation_errors),
    }


# Stages published while parsing runs, in order. The client polls
# GET /documents/{id} and renders whichever stage is current, so these names
# are part of the UI contract — see EXTRACTION_STAGES in document_preview.tsx.
PARSING_STAGES = ("reading", "extracting_text", "detecting_fields", "saving")


def _set_stage(db: Session, document: Document, stage: str, used_ocr: bool | None = None) -> None:
    """Publish progress on the document itself so a poller can watch the
    extraction advance. `in_progress` is the discriminator that tells the UI
    this payload is a status update rather than a finished result — the final
    write from as_extracted_fields() has no such key.

    Committed immediately: a value only visible at the end of the job would
    tell the user nothing while the job is the thing they're waiting on.
    """
    document.extracted_fields = {"in_progress": True, "parsing_stage": stage, "used_ocr": used_ocr}
    document.status = "PROCESSING"
    db.commit()


def _record_failure(db: Session, document: Document, message: str) -> dict:
    """Persist a parse failure on the document itself. Without this the UI
    has no way to tell 'still processing' apart from 'processing finished
    and found nothing', and sits on a spinner forever."""
    document.extracted_fields = {"parsing_confidence": 0.0, "warnings": [message], "validation_errors": []}
    document.status = "REVIEW_REQUIRED"
    db.commit()
    return {"success": False, "error": message}


def process_document(db: Session, document: Document) -> dict:
    """Entry point called after a Document has been persisted and its file
    written to storage. Returns a plain dict describing the outcome — this
    is what document upload / background-task code should log or attach to
    its own response/notification, following whatever pattern the project
    already uses for background job results.

    On any parsing failure, the Document itself is left untouched (the
    uploaded file is never deleted or rolled back) and a warning-carrying
    result is returned instead of raising, so a bad invoice document never
    takes down the upload flow.
    """
    stored_file = Path(settings.storage_path) / document.object_key
    started_at = time.monotonic()
    logger.info("invoice_parsing.started document_id=%s", document.id)

    if not stored_file.exists():
        logger.warning("invoice_parsing.failed document_id=%s reason=file_missing", document.id)
        return _record_failure(db, document, "Stored file not found.")

    _set_stage(db, document, "reading")

    extraction_started_at = time.monotonic()
    try:
        extracted = extract_document(
            str(stored_file),
            on_stage=lambda stage: _set_stage(db, document, stage),
        )
    except UnsupportedDocumentError as exc:
        logger.warning("invoice_parsing.failed document_id=%s reason=%s", document.id, exc)
        return _record_failure(db, document, str(exc))
    extraction_seconds = time.monotonic() - extraction_started_at

    if not extracted.text.strip() and not any(extracted.tables_per_page):
        logger.warning("invoice_parsing.failed document_id=%s reason=no_readable_text", document.id)
        return _record_failure(db, document, "No readable text found in document.")

    # Only known after the fact: extract_document decides internally whether
    # the file needed OCR, and the UI names that step differently.
    _set_stage(db, document, "detecting_fields", used_ocr=extracted.used_ocr)

    parser = select_parser(extracted.text)
    logger.info("invoice_parsing.parser_selected document_id=%s parser=%s", document.id, parser.name)

    parse_started_at = time.monotonic()
    parsed: ParsedInvoiceResult = parser.parse(extracted)
    parse_seconds = time.monotonic() - parse_started_at
    _set_stage(db, document, "saving", used_ocr=extracted.used_ocr)
    # Text extraction and field parsing are logged separately because they
    # have nothing in common: extraction is dominated by OCR when OCR runs at
    # all, parsing is regex over a string. A single total tells you a
    # document was slow; these two tell you which half to look at.
    logger.info(
        "invoice_parsing.completed document_id=%s confidence=%.2f warnings=%d errors=%d "
        "ocr=%s pages=%s extraction_seconds=%.2f parse_seconds=%.2f total_seconds=%.2f",
        document.id, parsed.parsing_confidence, len(parsed.warnings), len(parsed.validation_errors),
        extracted.used_ocr, extracted.page_count, extraction_seconds, parse_seconds,
        time.monotonic() - started_at,
    )

    # Persist the extracted values on the Document before anything else can
    # fail: the preview UI reads them from here, and a duplicate or a
    # persistence problem downstream shouldn't cost the user the extraction.
    document.extracted_fields = as_extracted_fields(parsed)
    document.status = "REVIEW_REQUIRED"
    db.commit()

    duplicate = invoice_service.find_probable_duplicate(db, parsed)
    if duplicate is not None:
        logger.info("invoice_parsing.duplicate_detected document_id=%s existing_invoice_id=%s", document.id, duplicate.id)
        return {
            "success": True,
            "duplicate": True,
            "existing_invoice_id": duplicate.id,
            "parsed_invoice_confidence": parsed.parsing_confidence,
            "warnings": parsed.warnings,
        }

    status = "PARSED" if parsed.parsing_confidence >= REVIEW_REQUIRED_THRESHOLD and not parsed.validation_errors else "REVIEW_REQUIRED"

    invoice = invoice_service.create_from_parsed_document(
        db=db,
        document=document,
        parsed=parsed,
        status=status,
    )
    logger.info("invoice_parsing.invoice_persisted document_id=%s invoice_id=%s status=%s", document.id, invoice.id, status)

    return {
        "success": True,
        "duplicate": False,
        "invoice_id": invoice.id,
        "status": status,
        "parsing_confidence": parsed.parsing_confidence,
        "warnings": parsed.warnings,
        "validation_errors": parsed.validation_errors,
    }
