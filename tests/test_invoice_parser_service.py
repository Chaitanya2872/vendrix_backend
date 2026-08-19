"""Invoice parsing: field extraction, persistence, stage reporting, failures.

These exercise the pipeline through `process_invoice_document_now` — the
function the upload endpoint actually queues — rather than calling the
service directly, so the worker's error handling is part of what's covered.
"""
from datetime import date
from decimal import Decimal

import pytest

from app.db.session import SessionLocal
from app.models import Document, Invoice
from app.modules.invoices.dto import ParsedInvoiceResult, ParsedLineItem, ParsedParty
from app.modules.invoices.services import invoice_parser_service
from app.modules.invoices.services.invoice_parser_service import (
    PARSING_STAGES,
    as_extracted_fields,
    should_parse_as_invoice,
)
from app.workers.document_tasks import process_invoice_document_now

from tests.invoice_samples import EXPECTED_FIELDS


@pytest.fixture
def stages(monkeypatch):
    """Records every stage the service publishes, and — reading through a
    *separate* session — whether it was actually committed and therefore
    visible to a polling client at that moment. A stage only observable at
    the end of the job would be useless to the UI it exists to drive."""
    observed = []
    original = invoice_parser_service._set_stage

    def spy(db, document, stage, used_ocr=None):
        original(db, document, stage, used_ocr)
        with SessionLocal() as observer:
            row = observer.get(Document, document.id)
            fields = row.extracted_fields or {}
            observed.append({
                "stage": stage,
                "visible_stage": fields.get("parsing_stage"),
                "in_progress": fields.get("in_progress"),
                "status": row.status,
            })

    monkeypatch.setattr(invoice_parser_service, "_set_stage", spy)
    return observed


# ─── Serialisation ──────────────────────────────────────────────────────────

def test_as_extracted_fields_is_json_safe():
    """extracted_fields is a JSON column: Decimals and dates must be coerced
    or the commit blows up at write time."""
    import json

    parsed = ParsedInvoiceResult(
        invoice_number="INV-1",
        invoice_date=date(2024, 3, 15),
        due_date=date(2024, 4, 14),
        subtotal=Decimal("10000.00"),
        total_amount=Decimal("11800.00"),
        vendor=ParsedParty(name="Acme", gstin="29ABCDE1234F1Z5"),
        customer=ParsedParty(name="IoTIQ"),
        line_items=[ParsedLineItem(description="Widget", quantity=Decimal("2"), unit_price=Decimal("50.5"))],
        parsing_confidence=0.85,
    )

    fields = as_extracted_fields(parsed)
    json.dumps(fields)  # must not raise

    assert fields["invoice_date"] == "2024-03-15"
    assert fields["total_amount"] == 11800.0
    assert isinstance(fields["total_amount"], float)
    assert fields["vendor_name"] == "Acme"       # nested party is flattened
    assert fields["vendor_gstin"] == "29ABCDE1234F1Z5"
    assert fields["line_items"][0]["quantity"] == 2.0
    assert "in_progress" not in fields, "a finished result must not look like a progress update"


def test_as_extracted_fields_tolerates_an_empty_parse():
    fields = as_extracted_fields(ParsedInvoiceResult())

    assert fields["invoice_number"] is None
    assert fields["line_items"] == []
    assert fields["parsing_confidence"] == 0.0


# ─── Document type gating ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("document_type", "expected"),
    [("INVOICE", True), ("invoice", True), ("Invoice", True), ("CERTIFICATION", False), ("", False)],
)
def test_should_parse_as_invoice(document_type, expected):
    assert should_parse_as_invoice(Document(document_type=document_type)) is expected


# ─── Happy path ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["text_pdf", "jpg"])
def test_fields_are_persisted_for_every_format(kind, stored_document, db):
    """The core of the feature: the same invoice, uploaded as a PDF or as a
    photograph, must land the same values on the document."""
    document = stored_document(kind)

    process_invoice_document_now(document.id)

    db.refresh(document)
    fields = document.extracted_fields
    assert fields is not None
    assert not fields.get("in_progress"), "extraction finished, so this must be a result not a status"
    for key, expected in EXPECTED_FIELDS.items():
        assert fields[key] == expected, f"{key} differed for {kind}"
    assert document.status == "REVIEW_REQUIRED"


def test_pdf_and_image_of_the_same_invoice_agree(stored_document, db):
    """Guards the promise that format doesn't change the answer."""
    from_pdf = stored_document("text_pdf")
    from_image = stored_document("jpg")

    process_invoice_document_now(from_pdf.id)
    process_invoice_document_now(from_image.id)
    db.refresh(from_pdf)
    db.refresh(from_image)

    compared = ("invoice_number", "invoice_date", "vendor_gstin", "total_amount")
    assert {k: from_pdf.extracted_fields[k] for k in compared} == \
           {k: from_image.extracted_fields[k] for k in compared}


# ─── Stage reporting (drives the extraction animation) ──────────────────────

def test_stages_are_published_in_order_and_committed(stored_document, stages):
    document = stored_document("text_pdf")

    process_invoice_document_now(document.id)

    published = [entry["stage"] for entry in stages]
    assert published, "no stages were published at all"
    assert published == sorted(published, key=PARSING_STAGES.index), "stages went backwards"
    assert published[0] == "reading"
    assert "detecting_fields" in published and published[-1] == "saving"

    for entry in stages:
        assert entry["visible_stage"] == entry["stage"], "stage was not committed when published"
        assert entry["in_progress"] is True
        assert entry["status"] == "PROCESSING"


def test_ocr_stage_is_published_only_when_ocr_runs(stored_document, stages):
    """A text PDF skips OCR; the UI relies on this to say so rather than
    showing a step that never lights up."""
    process_invoice_document_now(stored_document("text_pdf").id)
    text_pdf_stages = [entry["stage"] for entry in stages]

    stages.clear()
    process_invoice_document_now(stored_document("jpg").id)
    image_stages = [entry["stage"] for entry in stages]

    assert "extracting_text" not in text_pdf_stages
    assert "extracting_text" in image_stages


def test_used_ocr_is_reported_once_known(stored_document, stages):
    process_invoice_document_now(stored_document("jpg").id)

    late_stages = [entry for entry in stages if entry["stage"] in ("detecting_fields", "saving")]
    assert late_stages, "expected stages after text extraction"


def test_progress_marker_is_cleared_when_extraction_finishes(stored_document, db):
    document = stored_document("text_pdf")

    process_invoice_document_now(document.id)

    db.refresh(document)
    assert "in_progress" not in document.extracted_fields
    assert document.status != "PROCESSING"


# ─── Failure paths ──────────────────────────────────────────────────────────

def test_unparseable_format_records_a_warning_rather_than_hanging(stored_document, db):
    """A document stuck with no fields is indistinguishable from one still
    processing, so failures have to be written down."""
    document = stored_document("unsupported")

    process_invoice_document_now(document.id)

    db.refresh(document)
    fields = document.extracted_fields
    assert fields is not None
    assert not fields.get("in_progress")
    assert fields["parsing_confidence"] == 0.0
    assert any(".txt" in warning for warning in fields["warnings"])


def test_blank_image_reports_no_readable_text(stored_document, db):
    document = stored_document("blank_png")

    process_invoice_document_now(document.id)

    db.refresh(document)
    assert any("readable text" in warning.lower() for warning in document.extracted_fields["warnings"])


def test_missing_stored_file_is_reported(stored_document, db):
    document = stored_document("text_pdf")
    from pathlib import Path

    from app.core.config import settings
    (Path(settings.storage_path) / document.object_key).unlink()

    process_invoice_document_now(document.id)

    db.refresh(document)
    assert any("not found" in warning.lower() for warning in document.extracted_fields["warnings"])


def test_unknown_document_id_does_not_raise():
    """The worker is a background task; an exception here has nowhere to go."""
    process_invoice_document_now("00000000-0000-0000-0000-000000000000")


def test_corrupt_pdf_does_not_take_down_the_worker(stored_document, db):
    document = stored_document("corrupt_pdf")

    process_invoice_document_now(document.id)

    db.refresh(document)
    assert document.extracted_fields["warnings"], "expected a recorded failure"


@pytest.fixture
def no_existing_invoices(db):
    """Start from an empty invoices table.

    Every sample in this file is the *same* invoice, and earlier tests here
    persist it. Duplicate detection then correctly refuses to create a second
    row — so without this, a test about "parsing creates an invoice" would
    really be measuring how many tests ran before it.
    """
    from app.modules.invoices.models import InvoiceLineItem

    db.rollback()
    db.query(InvoiceLineItem).delete(synchronize_session=False)
    db.query(Invoice).delete(synchronize_session=False)
    db.commit()
    return db


def test_parsing_creates_an_invoice_row(stored_document, db, no_existing_invoices):
    """`invoices.vendor_id` is nullable, so a parsed invoice persists without
    a resolved vendor. Matching to vendor master data stays a review-time
    decision — inventing a link from an OCR'd name would be worse than
    leaving it blank for a human."""
    document = stored_document("text_pdf")

    invoice_parser_service.process_document(db, document)

    assert db.query(Invoice).filter_by(document_id=document.id).one_or_none() is not None


def test_the_same_invoice_uploaded_twice_does_not_create_a_second_row(
    stored_document, db, no_existing_invoices
):
    """The other half of the same behaviour: re-uploading an invoice already
    on file is recognised rather than duplicated. Vendors resend invoices and
    mail ingests replay, so this is a routine event, not an edge case."""
    first = stored_document("text_pdf")
    invoice_parser_service.process_document(db, first)

    second = stored_document("text_pdf")
    result = invoice_parser_service.process_document(db, second)

    assert result["duplicate"] is True
    assert db.query(Invoice).count() == 1
    # The extraction still lands on the second document, so a reviewer can
    # see what was read and decide for themselves.
    db.refresh(second)
    assert second.extracted_fields["invoice_number"] == EXPECTED_FIELDS["invoice_number"]


def test_extraction_survives_the_invoice_insert_failing(stored_document, db):
    """Whatever happens downstream, the user must not lose the extraction —
    which is why fields are committed before the Invoice is attempted."""
    document = stored_document("text_pdf")

    process_invoice_document_now(document.id)

    db.refresh(document)
    assert document.extracted_fields["invoice_number"] == EXPECTED_FIELDS["invoice_number"]
