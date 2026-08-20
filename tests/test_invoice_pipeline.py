"""The pipeline end to end, driving the real state machine.

Only text-PDF samples here — they exercise every stage transition without
paying for OCR. The OCR path itself is covered by the OCR service tests with
the engine stubbed, and by the format tests in test_invoice_parser_service.
"""
import pytest

from app.modules.documents import repository as document_repository
from app.modules.documents.stages import (
    STAGE_DONE,
    STATUS_FAILED,
    STATUS_PROCESSING,
    is_terminal,
)
from app.modules.invoices.services.invoice_pipeline import process


@pytest.fixture(autouse=True)
def clean_up_invoices(db):
    """Remove the invoices these tests persist.

    The schema is session-scoped, so rows outlive the test that made them.
    Every sample here is the *same* invoice, and duplicate detection works on
    vendor GSTIN + invoice number — so a leaked row makes the next test's
    parse take the duplicate path and create nothing, which surfaces as an
    unrelated test failing somewhere later in the run.
    """
    from app.models import Invoice
    from app.modules.invoices.models import InvoiceLineItem

    yield

    db.rollback()
    invoice_ids = [row.id for row in db.query(Invoice).all()]
    if invoice_ids:
        db.query(InvoiceLineItem).filter(InvoiceLineItem.invoice_id.in_(invoice_ids)).delete(
            synchronize_session=False
        )
        db.query(Invoice).filter(Invoice.id.in_(invoice_ids)).delete(synchronize_session=False)
        db.commit()


@pytest.fixture
def run_pipeline(db):
    def _run(document):
        return process(db, document)
    return _run


class TestHappyPath:
    def test_a_text_pdf_runs_to_a_terminal_status(self, stored_document, run_pipeline, db):
        document = stored_document("text_pdf")

        result = run_pipeline(document)

        assert result["success"] is True
        run = document_repository.latest_run(db, document.id)
        assert is_terminal(run.status)
        assert run.progress == 100
        assert run.current_stage == STAGE_DONE

    def test_the_run_records_what_it_learned_about_the_document(self, stored_document, run_pipeline, db):
        document = stored_document("text_pdf")

        run_pipeline(document)

        run = document_repository.latest_run(db, document.id)
        assert run.page_count == 1
        assert run.used_ocr is False, "a text PDF must not be sent through OCR"
        assert run.extraction_confidence is not None
        assert run.started_at is not None and run.finished_at is not None

    def test_every_stage_is_timed(self, stored_document, run_pipeline, db):
        document = stored_document("text_pdf")

        run_pipeline(document)

        timings = document_repository.latest_run(db, document.id).stage_timings
        assert timings, "no stage timings were recorded"
        assert all(value >= 0 for value in timings.values())

    def test_extracted_fields_land_on_the_document(self, stored_document, run_pipeline, db):
        from tests.invoice_samples import EXPECTED_FIELDS

        document = stored_document("text_pdf")

        run_pipeline(document)

        db.refresh(document)
        for key, expected in EXPECTED_FIELDS.items():
            assert document.extracted_fields[key] == expected


class TestFailurePaths:
    def test_a_missing_file_fails_the_run_rather_than_hanging_it(self, stored_document, run_pipeline, db):
        from pathlib import Path

        from app.core.config import settings

        document = stored_document("text_pdf")
        (Path(settings.storage_path) / document.object_key).unlink()

        result = run_pipeline(document)

        assert result["success"] is False
        run = document_repository.latest_run(db, document.id)
        assert run.status == STATUS_FAILED
        assert run.error_code == "FILE_MISSING"

    def test_a_corrupt_document_fails_with_a_reason(self, stored_document, run_pipeline, db):
        document = stored_document("corrupt_pdf")

        result = run_pipeline(document)

        run = document_repository.latest_run(db, document.id)
        assert run.status == STATUS_FAILED
        assert run.error_code == "UNREADABLE_DOCUMENT"
        assert run.error_message
        assert result["success"] is False

    def test_a_blank_scan_needs_review_rather_than_failing(self, stored_document, run_pipeline, db):
        """A blank page is a real thing a user uploads. The right answer is
        'we found nothing, please check', not an error the UI renders as a
        bug in the system."""
        document = stored_document("blank_png")

        result = run_pipeline(document)

        run = document_repository.latest_run(db, document.id)
        assert run.status != STATUS_FAILED
        assert is_terminal(run.status)
        assert result["success"] is True

    def test_an_extraction_with_no_invoice_number_is_kept_for_review(self, stored_document, run_pipeline, db):
        """`invoices.invoice_number` is NOT NULL, so there is no invoice row
        to create — but the extraction still belongs to the reviewer rather
        than being thrown away with a failed run."""
        from app.models import Invoice

        document = stored_document("blank_png")

        result = run_pipeline(document)

        assert result.get("review_required") is True
        assert db.query(Invoice).filter_by(document_id=document.id).one_or_none() is None
        db.refresh(document)
        assert document.extracted_fields is not None

    def test_an_unexpected_error_is_recorded_not_swallowed(self, stored_document, db, monkeypatch):
        """A run left in PROCESSING is the one state a polling client cannot
        recover from, so nothing may escape uncaught.

        Both extraction paths are broken here: the structured one falls back
        on error by design, so breaking only it would prove nothing.
        """
        import app.modules.invoices.services.invoice_pipeline as pipeline

        def explode(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(pipeline.structured_extraction_service, "extract", explode)
        monkeypatch.setattr(pipeline, "extract_document", explode)
        document = stored_document("text_pdf")

        result = process(db, document)

        run = document_repository.latest_run(db, document.id)
        assert run.status == STATUS_FAILED
        assert run.error_code == "PIPELINE_ERROR"
        assert result["success"] is False

    def test_a_broken_structured_path_falls_back_rather_than_failing(self, stored_document, db, monkeypatch):
        """The text parser is a complete extractor. A document the geometry
        path cannot handle should still be extracted, not failed."""
        import app.modules.invoices.services.invoice_pipeline as pipeline

        def explode(*args, **kwargs):
            raise RuntimeError("structured path is broken")

        monkeypatch.setattr(pipeline.structured_extraction_service, "extract", explode)
        document = stored_document("text_pdf")

        result = process(db, document)

        assert result["success"] is True
        assert document_repository.latest_run(db, document.id).status != STATUS_FAILED
        db.refresh(document)
        assert document.extracted_fields["invoice_number"]

    def test_an_incomplete_structured_result_falls_back(self, stored_document, db, monkeypatch):
        """A successful OCR call is not necessarily a usable extraction."""
        import app.modules.invoices.services.invoice_pipeline as pipeline
        from decimal import Decimal
        from types import SimpleNamespace
        from app.modules.invoices.dto import ParsedInvoiceResult

        incomplete = SimpleNamespace(
            parsed=ParsedInvoiceResult(total_amount=Decimal("128136")),
            confidence=SimpleNamespace(document_confidence=0.31),
        )
        monkeypatch.setattr(pipeline.structured_extraction_service, "extract", lambda *a, **k: incomplete)
        document = stored_document("text_pdf")

        result = process(db, document)

        assert result["success"] is True
        db.refresh(document)
        assert document.extracted_fields["invoice_number"]

    def test_an_unavailable_ocr_engine_is_distinguished_from_a_bad_document(self, stored_document, db, monkeypatch):
        # Operational versus per-document: an operator reading a list of
        # failures needs to tell "this scan is bad" from "OCR is down". This
        # one must NOT fall back — a deployment with no OCR should say so.
        import app.modules.invoices.services.invoice_pipeline as pipeline
        from app.modules.ocr.exceptions import OcrEngineUnavailable

        def unavailable(*args, **kwargs):
            raise OcrEngineUnavailable("paddle not installed")

        monkeypatch.setattr(pipeline.structured_extraction_service, "extract", unavailable)
        document = stored_document("text_pdf")

        process(db, document)

        assert document_repository.latest_run(db, document.id).error_code == "OCR_UNAVAILABLE"


class TestReruns:
    def test_a_second_run_is_a_new_attempt_and_the_first_is_preserved(self, stored_document, db):
        document = stored_document("text_pdf")

        process(db, document)
        first = document_repository.latest_run(db, document.id)
        first_id, first_status = first.id, first.status

        second_run = document_repository.create_run(db, document.id)
        db.commit()
        process(db, document, run_id=second_run.id)

        latest = document_repository.latest_run(db, document.id)
        assert latest.attempt == 2
        assert latest.id != first_id
        # The earlier attempt still says what happened last time — which is
        # the whole point of a run table rather than columns on the document.
        assert document_repository.get_run(db, first_id).status == first_status


class TestProgressVisibility:
    def test_progress_is_observable_while_the_pipeline_runs(self, stored_document, db, monkeypatch):
        """Progress written only at the end tells the user nothing during the
        one period they are watching."""
        from app.db.session import SessionLocal
        from app.modules.documents.models import DocumentProcessingRun

        observed: list[tuple[str, int, str]] = []
        document = stored_document("text_pdf")
        run = document_repository.create_run(db, document.id)
        db.commit()
        run_id = run.id

        original = document_repository.save_progress

        def spy(session, tracked_run, stage, progress, status, stage_timings=None):
            original(session, tracked_run, stage, progress, status, stage_timings)
            with SessionLocal() as observer:
                row = observer.get(DocumentProcessingRun, run_id)
                observed.append((row.current_stage, row.progress, row.status))

        monkeypatch.setattr(document_repository, "save_progress", spy)
        process(db, document, run_id=run_id)

        assert observed, "no progress was published"
        assert [entry[1] for entry in observed] == sorted(entry[1] for entry in observed), "progress went backwards"
        assert any(entry[2] == STATUS_PROCESSING for entry in observed)
        stages_seen = [entry[0] for entry in observed]
        assert "FIELD_EXTRACTION" in stages_seen
        assert "TABLE_EXTRACTION" in stages_seen
