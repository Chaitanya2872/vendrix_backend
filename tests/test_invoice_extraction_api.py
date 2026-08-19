"""Upload, status and reprocess endpoints.

The pipeline itself is stubbed out in most of these: OCR takes minutes, and
what needs pinning here is the endpoint contract — what a client sends, what
comes back, and which HTTP code distinguishes "wrong kind of file" from "file
too big" from "file is broken".
"""
import io
from uuid import uuid4

import pytest
from PIL import Image

from app.modules.documents import repository as document_repository
from app.modules.documents.stages import STAGE_OCR, STATUS_FAILED, STATUS_PROCESSING


@pytest.fixture(autouse=True)
def no_background_processing(monkeypatch):
    """Stop uploads from actually running the pipeline.

    Without this every upload test pays for a real OCR pass, and the suite
    stops being runnable on a laptop.
    """
    import app.modules.invoices.extraction_router as extraction_router

    queued: list[tuple[str, str]] = []
    monkeypatch.setattr(
        extraction_router, "_enqueue",
        lambda background_tasks, document_id, run_id: queued.append((document_id, run_id)),
    )
    return queued


def png_bytes(width=800, height=1000):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def pdf_bytes(pages=1, marker=None):
    """A valid PDF whose bytes are unique per call unless `marker` is pinned.

    Uniqueness by default matters because the upload endpoint deduplicates on
    content hash: identical payloads across tests would make whichever ran
    second take the duplicate path and queue nothing, failing for a reason
    that has nothing to do with what it was testing. The dedupe tests below
    reuse one payload deliberately.
    """
    import fitz

    marker = marker if marker is not None else uuid4().hex
    document = fitz.open()
    for _ in range(pages):
        document.new_page().insert_text((72, 72), f"Tax Invoice {marker}")
    payload = document.tobytes()
    document.close()
    return payload


def upload(client, payload, filename, content_type, **data):
    return client.post(
        "/api/v1/invoices/upload",
        files={"file": (filename, payload, content_type)},
        data=data,
    )


class TestUploadAcceptance:
    def test_a_valid_pdf_is_accepted_and_queued(self, client, no_background_processing):
        response = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf")

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == STATUS_PROCESSING
        assert body["filename"] == "invoice.pdf"
        assert body["file_format"] == "pdf"
        assert len(no_background_processing) == 1, "the document was never queued"

    def test_the_document_id_is_the_human_readable_number(self, client):
        body = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()
        assert body["document_id"].startswith("DOC-")
        year, sequence = body["document_id"].split("-")[1:]
        assert len(year) == 4 and len(sequence) == 6

    def test_numbers_are_handed_out_sequentially_without_reuse(self, client):
        numbers = [
            upload(client, pdf_bytes(pages=index + 1), f"invoice{index}.pdf", "application/pdf").json()["document_id"]
            for index in range(3)
        ]
        assert len(set(numbers)) == 3
        assert numbers == sorted(numbers)

    @pytest.mark.parametrize(
        ("filename", "content_type"),
        [("invoice.pdf", "application/pdf"), ("invoice.png", "image/png"), ("invoice.jpg", "image/jpeg")],
    )
    def test_every_format_the_brief_names_is_accepted(self, client, filename, content_type):
        payload = pdf_bytes() if filename.endswith(".pdf") else _image(filename)
        assert upload(client, payload, filename, content_type).status_code == 201

    def test_a_multi_page_pdf_reports_its_page_count_up_front(self, client):
        body = upload(client, pdf_bytes(pages=7), "invoice.pdf", "application/pdf").json()
        assert body["page_count"] == 7

    def test_the_response_does_not_pretend_to_carry_extracted_fields(self, client):
        # OCR has not run yet. A client shaped around fields in this response
        # breaks the first time someone uploads a ten-page scan.
        body = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()
        assert "invoice_number" not in body and "total_amount" not in body


class TestUploadRejection:
    def test_an_unsupported_type_is_415(self, client):
        response = upload(client, b"plain text content" * 10, "notes.txt", "text/plain")
        assert response.status_code == 415
        assert response.json()["detail"]["code"] == "UNSUPPORTED_EXTENSION"

    def test_an_oversized_file_is_413(self, client, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "max_upload_size_mb", 0)
        response = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf")
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"

    def test_a_corrupt_file_is_422_not_415(self, client):
        # The right kind of file, broken. A client should retry a different
        # scan, not a different format. A fixed marker keeps the truncation
        # point deterministic — with a varying-length payload the cut lands
        # somewhere different each run and the test goes intermittent.
        payload = pdf_bytes(marker="fixed-for-truncation")
        response = upload(client, payload[:150], "invoice.pdf", "application/pdf")
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "CORRUPT_PDF"

    def test_a_renamed_file_is_rejected_with_the_mismatch_named(self, client):
        response = upload(client, png_bytes(), "invoice.pdf", "application/pdf")
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "CONTENT_EXTENSION_MISMATCH"

    def test_an_empty_file_is_rejected(self, client):
        assert upload(client, b"", "invoice.pdf", "application/pdf").status_code == 422

    def test_a_rejected_upload_leaves_nothing_queued(self, client, no_background_processing):
        upload(client, b"plain text" * 10, "notes.txt", "text/plain")
        assert no_background_processing == []


class TestDeduplication:
    def test_re_uploading_the_same_file_returns_the_first_document(self, client, no_background_processing):
        payload = pdf_bytes()
        first = upload(client, payload, "invoice.pdf", "application/pdf").json()
        second = upload(client, payload, "invoice-copy.pdf", "application/pdf").json()

        assert second["document_id"] == first["document_id"]
        assert second["duplicate_of"] == first["document_id"]
        assert len(no_background_processing) == 1, "OCR must not run twice for identical bytes"

    def test_duplicate_detection_can_be_overridden(self, client, no_background_processing):
        payload = pdf_bytes()
        first = upload(client, payload, "invoice.pdf", "application/pdf").json()
        second = upload(
            client, payload, "invoice.pdf", "application/pdf", reprocess_duplicates="true"
        ).json()

        assert second["document_id"] != first["document_id"]
        assert len(no_background_processing) == 2

    def test_different_files_are_not_treated_as_duplicates(self, client):
        first = upload(client, pdf_bytes(pages=1), "a.pdf", "application/pdf").json()
        second = upload(client, pdf_bytes(pages=2), "b.pdf", "application/pdf").json()
        assert first["document_id"] != second["document_id"]


class TestStatus:
    def test_a_freshly_uploaded_document_reports_zero_progress(self, client):
        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]

        body = client.get(f"/api/v1/invoices/{document_id}/status").json()
        assert body["document_id"] == document_id
        assert body["progress"] == 0
        assert body["current_stage"] == "UPLOADED"

    def test_the_payload_carries_both_lifecycle_and_stage(self, client, db):
        """The brief's example: status PROCESSING, current_stage
        TABLE_EXTRACTION. One field cannot answer both questions."""
        from app.modules.documents.service import ProcessingTracker

        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)
        run = document_repository.latest_run(db, document.id)
        ProcessingTracker(db, run).stage(STAGE_OCR)

        body = client.get(f"/api/v1/invoices/{document_id}/status").json()
        assert body["status"] == STATUS_PROCESSING
        assert body["current_stage"] == STAGE_OCR
        assert 0 < body["progress"] < 100
        assert body["stage_label"] == "Reading text"

    def test_a_failure_is_reported_with_a_code_and_a_message(self, client, db):
        from app.modules.documents.service import ProcessingTracker

        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)
        run = document_repository.latest_run(db, document.id)
        ProcessingTracker(db, run).fail("OCR_UNAVAILABLE", "PaddleOCR could not be initialised")

        body = client.get(f"/api/v1/invoices/{document_id}/status").json()
        assert body["status"] == STATUS_FAILED
        assert body["error"]["code"] == "OCR_UNAVAILABLE"
        assert body["error"]["message"]

    def test_status_can_be_read_by_internal_id_as_well(self, client, db):
        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)

        assert client.get(f"/api/v1/invoices/{document.id}/status").status_code == 200

    def test_an_unknown_document_is_404(self, client):
        response = client.get("/api/v1/invoices/DOC-1999-000001/status")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "NOT_FOUND"


class TestReprocess:
    def test_a_finished_document_can_be_reprocessed_as_a_new_attempt(self, client, db, no_background_processing):
        from app.modules.documents.service import ProcessingTracker
        from app.modules.documents.stages import STATUS_COMPLETED

        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)
        ProcessingTracker(db, document_repository.latest_run(db, document.id)).finish(STATUS_COMPLETED)

        response = client.post(f"/api/v1/invoices/{document_id}/reprocess")
        assert response.status_code == 202
        assert response.json()["attempt"] == 2
        assert len(no_background_processing) == 2

    def test_reprocessing_an_in_flight_document_is_refused(self, client, db):
        from app.modules.documents.service import ProcessingTracker

        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)
        ProcessingTracker(db, document_repository.latest_run(db, document.id)).stage(STAGE_OCR)

        response = client.post(f"/api/v1/invoices/{document_id}/reprocess")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "ALREADY_PROCESSING"

    def test_reprocessing_a_document_whose_file_is_gone_is_410(self, client, db):
        from pathlib import Path

        from app.core.config import settings
        from app.modules.documents.service import ProcessingTracker
        from app.modules.documents.stages import STATUS_COMPLETED

        document_id = upload(client, pdf_bytes(), "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)
        ProcessingTracker(db, document_repository.latest_run(db, document.id)).finish(STATUS_COMPLETED)
        (Path(settings.storage_path) / document.object_key).unlink()

        response = client.post(f"/api/v1/invoices/{document_id}/reprocess")
        assert response.status_code == 410
        assert response.json()["detail"]["code"] == "FILE_MISSING"

    def test_reprocessing_an_unknown_document_is_404(self, client):
        assert client.post("/api/v1/invoices/DOC-1999-000001/reprocess").status_code == 404


class TestStorage:
    def test_the_uploaded_bytes_land_on_disk_under_the_document_number(self, client, db):
        from pathlib import Path

        from app.core.config import settings

        payload = pdf_bytes()
        document_id = upload(client, payload, "invoice.pdf", "application/pdf").json()["document_id"]
        document = document_repository.find_document(db, document_id)

        stored = Path(settings.storage_path) / document.object_key
        assert stored.exists()
        assert stored.read_bytes() == payload
        assert document_id in document.object_key

    def test_a_traversal_filename_cannot_escape_the_storage_root(self, client, db):
        from pathlib import Path

        from app.core.config import settings

        response = upload(client, pdf_bytes(), "../../../../evil.pdf", "application/pdf")
        assert response.status_code == 201
        document = document_repository.find_document(db, response.json()["document_id"])
        assert ".." not in document.object_key
        resolved = (Path(settings.storage_path) / document.object_key).resolve()
        assert resolved.is_relative_to(Path(settings.storage_path).resolve())


def _image(filename):
    buffer = io.BytesIO()
    image = Image.new("RGB", (800, 1000), (255, 255, 255))
    image.save(buffer, format="JPEG" if filename.endswith((".jpg", ".jpeg")) else "PNG")
    return buffer.getvalue()
