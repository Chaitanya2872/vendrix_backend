"""The endpoints the review screen depends on: result, page image, corrections.

These are the contract between the extraction pipeline and the human who has
to check its work, so what matters is that everything the screen needs
arrives together, that evidence boxes can be mapped onto a rendered page, and
that a correction is recorded rather than silently absorbed.
"""
import io
from uuid import uuid4

import pytest
from PIL import Image

from app.modules.documents import repository as document_repository


@pytest.fixture(autouse=True)
def no_background_processing(monkeypatch):
    import app.modules.invoices.extraction_router as extraction_router

    monkeypatch.setattr(
        extraction_router, "_enqueue",
        lambda background_tasks, document_id, run_id: None,
    )


def pdf_bytes(pages=1, marker=None):
    import fitz

    marker = marker if marker is not None else uuid4().hex
    document = fitz.open()
    for _ in range(pages):
        document.new_page().insert_text((72, 72), f"Tax Invoice {marker}")
    payload = document.tobytes()
    document.close()
    return payload


def png_bytes(width=800, height=1000):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client, payload, filename, content_type):
    return client.post(
        "/api/v1/invoices/upload",
        files={"file": (filename, payload, content_type)},
    )


@pytest.fixture
def uploaded(client):
    def _upload(payload=None, filename="invoice.pdf", content_type="application/pdf"):
        response = upload(client, payload or pdf_bytes(), filename, content_type)
        assert response.status_code == 201, response.text
        return response.json()["document_id"]
    return _upload


@pytest.fixture
def extracted(client, db, uploaded):
    """A document carrying a finished extraction, evidence and confidence."""
    def _make(fields=None):
        document_id = uploaded()
        document = document_repository.find_document(db, document_id)
        document.extracted_fields = fields if fields is not None else {
            "invoice_number": "INV-42",
            "total_amount": 11800.0,
            "evidence": {
                "invoice_number": {
                    "box": {"x": 10, "y": 20, "width": 100, "height": 30},
                    "page_number": 1, "score": 0.95, "reasons": ["label match"],
                },
            },
            "confidence": {"document_confidence": 0.91, "needs_review": False, "fields": {}},
        }
        db.commit()
        return document_id, document
    return _make


class TestExtractionResult:
    def test_fields_evidence_and_confidence_arrive_in_one_call(self, client, extracted):
        """The review screen is useless until it has all three, so splitting
        them would only add round trips to the same wait."""
        document_id, _ = extracted()

        body = client.get(f"/api/v1/invoices/{document_id}/result").json()

        assert body["fields"]["invoice_number"] == "INV-42"
        assert body["evidence"]["invoice_number"]["box"]["x"] == 10
        assert body["confidence"]["document_confidence"] == 0.91

    def test_evidence_and_confidence_are_lifted_out_of_the_field_bag(self, client, extracted):
        document_id, _ = extracted()

        body = client.get(f"/api/v1/invoices/{document_id}/result").json()

        assert "evidence" not in body["fields"], "the client should not have to filter these out"
        assert "confidence" not in body["fields"]

    def test_a_document_with_no_extraction_yet_still_answers(self, client, uploaded):
        document_id = uploaded()

        response = client.get(f"/api/v1/invoices/{document_id}/result")

        assert response.status_code == 200
        assert response.json()["fields"] == {}

    def test_the_result_reports_how_the_document_was_read(self, client, extracted):
        document_id, _ = extracted()

        body = client.get(f"/api/v1/invoices/{document_id}/result").json()

        assert "used_ocr" in body
        assert "page_count" in body
        assert body["filename"] == "invoice.pdf"

    def test_an_unknown_document_is_404(self, client):
        assert client.get("/api/v1/invoices/DOC-1999-000001/result").status_code == 404


class TestPageImage:
    def test_a_page_renders_as_a_png(self, client, uploaded):
        document_id = uploaded()

        response = client.get(f"/api/v1/invoices/{document_id}/pages/1")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_both_scales_are_reported_so_boxes_can_be_mapped(self, client, uploaded):
        """Evidence boxes are in extraction-DPI pixels but the browser is
        shown a smaller render. Without both numbers the overlay cannot line
        up with the text it is pointing at."""
        document_id = uploaded()

        response = client.get(f"/api/v1/invoices/{document_id}/pages/1?dpi=150")

        assert response.headers["X-Render-Dpi"] == "150"
        assert int(response.headers["X-Extraction-Dpi"]) > 0

    def test_a_page_beyond_the_end_is_404(self, client, uploaded):
        document_id = uploaded(pdf_bytes(pages=2))

        assert client.get(f"/api/v1/invoices/{document_id}/pages/9").status_code == 404

    def test_the_requested_dpi_is_clamped(self, client, uploaded):
        # A caller asking for 2000 DPI would be handed hundreds of megabytes
        # to draw a thumbnail.
        document_id = uploaded()

        response = client.get(f"/api/v1/invoices/{document_id}/pages/1?dpi=2000")

        assert int(response.headers["X-Render-Dpi"]) <= 300

    def test_a_raster_upload_also_renders(self, client, uploaded):
        document_id = uploaded(png_bytes(), "invoice.png", "image/png")

        assert client.get(f"/api/v1/invoices/{document_id}/pages/1").status_code == 200

    def test_a_second_page_renders_independently(self, client, uploaded):
        document_id = uploaded(pdf_bytes(pages=3))

        assert client.get(f"/api/v1/invoices/{document_id}/pages/3").status_code == 200


class TestCorrections:
    def test_a_correction_updates_the_extraction_and_confirms_the_document(self, client, db, extracted):
        document_id, document = extracted()

        response = client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99"}, "confirm": True},
        )

        assert response.status_code == 200
        db.refresh(document)
        assert document.extracted_fields["invoice_number"] == "INV-99"
        assert document.status == "CONFIRMED"

    def test_only_the_changed_fields_are_reported(self, client, extracted):
        document_id, _ = extracted()

        body = client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99", "total_amount": 11800.0}},
        ).json()

        assert body["corrected_fields"] == ["invoice_number"]

    def test_the_before_and_after_are_recorded_as_training_data(self, client, db, extracted):
        """The difference between what was read and what was right is the
        only real training signal this system will ever get."""
        from app.modules.documents.models import FieldCorrectionRecord

        document_id, document = extracted()

        client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99"}},
        )

        record = db.query(FieldCorrectionRecord).filter_by(document_id=document.id).one()
        assert record.corrections["invoice_number"] == {"was": "INV-42", "now": "INV-99"}

    def test_confirming_without_changes_records_no_correction(self, client, db, extracted):
        from app.modules.documents.models import FieldCorrectionRecord

        document_id, document = extracted()

        client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-42"}},
        )

        assert db.query(FieldCorrectionRecord).filter_by(document_id=document.id).count() == 0
        db.refresh(document)
        assert document.status == "CONFIRMED"

    def test_a_field_the_reviewer_left_alone_is_not_blanked(self, client, db, extracted):
        document_id, document = extracted()

        client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99"}},
        )

        db.refresh(document)
        assert document.extracted_fields["total_amount"] == 11800.0

    def test_saving_without_confirming_leaves_the_document_in_review(self, client, db, extracted):
        document_id, document = extracted()

        client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99"}, "confirm": False},
        )

        db.refresh(document)
        assert document.status != "CONFIRMED"
        assert document.extracted_fields["invoice_number"] == "INV-99"

    def test_evidence_survives_a_correction(self, client, db, extracted):
        # The reviewer corrected a value; the record of where the original
        # came from is still what explains why it was wrong.
        document_id, document = extracted()

        client.post(
            f"/api/v1/invoices/{document_id}/corrections",
            json={"fields": {"invoice_number": "INV-99"}},
        )

        db.refresh(document)
        assert "evidence" in document.extracted_fields

    def test_an_unknown_document_is_404(self, client):
        response = client.post(
            "/api/v1/invoices/DOC-1999-000001/corrections", json={"fields": {}},
        )
        assert response.status_code == 404
