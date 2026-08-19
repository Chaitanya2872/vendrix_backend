"""OCR service: fragment merging, confidence handling, page assembly.

PaddleOCR is stubbed out throughout. The engine's own accuracy is not this
project's to test, and a real OCR pass costs ~2 minutes per page — which
would make this file unrunnable as a unit test. What *is* tested is every
decision this project makes about the engine's output, since that is where
the invoice-specific correctness lives.
"""
import numpy as np
import pytest

from app.modules.ocr import service as ocr_service
from app.modules.ocr.engine import RawDetection
from app.modules.ocr.exceptions import OcrEngineUnavailable, OcrPageFailed


def detection(text, x0, y0, x1, y1, confidence=0.98):
    return RawDetection(text, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], confidence)


@pytest.fixture
def blank_page():
    return np.zeros((1000, 800, 3), dtype=np.uint8)


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace the engine with a canned list of detections."""
    def _install(detections):
        monkeypatch.setattr(ocr_service.engine, "recognize", lambda image, language="en": list(detections))
    return _install


class TestFragmentMerging:
    def test_label_and_value_separated_by_a_space_become_one_line(self, stub_engine, blank_page):
        # The detector splits these into two regions; leaving them split
        # loses the label→value adjacency field extraction runs on.
        stub_engine([
            detection("Invoice No:", 50, 100, 160, 120),
            detection("INV-2026-0042", 175, 100, 320, 120),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["Invoice No: INV-2026-0042"]

    def test_distant_table_columns_stay_separate(self, stub_engine, blank_page):
        # Same y-band, but a column gutter apart. Merging these would glue a
        # description to an amount and produce a line that means nothing.
        stub_engine([
            detection("Steel Fabrication Work", 50, 300, 300, 320),
            detection("1,92,407.04", 650, 300, 760, 320),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["Steel Fabrication Work", "1,92,407.04"]

    def test_stacked_rows_are_never_merged(self, stub_engine, blank_page):
        stub_engine([
            detection("Bill To:", 50, 100, 120, 120),
            detection("Acme Industries Pvt Ltd", 50, 128, 280, 148),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["Bill To:", "Acme Industries Pvt Ltd"]

    def test_three_fragments_merge_into_one_run(self, stub_engine, blank_page):
        stub_engine([
            detection("Total", 400, 500, 450, 520),
            detection("Amount", 460, 500, 530, 520),
            detection("Payable", 540, 500, 615, 520),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["Total Amount Payable"]

    def test_a_line_can_split_into_several_runs(self, stub_engine, blank_page):
        # A metadata grid: two independent label/value pairs on one visual row.
        stub_engine([
            detection("Invoice Date:", 50, 100, 160, 120),
            detection("14/08/2026", 172, 100, 260, 120),
            detection("Due Date:", 600, 100, 680, 120),
            detection("13/09/2026", 692, 100, 780, 120),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == [
            "Invoice Date: 14/08/2026",
            "Due Date: 13/09/2026",
        ]

    def test_merged_line_box_spans_all_its_fragments(self, stub_engine, blank_page):
        stub_engine([
            detection("Invoice No:", 50, 100, 160, 122),
            detection("INV-42", 175, 98, 320, 120),
        ])
        line = ocr_service.recognize_page(blank_page).lines[0]
        assert (line.box.x0, line.box.y0, line.box.x1, line.box.y1) == (50, 98, 320, 122)


class TestConfidence:
    def test_merged_line_takes_the_worst_fragment_confidence(self, stub_engine, blank_page):
        # Averaging would hide exactly the case a reviewer needs to see: a
        # crisp label next to an amount the engine barely read.
        stub_engine([
            detection("Grand Total", 400, 500, 500, 520, confidence=0.99),
            detection("1,92,407.04", 515, 500, 620, 520, confidence=0.42),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert page.lines[0].confidence == pytest.approx(0.42)

    def test_noise_below_the_threshold_is_dropped(self, stub_engine, blank_page):
        stub_engine([
            detection("Tax Invoice", 300, 50, 450, 75, confidence=0.97),
            detection("~~smudge~~", 700, 900, 780, 915, confidence=0.11),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["Tax Invoice"]

    def test_threshold_is_overridable_per_call(self, stub_engine, blank_page):
        stub_engine([detection("faint", 10, 10, 60, 25, confidence=0.35)])
        assert ocr_service.recognize_page(blank_page, min_confidence=0.5).lines == []
        assert len(ocr_service.recognize_page(blank_page, min_confidence=0.2).lines) == 1

    def test_page_confidence_summarises_its_lines(self, stub_engine, blank_page):
        stub_engine([
            detection("Alpha", 10, 10, 60, 30, confidence=0.90),
            detection("Beta", 10, 60, 60, 80, confidence=0.70),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert page.mean_confidence == pytest.approx(0.80)
        assert [line.text for line in page.low_confidence_lines] == ["Beta"]


class TestWordGeometry:
    def test_words_are_addressable_within_a_merged_line(self, stub_engine, blank_page):
        stub_engine([
            detection("Invoice No:", 50, 100, 160, 120),
            detection("INV-2026-0042", 175, 100, 320, 120),
        ])
        words = ocr_service.recognize_page(blank_page).lines[0].words
        assert [word.text for word in words] == ["Invoice", "No:", "INV-2026-0042"]

    def test_word_boxes_stay_within_their_own_fragment(self, stub_engine, blank_page):
        # Apportioning across the *merged* extent would smear the label's
        # words across the gap and put them on top of the value.
        stub_engine([
            detection("Invoice No:", 50, 100, 160, 120),
            detection("INV-2026-0042", 600, 100, 760, 120),
        ])
        page = ocr_service.recognize_page(blank_page)
        label_words = page.lines[0].words
        assert all(word.box.x1 <= 160 for word in label_words)

    def test_a_single_token_line_keeps_the_exact_detected_box(self, stub_engine, blank_page):
        stub_engine([detection("1,92,407.04", 650, 300, 760, 320)])
        word = ocr_service.recognize_page(blank_page).lines[0].words[0]
        assert word.approximate_box is False
        assert (word.box.x0, word.box.x1) == (650, 760)

    def test_multi_token_boxes_are_flagged_as_approximate(self, stub_engine, blank_page):
        stub_engine([detection("Steel Fabrication Work", 50, 300, 300, 320)])
        words = ocr_service.recognize_page(blank_page).lines[0].words
        assert [word.text for word in words] == ["Steel", "Fabrication", "Work"]
        assert all(word.approximate_box for word in words)
        # Left-to-right order is preserved even though widths are estimated.
        assert words[0].box.x0 < words[1].box.x0 < words[2].box.x0


class TestPageAssembly:
    def test_page_records_the_image_dimensions_the_boxes_refer_to(self, stub_engine, blank_page):
        stub_engine([detection("Tax Invoice", 300, 50, 450, 75)])
        page = ocr_service.recognize_page(blank_page, page_number=3)
        assert (page.width, page.height) == (800.0, 1000.0)
        assert page.page_number == 3
        assert page.lines[0].page_number == 3

    def test_lines_are_indexed_in_reading_order(self, stub_engine, blank_page):
        stub_engine([
            detection("third", 10, 300, 100, 320),
            detection("first", 10, 100, 100, 120),
            detection("second", 10, 200, 100, 220),
        ])
        page = ocr_service.recognize_page(blank_page)
        assert [line.text for line in page.lines] == ["first", "second", "third"]
        assert [line.line_index for line in page.lines] == [0, 1, 2]

    def test_page_text_is_the_flat_reading_order_string(self, stub_engine, blank_page):
        stub_engine([
            detection("Tax Invoice", 300, 50, 450, 75),
            detection("Invoice No:", 50, 150, 160, 170),
            detection("INV-42", 175, 150, 260, 170),
        ])
        assert ocr_service.recognize_page(blank_page).text == "Tax Invoice\nInvoice No: INV-42"

    def test_preprocessing_history_is_carried_on_the_page(self, stub_engine, blank_page):
        stub_engine([detection("x", 10, 10, 20, 20)])
        page = ocr_service.recognize_page(
            blank_page, preprocessing_applied=["deskew", "clahe"], rotation_applied=-1.4
        )
        assert page.preprocessing_applied == ["deskew", "clahe"]
        assert page.rotation_applied == -1.4

    def test_a_page_with_no_detections_is_empty_not_an_error(self, stub_engine, blank_page):
        stub_engine([])
        page = ocr_service.recognize_page(blank_page)
        assert page.lines == []
        assert page.text == ""
        assert page.mean_confidence == 0.0


class TestMultiPage:
    def test_pages_are_numbered_sequentially_and_progress_is_reported(self, stub_engine, blank_page):
        stub_engine([detection("page text", 10, 10, 100, 30)])
        progress: list[tuple[int, int]] = []
        document = ocr_service.recognize_pages(
            [blank_page, blank_page, blank_page], on_page=lambda done, total: progress.append((done, total))
        )
        assert [page.page_number for page in document.pages] == [1, 2, 3]
        assert progress == [(1, 3), (2, 3), (3, 3)]

    def test_one_unreadable_page_does_not_cost_the_others(self, monkeypatch, blank_page):
        calls = {"count": 0}

        def flaky(image, language="en"):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OcrPageFailed("corrupt raster")
            return [detection("readable", 10, 10, 100, 30)]

        monkeypatch.setattr(ocr_service.engine, "recognize", flaky)
        document = ocr_service.recognize_pages([blank_page, blank_page, blank_page])
        assert [len(page.lines) for page in document.pages] == [1, 0, 1]
        assert document.page_count == 3

    def test_an_unavailable_engine_aborts_rather_than_returning_empty_pages(self, monkeypatch, blank_page):
        # Operational failure: every page would fail identically, and silently
        # returning three empty pages would look like three blank invoices.
        def unavailable(image, language="en"):
            raise OcrEngineUnavailable("paddle not installed")

        monkeypatch.setattr(ocr_service.engine, "recognize", unavailable)
        with pytest.raises(OcrEngineUnavailable):
            ocr_service.recognize_pages([blank_page, blank_page])

    def test_document_confidence_spans_every_page(self, monkeypatch, blank_page):
        scores = iter([0.9, 0.5])

        def per_page(image, language="en"):
            return [detection("text", 10, 10, 100, 30, confidence=next(scores))]

        monkeypatch.setattr(ocr_service.engine, "recognize", per_page)
        document = ocr_service.recognize_pages([blank_page, blank_page])
        assert document.mean_confidence == pytest.approx(0.70)
        assert document.used_ocr is True
        assert document.text == "text\ntext"


class TestFingerprint:
    def test_identical_images_hash_identically_and_different_ones_do_not(self):
        first = np.zeros((10, 10, 3), dtype=np.uint8)
        second = np.zeros((10, 10, 3), dtype=np.uint8)
        third = np.ones((10, 10, 3), dtype=np.uint8)
        assert ocr_service.image_fingerprint(first) == ocr_service.image_fingerprint(second)
        assert ocr_service.image_fingerprint(first) != ocr_service.image_fingerprint(third)


class TestImageDecoding:
    def test_undecodable_bytes_fail_as_a_page_error_not_a_crash(self):
        with pytest.raises(OcrPageFailed):
            ocr_service.recognize_image_bytes(b"this is not an image")
