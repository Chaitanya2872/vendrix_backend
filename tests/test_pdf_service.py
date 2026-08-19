"""PDF routing: text layer versus rasterise, decided per page.

The expensive mistake this module exists to prevent is OCR-ing a page that
already has perfectly good text — minutes of compute for a worse answer than
was available for free. The other expensive mistake is trusting a text layer
that is really a stamp's embedded label on a scan.
"""
import fitz
import numpy as np
import pytest

from app.modules.ocr import pdf_service
from app.modules.ocr.dto import SOURCE_NATIVE
from app.modules.ocr.exceptions import OcrPageFailed


def text_pdf(tmp_path, name="text.pdf", pages=1, body=None):
    """A generated PDF with a real text layer."""
    body = body or (
        "TAX INVOICE\n"
        "Invoice No: INV-2026-0042        Invoice Date: 14/08/2026\n"
        "Bill To: Acme Industries Private Limited\n"
        "GSTIN: 29ABCDE1234F1Z5\n"
        "Description            HSN      Qty     Rate       Amount\n"
        "Steel Fabrication      7308      10   1500.00    15000.00\n"
        "Sub Total                                        15000.00\n"
        "CGST 9%                                           1350.00\n"
        "SGST 9%                                           1350.00\n"
        "Grand Total                                      17700.00\n"
    )
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((60, 80), body, fontsize=10)
    path = tmp_path / name
    document.save(str(path))
    document.close()
    return path


def scanned_pdf(tmp_path, name="scan.pdf", pages=1, embedded_text=None):
    """A PDF whose pages are images — no usable text layer."""
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (1240, 1754), (245, 245, 245)).save(image_path)

    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 595, 842), filename=str(image_path))
        if embedded_text:
            page.insert_text((40, 820), embedded_text, fontsize=6)
    path = tmp_path / name
    document.save(str(path))
    document.close()
    return path


class TestRouting:
    def test_a_text_pdf_is_read_natively_and_never_sent_to_ocr(self, tmp_path):
        analysis = pdf_service.analyse(text_pdf(tmp_path))

        assert analysis.ocr_page_count == 0
        assert analysis.native_page_count == 1
        assert analysis.pages[0].needs_ocr is False
        assert analysis.pages[0].native.source == SOURCE_NATIVE

    def test_a_scanned_pdf_is_rasterised_for_ocr(self, tmp_path):
        analysis = pdf_service.analyse(scanned_pdf(tmp_path))

        assert analysis.native_page_count == 0
        page = analysis.pages[0]
        assert page.needs_ocr is True
        assert isinstance(page.image, np.ndarray)
        assert page.image.ndim == 3 and page.image.shape[2] == 3, "OCR expects a 3-channel image"

    def test_a_mixed_pdf_routes_each_page_independently(self, tmp_path):
        """The case a per-document decision gets wrong either way: a
        generated invoice with a scanned annexure stapled to it."""
        from PIL import Image

        image_path = tmp_path / "page.png"
        Image.new("RGB", (1240, 1754), (245, 245, 245)).save(image_path)

        document = fitz.open()
        first = document.new_page()
        # Short lines: PyMuPDF clips a line that runs past the page edge, so
        # one long string would silently yield far less text than it looks.
        first.insert_text((60, 80), "TAX INVOICE\n" + "\n".join(["Line item detail here."] * 20), fontsize=10)
        second = document.new_page(width=595, height=842)
        second.insert_image(fitz.Rect(0, 0, 595, 842), filename=str(image_path))
        path = tmp_path / "mixed.pdf"
        document.save(str(path))
        document.close()

        analysis = pdf_service.analyse(path)

        assert analysis.is_mixed
        assert analysis.pages[0].needs_ocr is False
        assert analysis.pages[1].needs_ocr is True

    def test_a_scan_carrying_a_stray_embedded_string_is_still_ocrd(self, tmp_path):
        # A character count alone would be fooled by a long footer; the area
        # ratio is what catches it.
        path = scanned_pdf(tmp_path, embedded_text="Scanned by DeptScanner 9000 " * 12)

        analysis = pdf_service.analyse(path)

        decision = analysis.pages[0].decision
        assert analysis.pages[0].needs_ocr is True
        assert decision.character_count >= 120, "the stray text cleared the character threshold"
        assert decision.image_coverage >= 0.60, "the page is a full-page image"
        assert "covered by an image" in decision.reason

    def test_a_sparse_but_genuine_text_page_is_not_sent_to_ocr(self, tmp_path):
        """A terms page or a signature page is legitimately sparse. Judging
        scans by text density alone would send these for two minutes of OCR
        to recover text that was already there for free."""
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (60, 80),
            "TERMS AND CONDITIONS\n"
            + "\n".join(["Payment is due within thirty days of invoice date."] * 4)
            + "\n\nFor Alpha Steel Works\n\nAuthorised Signatory",
            fontsize=10,
        )
        path = tmp_path / "terms.pdf"
        document.save(str(path))
        document.close()

        decision = pdf_service.analyse(path).pages[0].decision

        assert decision.route == "native"
        assert decision.image_coverage == 0.0

    def test_a_sparse_imageless_page_is_read_natively_not_ocrd(self, tmp_path):
        """OCR can only recover what is in pixels. A page with a text layer
        and no raster content holds nothing OCR could find that the text
        layer does not already state exactly — so sending it costs two
        minutes a page and returns a worse answer."""
        document = fitz.open()
        document.new_page().insert_text((60, 80), "Invoice No: ASW/2026/0042", fontsize=10)
        path = tmp_path / "sparse.pdf"
        document.save(str(path))
        document.close()

        decision = pdf_service.analyse(path).pages[0].decision

        assert decision.route == "native"
        assert decision.character_count < pdf_service.MIN_NATIVE_CHARACTERS_PER_PAGE
        assert "no images" in decision.reason

    def test_a_sparse_page_that_does_carry_an_image_is_still_ocrd(self, tmp_path):
        # Here the pixels may hold the whole invoice, so OCR earns its cost.
        path = scanned_pdf(tmp_path, embedded_text="Invoice 42")

        assert pdf_service.analyse(path).pages[0].needs_ocr is True

    def test_ocr_can_be_forced_over_a_usable_text_layer(self, tmp_path):
        analysis = pdf_service.analyse(text_pdf(tmp_path), force_ocr=True)

        assert analysis.pages[0].needs_ocr is True
        assert analysis.pages[0].decision.reason == "OCR forced by caller"

    def test_every_page_records_why_it_was_routed_that_way(self, tmp_path):
        decisions = pdf_service.analyse(text_pdf(tmp_path, pages=3)).decisions()

        assert len(decisions) == 3
        assert all(entry["reason"] for entry in decisions)
        assert [entry["page_number"] for entry in decisions] == [1, 2, 3]


class TestNativeExtraction:
    def test_words_carry_exact_boxes_not_estimates(self, tmp_path):
        page = pdf_service.analyse(text_pdf(tmp_path)).pages[0].native

        words = page.words()
        assert words
        assert all(word.approximate_box is False for word in words)
        assert all(word.confidence == 1.0 for word in words), "typed characters are not a recognition guess"

    def test_coordinates_are_in_raster_pixels_not_points(self, tmp_path):
        """Both routes must produce boxes in the same units, or every
        geometric rule downstream is wrong on half the pages."""
        analysis = pdf_service.analyse(text_pdf(tmp_path), render_dpi=300)
        page = analysis.pages[0].native

        # A4 at 300 DPI is ~2480x3508 px; at 72 points it would be 595x842.
        assert 2400 < page.width < 2600
        assert 3400 < page.height < 3600
        assert all(line.box.x1 <= page.width + 1 for line in page.lines)

    def test_render_dpi_scales_the_coordinates_proportionally(self, tmp_path):
        path = text_pdf(tmp_path)
        low = pdf_service.analyse(path, render_dpi=150).pages[0].native
        high = pdf_service.analyse(path, render_dpi=300).pages[0].native

        assert high.width == pytest.approx(low.width * 2, rel=0.01)
        assert high.lines[0].box.x0 == pytest.approx(low.lines[0].box.x0 * 2, rel=0.01)

    def test_lines_come_out_in_reading_order(self, tmp_path):
        page = pdf_service.analyse(text_pdf(tmp_path)).pages[0].native

        tops = [line.box.y0 for line in page.lines]
        assert tops == sorted(tops)
        assert [line.line_index for line in page.lines] == list(range(len(page.lines)))

    def test_the_text_layer_content_is_recovered(self, tmp_path):
        page = pdf_service.analyse(text_pdf(tmp_path)).pages[0].native

        text = page.text
        assert "TAX INVOICE" in text
        assert "INV-2026-0042" in text
        assert "29ABCDE1234F1Z5" in text

    def test_side_by_side_columns_stay_separable(self, tmp_path):
        """The defect this module was written to fix: a line-level text
        extractor collapses a two-column seller/buyer block into one string,
        and a line classifier can then only assign it to one of them."""
        document = fitz.open()
        page = document.new_page()
        page.insert_text((60, 100), "Seller: Alpha Steel Works", fontsize=10)
        page.insert_text((340, 100), "Buyer: Beta Constructions", fontsize=10)
        page.insert_text((60, 120), "GSTIN: 29AAAAA0000A1Z5", fontsize=10)
        page.insert_text((340, 120), "GSTIN: 27BBBBB1111B2Z6", fontsize=10)
        page.insert_text((60, 200), "Filler content to clear the text threshold. " * 12, fontsize=9)
        path = tmp_path / "columns.pdf"
        document.save(str(path))
        document.close()

        result = pdf_service.analyse(path).pages[0].native

        seller_words = [word for word in result.words() if "29AAAAA0000A1Z5" in word.text]
        buyer_words = [word for word in result.words() if "27BBBBB1111B2Z6" in word.text]
        assert seller_words and buyer_words
        # They are distinguishable by position, which is what the layout
        # stage needs; a collapsed string offers no such handle.
        assert seller_words[0].box.x0 < buyer_words[0].box.x0

    def test_page_numbers_are_one_based_throughout(self, tmp_path):
        analysis = pdf_service.analyse(text_pdf(tmp_path, pages=3))

        assert [page.page_number for page in analysis.pages] == [1, 2, 3]
        assert [page.native.page_number for page in analysis.pages] == [1, 2, 3]
        assert all(line.page_number == page.native.page_number
                   for page in analysis.pages for line in page.native.lines)


class TestRendering:
    def test_rendering_produces_the_expected_pixel_size_for_the_dpi(self, tmp_path):
        document = fitz.open(str(scanned_pdf(tmp_path)))
        try:
            image = pdf_service.render_page(document[0], dpi=150)
        finally:
            document.close()

        # A4 at 150 DPI ≈ 1240x1754.
        assert 1200 < image.shape[1] < 1290
        assert 1700 < image.shape[0] < 1800

    def test_render_all_produces_an_image_for_every_page_including_text_ones(self, tmp_path):
        # The review UI overlays boxes on a rendered page, so it needs an
        # image even where the text was read natively.
        images = pdf_service.render_all(text_pdf(tmp_path, pages=3), dpi=100)

        assert len(images) == 3
        assert all(isinstance(image, np.ndarray) for image in images)


class TestFailures:
    def test_a_password_protected_pdf_reports_that_specifically(self, tmp_path):
        document = fitz.open()
        document.new_page()
        path = tmp_path / "locked.pdf"
        document.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
        document.close()

        with pytest.raises(OcrPageFailed, match="password"):
            pdf_service.analyse(path)

    def test_a_corrupt_pdf_fails_cleanly(self, tmp_path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nnot really a pdf")

        with pytest.raises(OcrPageFailed):
            pdf_service.analyse(path)

    def test_a_missing_file_fails_cleanly(self, tmp_path):
        with pytest.raises(OcrPageFailed):
            pdf_service.analyse(tmp_path / "nope.pdf")

    def test_page_count_does_not_read_the_whole_document(self, tmp_path):
        assert pdf_service.page_count(text_pdf(tmp_path, pages=4)) == 4
