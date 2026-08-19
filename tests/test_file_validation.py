"""Upload validation.

The theme throughout: the client's claims are checked against the bytes, and
the bytes win. Each test names the attack or accident it prevents, because
"validate the file" is otherwise a checklist nobody can audit.
"""
import io
import zipfile

import pytest
from PIL import Image

from app.utils import file_utils
from app.utils.file_utils import FileValidationError, detect_format, validate_upload

ONE_MB = 1024 * 1024


def png_bytes(width=800, height=1000, colour=(255, 255, 255)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width=800, height=1000):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, format="JPEG")
    return buffer.getvalue()


def tiff_bytes(pages=1, width=800, height=1000):
    buffer = io.BytesIO()
    images = [Image.new("RGB", (width, height), (255, 255, 255)) for _ in range(pages)]
    images[0].save(buffer, format="TIFF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def pdf_bytes(pages=1):
    import fitz

    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), "Tax Invoice")
    payload = document.tobytes()
    document.close()
    return payload


def ooxml_bytes(marker):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(f"{marker}/document.xml", "<x/>")
    return buffer.getvalue()


class TestFormatDetection:
    @pytest.mark.parametrize(
        ("payload_factory", "expected"),
        [
            (pdf_bytes, file_utils.PDF),
            (jpeg_bytes, file_utils.JPEG),
            (png_bytes, file_utils.PNG),
            (tiff_bytes, file_utils.TIFF),
        ],
    )
    def test_identifies_each_supported_format_from_its_bytes(self, payload_factory, expected):
        assert detect_format(payload_factory()) == expected

    def test_distinguishes_docx_from_xlsx_inside_the_zip_container(self):
        # Both are PK zips; only the part names tell them apart.
        assert detect_format(ooxml_bytes("word")) == file_utils.DOCX
        assert detect_format(ooxml_bytes("xl")) == file_utils.XLSX

    def test_big_endian_tiff_is_recognised(self):
        # Scanners emit both byte orders; only accepting 'II' silently
        # rejects half of them as unrecognised content.
        assert detect_format(b"MM\x00\x2a" + b"\x00" * 32) == file_utils.TIFF

    def test_unknown_content_is_none_rather_than_a_guess(self):
        assert detect_format(b"#!/bin/sh\necho hello\n" + b"x" * 40) is None

    def test_a_file_too_short_to_identify_is_none(self):
        assert detect_format(b"%PDF") is None


class TestExtensionAndContentAgreement:
    def test_a_jpeg_renamed_to_pdf_is_rejected_with_a_specific_reason(self):
        # The case that matters: the extension check alone passes, and the
        # PDF reader downstream would fail with something incomprehensible.
        with pytest.raises(FileValidationError) as caught:
            validate_upload(jpeg_bytes(), "invoice.pdf", "application/pdf", 10 * ONE_MB)
        assert caught.value.code == "CONTENT_EXTENSION_MISMATCH"
        assert "JPEG" in caught.value.message

    def test_an_executable_renamed_to_png_is_rejected(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(b"MZ\x90\x00" + b"\x00" * 100, "invoice.png", "image/png", 10 * ONE_MB)
        assert caught.value.code == "UNRECOGNISED_CONTENT"

    def test_jpg_and_jpeg_extensions_are_both_accepted(self):
        for name in ("invoice.jpg", "invoice.jpeg"):
            assert validate_upload(jpeg_bytes(), name, "image/jpeg", 10 * ONE_MB).file_format == file_utils.JPEG

    def test_tif_and_tiff_extensions_are_both_accepted(self):
        for name in ("scan.tif", "scan.tiff"):
            assert validate_upload(tiff_bytes(), name, "image/tiff", 10 * ONE_MB).file_format == file_utils.TIFF

    def test_uppercase_extensions_are_accepted(self):
        # Scanners and Windows both produce these routinely.
        assert validate_upload(pdf_bytes(), "INVOICE.PDF", None, 10 * ONE_MB).file_format == file_utils.PDF

    def test_a_wrong_content_type_header_is_tolerated_because_the_bytes_decided(self):
        # Browsers and mobile clients get Content-Type wrong constantly;
        # rejecting on it would fail real uploads for no security gain.
        result = validate_upload(png_bytes(), "invoice.png", "application/octet-stream", 10 * ONE_MB)
        assert result.file_format == file_utils.PNG
        assert result.media_type == "image/png"


class TestSizeAndEmptiness:
    def test_an_empty_upload_is_rejected(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(b"", "invoice.pdf", "application/pdf", 10 * ONE_MB)
        assert caught.value.code == "EMPTY_FILE"

    def test_an_oversized_file_is_rejected_before_it_is_parsed(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(pdf_bytes() + b"\x00" * ONE_MB, "invoice.pdf", "application/pdf", 1024)
        assert caught.value.code == "FILE_TOO_LARGE"
        assert "MB" in caught.value.message or "KB" in caught.value.message


class TestIntegrity:
    @pytest.mark.parametrize("keep_bytes", [100, 150, 200, 400])
    def test_a_truncated_pdf_fails_at_upload_not_in_a_worker_an_hour_later(self, keep_bytes):
        """Several truncation points, because MuPDF silently rebuilds a
        damaged cross-reference table: at some lengths a half-uploaded file
        *opens*, reports a plausible page count, and has nothing inside.
        Checking only that `open()` succeeded lets those through."""
        payload = pdf_bytes()
        with pytest.raises(FileValidationError) as caught:
            validate_upload(payload[:keep_bytes], "invoice.pdf", "application/pdf", 10 * ONE_MB)
        assert caught.value.code == "CORRUPT_PDF"

    def test_a_valid_pdf_that_merely_needed_repair_is_still_accepted(self):
        """Repair alone is not corruption — real generators emit files that
        need it, and rejecting those would refuse genuine invoices."""
        import fitz

        document = fitz.open()
        document.new_page().insert_text((72, 72), "Tax Invoice INV-2026-0042")
        payload = document.tobytes()
        document.close()
        # Damage the trailer only: the objects survive, the xref does not.
        damaged = payload.replace(b"startxref", b"startxrEf", 1)

        result = validate_upload(damaged, "invoice.pdf", "application/pdf", 10 * ONE_MB)

        assert result.page_count == 1

    def test_a_truncated_jpeg_is_rejected_rather_than_padded_with_grey(self):
        # Pillow's default is to pad truncated images, which would send half
        # an invoice to OCR and report success.
        payload = jpeg_bytes()
        with pytest.raises(FileValidationError) as caught:
            validate_upload(payload[: len(payload) // 3], "invoice.jpg", "image/jpeg", 10 * ONE_MB)
        assert caught.value.code == "CORRUPT_IMAGE"

    def test_a_thumbnail_sized_image_is_rejected_as_unreadable(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(png_bytes(40, 40), "invoice.png", "image/png", 10 * ONE_MB)
        assert caught.value.code == "IMAGE_TOO_SMALL"

    def test_a_password_protected_pdf_says_so_instead_of_failing_obscurely(self):
        import fitz

        document = fitz.open()
        document.new_page()
        payload = document.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
        document.close()

        with pytest.raises(FileValidationError) as caught:
            validate_upload(payload, "invoice.pdf", "application/pdf", 10 * ONE_MB)
        assert caught.value.code == "PASSWORD_PROTECTED"
        assert "password" in caught.value.message.lower()


class TestDescription:
    def test_reports_the_page_count_of_a_multi_page_pdf(self):
        assert validate_upload(pdf_bytes(pages=5), "invoice.pdf", "application/pdf", 10 * ONE_MB).page_count == 5

    def test_reports_the_frame_count_of_a_multi_page_tiff(self):
        # Departmental scanners emit a stapled invoice as one multi-frame
        # TIFF; reading only the first frame would silently drop pages.
        assert validate_upload(tiff_bytes(pages=3), "scan.tiff", "image/tiff", 10 * ONE_MB).page_count == 3

    def test_checksum_identifies_identical_uploads(self):
        payload = pdf_bytes()
        first = validate_upload(payload, "a.pdf", "application/pdf", 10 * ONE_MB)
        second = validate_upload(payload, "b.pdf", "application/pdf", 10 * ONE_MB)
        assert first.sha256 == second.sha256
        assert first.sha256 != validate_upload(pdf_bytes(pages=2), "c.pdf", None, 10 * ONE_MB).sha256

    def test_raster_formats_are_marked_as_needing_ocr_and_pdfs_are_not(self):
        # A PDF is undecided at this point — its text layer decides.
        assert validate_upload(png_bytes(), "a.png", None, 10 * ONE_MB).needs_ocr is True
        assert validate_upload(pdf_bytes(), "a.pdf", None, 10 * ONE_MB).needs_ocr is False


class TestAllowedFormatNarrowing:
    def test_an_endpoint_can_refuse_a_format_the_project_otherwise_supports(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(
                ooxml_bytes("word"), "invoice.docx", None, 10 * ONE_MB,
                allowed_formats={file_utils.PDF, file_utils.PNG},
            )
        assert caught.value.code == "UNSUPPORTED_FORMAT"
        assert "PDF" in caught.value.message

    def test_the_rejection_message_lists_what_is_accepted(self):
        with pytest.raises(FileValidationError) as caught:
            validate_upload(b"x" * 100, "invoice.txt", None, 10 * ONE_MB, allowed_formats={file_utils.PDF})
        assert "PDF" in caught.value.message


class TestStorageKeys:
    def test_a_traversal_attempt_in_the_filename_cannot_escape_the_root(self):
        key = file_utils.safe_storage_key("invoices", "DOC-2026-000001", "../../../etc/passwd")
        assert ".." not in key
        assert key.startswith("invoices/DOC-2026-000001/")

    def test_a_windows_path_in_the_filename_is_stripped(self):
        key = file_utils.safe_storage_key("invoices", "DOC-2026-000001", r"C:\Users\me\invoice.pdf")
        assert "\\" not in key and ":" not in key.split("/")[-1]

    def test_an_absent_filename_still_yields_a_usable_key(self):
        assert file_utils.safe_storage_key("invoices", "DOC-2026-000001", "").endswith("/upload")

    def test_the_document_number_partitions_storage(self):
        # One directory per document keeps page renders and debug artefacts
        # alongside the original instead of in one flat, unbounded folder.
        key = file_utils.safe_storage_key("invoices", "DOC-2026-000042", "invoice.pdf")
        assert key == "invoices/DOC-2026-000042/invoice.pdf"
