"""Format routing and OCR fallback in the invoice text-extraction layer."""
import pytest

from app.modules.invoices.parsers import text_extraction
from app.modules.invoices.parsers.text_extraction import (
    UnsupportedDocumentError,
    extract_document,
)

# Real OCR (PaddleOCR) runs in a few of these. It is the actual thing under
# test for the image paths, so it isn't stubbed there — but it costs seconds,
# so the routing-only tests use the stub fixture below.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def stub_ocr(monkeypatch):
    """Replaces the OCR engine with a canned reader, for tests that care
    which path was taken rather than what the engine can read."""
    calls = []

    def fake(raw: bytes) -> str:
        calls.append(len(raw))
        return "TAX INVOICE\nInvoice No: INV-2024-0042\nGrand Total: 11800.00"

    monkeypatch.setattr(text_extraction, "_ocr_image_bytes", fake)
    return calls


def test_text_pdf_uses_native_extraction(sample):
    """A PDF with a text layer must not pay for OCR."""
    result = extract_document(str(sample("text_pdf")))

    assert result.used_ocr is False
    assert result.page_count == 1
    assert "INV-2024-0042" in result.text


def test_scanned_pdf_falls_back_to_ocr(sample):
    result = extract_document(str(sample("scanned_pdf")))

    assert result.used_ocr is True
    assert "INV-2024-0042" in result.text.replace(" ", "")


def test_jpg_is_extracted_via_ocr(sample):
    """The format that failed outright before: pdfplumber was handed a JPG."""
    result = extract_document(str(sample("jpg")))

    assert result.used_ocr is True
    assert result.page_count == 1
    assert "INV-2024-0042" in result.text.replace(" ", "")


@pytest.mark.parametrize("kind", ["jpg", "png", "webp"])
def test_all_image_formats_route_to_ocr(kind, sample, stub_ocr):
    result = extract_document(str(sample(kind)))

    assert result.used_ocr is True
    assert result.page_count == 1
    assert len(stub_ocr) == 1, "expected exactly one OCR call per single-page image"


def test_image_extraction_reports_the_ocr_stage(sample, stub_ocr):
    """The UI's stage animation is driven by these callbacks."""
    stages = []
    extract_document(str(sample("png")), on_stage=stages.append)

    assert stages == ["extracting_text"]


def test_native_pdf_reports_no_ocr_stage(sample):
    """A text PDF never enters OCR, so it must not claim to."""
    stages = []
    extract_document(str(sample("text_pdf")), on_stage=stages.append)

    assert stages == []


def test_scanned_pdf_reports_the_ocr_stage(sample, stub_ocr):
    stages = []
    extract_document(str(sample("scanned_pdf")), on_stage=stages.append)

    assert stages == ["extracting_text"]


def test_unsupported_extension_names_what_is_supported(sample):
    with pytest.raises(UnsupportedDocumentError) as raised:
        extract_document(str(sample("unsupported")))

    message = str(raised.value)
    assert ".txt" in message
    # The message has to list what *is* accepted, or the user is told only
    # that their file failed.
    assert ".pdf" in message and ".jpg" in message and ".docx" in message


def test_docx_is_read_natively(sample):
    """Word invoices are an accepted upload type. They used to reach the
    parser and be rejected outright, so a .docx invoice produced no fields
    at all."""
    result = extract_document(str(sample("docx")))

    assert result.used_ocr is False
    assert "INV-2024-0042" in result.text
    assert "29ABCDE1234F1Z5" in result.text


def test_docx_tables_reach_the_line_item_parser(sample):
    """The table is the reason DOCX is worth reading natively — OCR of the
    same invoice cannot recover column structure at all."""
    result = extract_document(str(sample("docx")))

    tables = [table for page in result.tables_per_page for table in page]
    assert tables, "no table recovered from the DOCX"
    assert any("Steel fasteners M12" in cell for row in tables[0] for cell in row)


def test_xlsx_is_read_natively(sample):
    """Spreadsheet invoices put label and value in adjacent cells; the
    extractor has to flatten them into a line the field regexes can read."""
    result = extract_document(str(sample("xlsx")))

    assert result.used_ocr is False
    assert "INV-2024-0042" in result.text
    assert any("Grand Total" in line and "11800.00" in line for line in result.text.splitlines())


def test_office_formats_need_no_ocr(sample, stub_ocr):
    """Paying for OCR on a format that carries exact text would be pure
    waste — and slow enough to notice."""
    extract_document(str(sample("docx")))
    extract_document(str(sample("xlsx")))

    assert stub_ocr == [], "OCR was invoked for an office document"


def test_corrupt_pdf_is_reported_not_crashed(sample):
    with pytest.raises(UnsupportedDocumentError):
        extract_document(str(sample("corrupt_pdf")))


def test_missing_file_raises_unsupported(tmp_path):
    with pytest.raises(UnsupportedDocumentError):
        extract_document(str(tmp_path / "does-not-exist.png"))


def test_ocr_failure_surfaces_as_unsupported(sample, monkeypatch):
    """With no usable OCR engine the caller gets a typed error it already
    handles, not an ImportError escaping from three layers down."""
    monkeypatch.setattr(
        text_extraction, "_ocr_image_bytes",
        lambda raw: (_ for _ in ()).throw(UnsupportedDocumentError("No usable OCR engine")),
    )

    with pytest.raises(UnsupportedDocumentError):
        extract_document(str(sample("jpg")))
