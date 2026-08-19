"""Decide between native PDF text extraction and OCR, and extract tables.

For PDFs, native extraction (pdfplumber) is always tried first and is cheap.
OCR is only invoked when the native text yield is below a configurable
threshold — most invoices are text-based PDFs and OCR-ing every one of them
would be slow and unnecessary.

Photographed/scanned invoices (JPG, JPEG, PNG, WEBP) have no native text
layer at all, so they skip straight to OCR. The OCR engine is the same one
the rest of the project uses (PaddleOCR, via app.workers.vision) rather than
a second stack: pytesseract is kept only as a fallback for environments where
paddle isn't importable, since it additionally needs a system tesseract
binary that isn't part of this project's dependency set.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Configurable threshold: total characters of native text below which we
# treat the PDF as "likely scanned" and fall back to OCR. Exposed as a
# module-level constant so app.core.config can override it if desired.
MIN_NATIVE_TEXT_CHARACTERS = 200

# Raster formats accepted by the upload endpoint. These carry no text layer,
# so they are OCR-only.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Office formats the upload endpoint accepts. They carry exact characters and
# real table structure, so they need neither native-vs-OCR arbitration nor a
# fallback — but they do need handling, since an invoice uploaded as .docx
# would otherwise be routed to the invoice parser and immediately rejected as
# an unsupported document.
OFFICE_EXTENSIONS = {".docx", ".xlsx"}

# Cell separator used when flattening a spreadsheet row into a text line. Two
# spaces, not a pipe: the header-level field regexes look for "Label value"
# with ordinary whitespace between, and a pipe would sit between every label
# and its own value.
_CELL_SEPARATOR = "  "


class UnsupportedDocumentError(Exception):
    """Raised for corrupt / password-protected / unreadable documents."""


@dataclass
class ExtractedDocument:
    text: str
    tables_per_page: list[list[list[list[str | None]]]] = field(default_factory=list)
    used_ocr: bool = False
    page_count: int = 0


def _extract_native(file_path: str) -> ExtractedDocument:
    import pdfplumber

    try:
        with pdfplumber.open(file_path) as pdf:
            texts = []
            tables_per_page = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                texts.append(page_text)
                try:
                    tables_per_page.append(page.extract_tables())
                except Exception:  # pdfplumber table extraction can fail on odd layouts
                    tables_per_page.append([])
            return ExtractedDocument(
                text="\n".join(texts),
                tables_per_page=tables_per_page,
                used_ocr=False,
                page_count=len(pdf.pages),
            )
    except Exception as exc:  # pdfplumber raises varied exceptions for bad/encrypted PDFs
        message = str(exc).lower()
        if "password" in message or "encrypt" in message:
            raise UnsupportedDocumentError("Document is password-protected.") from exc
        raise UnsupportedDocumentError(f"Unable to read PDF: {exc}") from exc


def _ocr_image_bytes(raw: bytes) -> str:
    """OCR a single raster image, preferring the project's PaddleOCR helper
    (app.workers.vision, already used by the vendor-document worker) and
    falling back to pytesseract where paddle isn't available. Imported lazily
    because loading paddle is expensive and most invoices are text PDFs that
    never reach this path."""
    try:
        from app.workers.vision import decode_image, read_text
        return read_text(decode_image(raw))
    except ImportError:
        pass
    except Exception as exc:
        # Paddle is present but failed on this image; try the fallback before
        # giving up on the page entirely.
        logger.warning("invoice_parsing.paddle_ocr_failed error=%s", exc)

    try:
        import io
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(raw)))
    except Exception as exc:
        raise UnsupportedDocumentError(
            f"No usable OCR engine: PaddleOCR unavailable and pytesseract failed ({exc})."
        ) from exc


def _extract_via_ocr(file_path: str) -> ExtractedDocument:
    """Render PDF pages with PyMuPDF and OCR them. Only used when native
    extraction yields too little text (i.e. a scanned invoice). Table
    structure cannot be reliably recovered from OCR text, so
    `tables_per_page` is left empty and downstream code falls back to
    header-level regex extraction only."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise UnsupportedDocumentError("OCR fallback requires PyMuPDF, which is not installed.") from exc

    try:
        document = fitz.open(file_path)
    except Exception as exc:
        raise UnsupportedDocumentError(f"Unable to open PDF for OCR: {exc}") from exc

    with document:
        texts = [_ocr_image_bytes(page.get_pixmap(dpi=300).tobytes("png")) for page in document]
        page_count = len(document)

    return ExtractedDocument(
        text="\n".join(texts),
        tables_per_page=[[] for _ in range(page_count)],
        used_ocr=True,
        page_count=page_count,
    )


def _extract_image(file_path: str) -> ExtractedDocument:
    """OCR a photographed or scanned invoice. There is no text layer and no
    recoverable table structure, so the downstream parser works from the
    header-level regex extraction only."""
    try:
        raw = Path(file_path).read_bytes()
    except OSError as exc:
        raise UnsupportedDocumentError(f"Unable to read image: {exc}") from exc

    text = _ocr_image_bytes(raw)
    logger.info("invoice_parsing.image_ocr_completed characters=%s", len(text.strip()))
    return ExtractedDocument(text=text, tables_per_page=[[]], used_ocr=True, page_count=1)


def _extract_docx(file_path: str) -> ExtractedDocument:
    """Read a Word invoice: paragraphs become text lines, and each table
    becomes a table the line-item parser can consume directly — DOCX keeps
    the cell structure that OCR of the same invoice would have destroyed."""
    try:
        from docx import Document as WordDocument
    except ImportError as exc:
        raise UnsupportedDocumentError("Reading .docx requires python-docx, which is not installed.") from exc

    try:
        document = WordDocument(file_path)
    except Exception as exc:
        raise UnsupportedDocumentError(f"Unable to read Word document: {exc}") from exc

    lines = [paragraph.text for paragraph in document.paragraphs]
    tables: list[list[list[str | None]]] = []
    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            tables.append(rows)
            # Also surface the table as text: header-level values (a total, a
            # GSTIN) are often inside a table rather than a paragraph, and the
            # regex extractors only ever see `text`.
            lines.extend(_CELL_SEPARATOR.join(filter(None, row)) for row in rows)

    return ExtractedDocument(
        text="\n".join(line for line in lines if line.strip()),
        tables_per_page=[tables],
        used_ocr=False,
        page_count=1,
    )


def _extract_xlsx(file_path: str) -> ExtractedDocument:
    """Read a spreadsheet invoice. Every sheet is treated as one table; rows
    are also flattened into text lines so the same header-level extractors
    that work on PDFs apply unchanged."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise UnsupportedDocumentError("Reading .xlsx requires openpyxl, which is not installed.") from exc

    try:
        workbook = load_workbook(file_path, read_only=True, data_only=True)
    except Exception as exc:
        raise UnsupportedDocumentError(f"Unable to read spreadsheet: {exc}") from exc

    lines: list[str] = []
    tables: list[list[list[str | None]]] = []
    try:
        for sheet in workbook.worksheets:
            rows = [
                ["" if value is None else str(value).strip() for value in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            if not rows:
                continue
            tables.append(rows)
            lines.extend(_CELL_SEPARATOR.join(cell for cell in row if cell) for row in rows)
    finally:
        workbook.close()

    return ExtractedDocument(
        text="\n".join(line for line in lines if line.strip()),
        tables_per_page=[tables],
        used_ocr=False,
        page_count=1,
    )


def extract_document(
    file_path: str,
    min_native_text_characters: int = MIN_NATIVE_TEXT_CHARACTERS,
    on_stage: Callable[[str], None] | None = None,
) -> ExtractedDocument:
    """Extract text (and, where possible, tables) from an invoice file.

    PDFs choose between native extraction and OCR based on how much usable
    text the native pass produced; raster images are OCR-only.

    `on_stage` is called with a stage name when the work changes character.
    OCR dominates the runtime when it happens, so a caller reporting progress
    would otherwise show one opaque step for the entire wait.
    """
    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    extension = Path(file_path).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        logger.info("invoice_parsing.image_ocr_selected extension=%s", extension)
        stage("extracting_text")
        return _extract_image(file_path)

    if extension in OFFICE_EXTENSIONS:
        logger.info("invoice_parsing.office_text_selected extension=%s", extension)
        stage("extracting_text")
        return _extract_docx(file_path) if extension == ".docx" else _extract_xlsx(file_path)

    if extension != ".pdf":
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | OFFICE_EXTENSIONS | {".pdf"}))
        raise UnsupportedDocumentError(
            f"Invoice parsing supports {supported}, not '{extension}'."
        )

    native = _extract_native(file_path)
    total_characters = len(native.text.strip())

    if total_characters >= min_native_text_characters:
        logger.info("invoice_parsing.native_text_selected characters=%s", total_characters)
        return native

    logger.info("invoice_parsing.ocr_fallback_selected native_characters=%s", total_characters)
    stage("extracting_text")
    try:
        ocr_result = _extract_via_ocr(file_path)
    except UnsupportedDocumentError:
        logger.warning("invoice_parsing.ocr_fallback_failed; continuing with sparse native text")
        return native  # degrade gracefully rather than fail the whole upload
    return ocr_result
