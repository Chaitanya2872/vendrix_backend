"""Generates the invoice test data.

Samples are built at run time rather than committed as binary fixtures, so
the exact content under test is visible in this file and reviewable in a
diff. Every sample derives from the same INVOICE_LINES, which means a value
asserted against a PDF is the same value asserted against its JPG render —
any difference in the result is the pipeline's, not the fixture's.

Kinds:
    text_pdf      PDF with a real text layer (native extraction path)
    scanned_pdf   PDF holding only a page image (forces the OCR fallback)
    jpg/png/webp  photographed-invoice equivalents (OCR-only path)
    blank_png     a valid image with no text on it
    corrupt_pdf   bytes that are not a PDF at all
    docx/xlsx     office uploads, read natively (exact text, real tables)
    unsupported   a file whose extension the parser has no reader for
"""
from __future__ import annotations

from pathlib import Path

# The invoice every sample renders. Kept deliberately plain — one value per
# line — because the point of these tests is the pipeline, not the parser's
# tolerance for exotic layouts.
INVOICE_LINES = [
    "TAX INVOICE",
    "Invoice No: INV-2024-0042",
    "Invoice Date: 15/03/2024",
    "Due Date: 14/04/2024",
    "Vendor: Acme Traders Pvt Ltd",
    "GSTIN: 29ABCDE1234F1Z5",
    "Bill To: IoTIQ Systems Pvt Ltd",
    "Subtotal: 10000.00",
    "CGST 9%: 900.00",
    "SGST 9%: 900.00",
    "Grand Total: 11800.00",
]

# What the parser is expected to recover from the above, regardless of which
# format it arrived in. Tests import this instead of restating literals.
LINE_ITEM_ROWS = (
    ("#", "Description", "HSN/SAC", "Qty", "Unit", "Rate", "Amount"),
    ("1", "Steel fasteners M12", "73181500", "4", "Nos", "1500.00", "6000.00"),
    ("2", "Freight charges", "996511", "1", "Trip", "4000.00", "4000.00"),
)

EXPECTED_FIELDS = {
    "invoice_number": "INV-2024-0042",
    "invoice_date": "2024-03-15",
    "vendor_gstin": "29ABCDE1234F1Z5",
    "total_amount": 11800.0,
}

SAMPLE_KINDS = (
    "text_pdf", "scanned_pdf", "jpg", "png", "webp",
    "blank_png", "corrupt_pdf", "docx", "xlsx", "unsupported",
)

_DEFAULT_NAMES = {
    "text_pdf": "invoice_text.pdf",
    "scanned_pdf": "invoice_scanned.pdf",
    "jpg": "invoice.jpg",
    "png": "invoice.png",
    "webp": "invoice.webp",
    "blank_png": "blank.png",
    "corrupt_pdf": "corrupt.pdf",
    "docx": "invoice.docx",
    "xlsx": "invoice.xlsx",
    "unsupported": "invoice.txt",
}


def _text_pdf_bytes() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    for index, line in enumerate(INVOICE_LINES):
        page.insert_text((60, 80 + index * 30), line, fontsize=13)
    data = document.tobytes()
    document.close()
    return data


def _render_png(dpi: int = 200) -> bytes:
    """Rasterise the text PDF — the resulting image has no text layer, which
    is what makes it a faithful stand-in for a photographed invoice."""
    import fitz

    with fitz.open(stream=_text_pdf_bytes(), filetype="pdf") as document:
        return document[0].get_pixmap(dpi=dpi).tobytes("png")


def _scanned_pdf_bytes() -> bytes:
    """A PDF whose only content is a page image, so native extraction finds
    nothing and the OCR fallback has to take over."""
    import fitz

    page_image = _render_png()
    document = fitz.open()
    with fitz.open(stream=page_image, filetype="png") as image:
        rect = image[0].rect
        pdf_bytes = image.convert_to_pdf()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as converted:
        page = document.new_page(width=rect.width, height=rect.height)
        page.show_pdf_page(rect, converted, 0)
    data = document.tobytes()
    document.close()
    return data


def _convert_png(target_format: str) -> bytes:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(_render_png())) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=target_format)
        return buffer.getvalue()


def _blank_png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (900, 500), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _docx_bytes() -> bytes:
    """Word invoice: the header lines as paragraphs plus a real line-item
    table, since the table is the part DOCX preserves and OCR cannot."""
    import io

    from docx import Document as WordDocument

    document = WordDocument()
    for line in INVOICE_LINES:
        document.add_paragraph(line)

    table = document.add_table(rows=1, cols=len(LINE_ITEM_ROWS[0]))
    for index, heading in enumerate(LINE_ITEM_ROWS[0]):
        table.rows[0].cells[index].text = heading
    for row in LINE_ITEM_ROWS[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _xlsx_bytes() -> bytes:
    """Spreadsheet invoice: label and value in adjacent cells rather than on
    one line, which is the shape the extractor has to flatten."""
    import io

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Invoice"
    for line in INVOICE_LINES:
        label, separator, value = line.partition(":")
        sheet.append([label, value.strip()] if separator else [label])
    sheet.append([])
    for row in LINE_ITEM_ROWS:
        sheet.append(list(row))

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


_BUILDERS = {
    "text_pdf": _text_pdf_bytes,
    "scanned_pdf": _scanned_pdf_bytes,
    "jpg": lambda: _convert_png("JPEG"),
    "png": _render_png,
    "webp": lambda: _convert_png("WEBP"),
    "blank_png": _blank_png_bytes,
    "corrupt_pdf": lambda: b"%PDF-1.4 this is not actually a pdf" + b"\x00\xff" * 40,
    "docx": _docx_bytes,
    "xlsx": _xlsx_bytes,
    "unsupported": lambda: "\n".join(INVOICE_LINES).encode("utf-8"),
}


def sample_bytes(kind: str) -> bytes:
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown sample kind {kind!r}; expected one of {', '.join(SAMPLE_KINDS)}")
    return _BUILDERS[kind]()


def write_sample(kind: str, directory: Path, name: str | None = None) -> Path:
    path = Path(directory) / (name or _DEFAULT_NAMES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sample_bytes(kind))
    return path
