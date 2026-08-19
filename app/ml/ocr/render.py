"""Render one synthetic invoice into every format the upload endpoint accepts.

The point of rendering the *same* invoice seven ways is that each format
reaches the parser through a different route, and those routes have very
different failure modes:

    pdf_native  pdfplumber text layer      — exact characters, table structure
    docx/xlsx   python-docx / openpyxl     — exact characters, no page layout
    jpg/png/webp  OCR of a degraded photo  — noisy characters, no structure
    pdf_scan    OCR of an image-only PDF   — same, via the PDF fallback path

A field extractor that only ever sees the first route quietly depends on
things OCR destroys (column alignment, an unbroken "Total: 1,234.00" line).
Rendering to all seven and training across them is what stops that.
"""
from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from .augment import degrade
from .labels import FORMATS
from .synth import RUPEE, SynthInvoice, group_digits

logger = logging.getLogger(__name__)

PAGE_WIDTH, PAGE_HEIGHT = 595.0, 842.0  # A4 in points

# Raster resolution for the scan/photo formats. 150 dpi puts an A4 page at
# 1240x1754 — the range a phone photo or an office scanner actually lands in,
# and low enough that OCR of a corpus finishes this decade. OCR cost scales
# with pixel count, and CPU PaddleOCR is the dominant cost of building the
# corpus by two orders of magnitude.
RASTER_DPI = 150

# Windows ships Arial, which covers U+20B9 (₹). reportlab's built-in
# Helvetica does not, and would emit a black box that OCR reads as garbage.
# Where no Unicode face is available the renderer falls back to "Rs." so the
# document stays readable instead of rendering a glyph nobody can parse.
_UNICODE_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_unicode_font_ready: bool | None = None


def _ensure_fonts() -> bool:
    """Register a Unicode TTF with reportlab once per process. Returns whether
    the rupee sign can be drawn."""
    global _unicode_font_ready, _FONT_REGULAR, _FONT_BOLD
    if _unicode_font_ready is not None:
        return _unicode_font_ready

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for candidate in _UNICODE_FONT_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            bold = candidate.with_name(candidate.stem + "bd" + candidate.suffix)
            pdfmetrics.registerFont(TTFont("InvoiceSans", str(candidate)))
            pdfmetrics.registerFont(TTFont("InvoiceSans-Bold", str(bold if bold.exists() else candidate)))
            _FONT_REGULAR, _FONT_BOLD = "InvoiceSans", "InvoiceSans-Bold"
            _unicode_font_ready = True
            return True
        except Exception as exc:  # a font file present but unusable is not fatal
            logger.warning("ocr_model.font_registration_failed font=%s error=%s", candidate, exc)

    _unicode_font_ready = False
    return False


def _money(invoice: SynthInvoice, value: Decimal, unicode_ok: bool) -> str:
    currency = invoice.currency
    if currency == RUPEE and not unicode_ok:
        currency = "Rs. "
    return f"{currency}{group_digits(value, invoice.indian_grouping)}"


@dataclass
class RenderedDocument:
    doc_id: str
    fmt: str
    path: Path
    severity: str


def render_pdf(invoice: SynthInvoice, path: Path) -> Path:
    """Draw a text-layer PDF. This is both a corpus format in its own right
    and the master copy every raster format is rasterised from, so that all
    formats of one document carry byte-identical content."""
    from reportlab.pdfgen import canvas

    unicode_ok = _ensure_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    page = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    left, right = 42.0, PAGE_WIDTH - 42.0
    cursor = PAGE_HEIGHT - 56.0

    def line(text: str, x: float, y: float, size: float = 9.0, bold: bool = False) -> None:
        page.setFont(_FONT_BOLD if bold else _FONT_REGULAR, size)
        page.drawString(x, y, text)

    def right_line(text: str, x: float, y: float, size: float = 9.0, bold: bool = False) -> None:
        page.setFont(_FONT_BOLD if bold else _FONT_REGULAR, size)
        page.drawRightString(x, y, text)

    line("TAX INVOICE", left, cursor, size=17, bold=True)
    cursor -= 26

    # Invoice metadata — always top-right, the way most invoice templates do it.
    meta_y = PAGE_HEIGHT - 60.0
    for label_key, value in (
        ("invoice_number", invoice.invoice_number),
        ("invoice_date", invoice.formatted_date(invoice.invoice_date)),
        ("due_date", invoice.formatted_date(invoice.due_date) if invoice.show_due_date else None),
    ):
        if value is None:
            continue
        right_line(f"{invoice.wording[label_key]}: {value}", right, meta_y, size=9.5)
        meta_y -= 14

    def party_block(x: float, y: float, heading: str, name: str, address: list[str], gstin: str) -> float:
        line(f"{heading}:", x, y, size=9, bold=True)
        y -= 14
        line(name, x, y, size=10.5, bold=True)
        y -= 13
        for address_line in address:
            line(address_line, x, y, size=8.5)
            y -= 11
        line(f"{invoice.wording['gstin']}: {gstin}", x, y, size=9)
        return y - 16

    if invoice.parties_side_by_side:
        end_left = party_block(left, cursor, invoice.wording["vendor_block"], invoice.vendor_name, invoice.vendor_address, invoice.vendor_gstin)
        end_right = party_block(PAGE_WIDTH / 2 + 10, cursor, invoice.wording["customer_block"], invoice.customer_name, invoice.customer_address, invoice.customer_gstin)
        cursor = min(end_left, end_right)
    else:
        cursor = party_block(left, cursor, invoice.wording["vendor_block"], invoice.vendor_name, invoice.vendor_address, invoice.vendor_gstin)
        cursor = party_block(left, cursor, invoice.wording["customer_block"], invoice.customer_name, invoice.customer_address, invoice.customer_gstin)

    # Line-item table.
    columns = (left, left + 26, left + 250, left + 300, left + 350, left + 425, right)
    page.setLineWidth(0.6)
    page.line(left, cursor + 4, right, cursor + 4)
    cursor -= 10
    headers = ("#", "Description", "HSN/SAC", "Qty", "Unit", "Rate", "Amount")
    for index, header in enumerate(headers):
        if index >= len(headers) - 2:
            right_line(header, columns[index], cursor, size=8.5, bold=True)
        else:
            line(header, columns[index], cursor, size=8.5, bold=True)
    cursor -= 6
    page.line(left, cursor, right, cursor)
    cursor -= 13

    for index, item in enumerate(invoice.line_items, start=1):
        line(str(index), columns[0], cursor, size=8.5)
        line(item.description, columns[1], cursor, size=8.5)
        line(item.hsn_sac, columns[2], cursor, size=8.5)
        line(str(item.quantity), columns[3], cursor, size=8.5)
        line(item.unit, columns[4], cursor, size=8.5)
        right_line(group_digits(item.unit_price, invoice.indian_grouping), columns[5], cursor, size=8.5)
        right_line(group_digits(item.taxable_value, invoice.indian_grouping), columns[6], cursor, size=8.5)
        cursor -= 14

    cursor -= 2
    page.line(left + 300, cursor, right, cursor)
    cursor -= 14

    totals: list[tuple[str, Decimal, bool]] = [(invoice.wording["subtotal"], invoice.subtotal, False)]
    if invoice.interstate:
        totals.append((f"{invoice.wording['igst']} @ {invoice.gst_rate}%", invoice.igst_amount, False))
    else:
        half = invoice.gst_rate / 2
        totals.append((f"{invoice.wording['cgst']} @ {half}%", invoice.cgst_amount, False))
        totals.append((f"{invoice.wording['sgst']} @ {half}%", invoice.sgst_amount, False))
    if invoice.show_tax_total:
        totals.append((invoice.wording["tax"], invoice.tax_amount, False))
    totals.append((invoice.wording["total"], invoice.total_amount, True))

    for label, value, bold in totals:
        line(f"{label}:", left + 300, cursor, size=9.5 if bold else 9, bold=bold)
        right_line(_money(invoice, value, unicode_ok), right, cursor, size=9.5 if bold else 9, bold=bold)
        cursor -= 15

    cursor -= 12
    line("Declaration: We certify that the particulars given above are true and correct.", left, cursor, size=7.5)
    cursor -= 11
    line(f"Place of Supply: {invoice.customer_address[-1]}", left, cursor, size=7.5)

    page.showPage()
    page.save()
    return path


def render_docx(invoice: SynthInvoice, path: Path) -> Path:
    """Word rendering. Content matches the PDF; layout does not — a DOCX has
    no page geometry to recover, so the parser sees paragraphs and a table
    and nothing else."""
    from docx import Document as WordDocument

    path.parent.mkdir(parents=True, exist_ok=True)
    document = WordDocument()
    document.add_heading("TAX INVOICE", level=1)

    document.add_paragraph(f"{invoice.wording['invoice_number']}: {invoice.invoice_number}")
    document.add_paragraph(f"{invoice.wording['invoice_date']}: {invoice.formatted_date(invoice.invoice_date)}")
    if invoice.show_due_date:
        document.add_paragraph(f"{invoice.wording['due_date']}: {invoice.formatted_date(invoice.due_date)}")

    for heading_key, name, address, gstin in (
        ("vendor_block", invoice.vendor_name, invoice.vendor_address, invoice.vendor_gstin),
        ("customer_block", invoice.customer_name, invoice.customer_address, invoice.customer_gstin),
    ):
        document.add_paragraph(f"{invoice.wording[heading_key]}:")
        document.add_paragraph(name)
        for address_line in address:
            document.add_paragraph(address_line)
        document.add_paragraph(f"{invoice.wording['gstin']}: {gstin}")

    table = document.add_table(rows=1, cols=7)
    header = table.rows[0].cells
    for index, title in enumerate(("#", "Description", "HSN/SAC", "Qty", "Unit", "Rate", "Amount")):
        header[index].text = title
    for index, item in enumerate(invoice.line_items, start=1):
        cells = table.add_row().cells
        values = (
            str(index), item.description, item.hsn_sac, str(item.quantity), item.unit,
            group_digits(item.unit_price, invoice.indian_grouping),
            group_digits(item.taxable_value, invoice.indian_grouping),
        )
        for position, value in enumerate(values):
            cells[position].text = value

    document.add_paragraph(f"{invoice.wording['subtotal']}: {invoice.money(invoice.subtotal)}")
    if invoice.interstate:
        document.add_paragraph(f"{invoice.wording['igst']}: {invoice.money(invoice.igst_amount)}")
    else:
        document.add_paragraph(f"{invoice.wording['cgst']}: {invoice.money(invoice.cgst_amount)}")
        document.add_paragraph(f"{invoice.wording['sgst']}: {invoice.money(invoice.sgst_amount)}")
    if invoice.show_tax_total:
        document.add_paragraph(f"{invoice.wording['tax']}: {invoice.money(invoice.tax_amount)}")
    document.add_paragraph(f"{invoice.wording['total']}: {invoice.money(invoice.total_amount)}")

    document.save(str(path))
    return path


def render_xlsx(invoice: SynthInvoice, path: Path) -> Path:
    """Spreadsheet rendering. Label and value land in adjacent *cells*, which
    the extractor joins with a separator — a different neighbourhood shape
    from the PDF's "label ... value" line, and one the model has to handle."""
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Invoice"

    rows: list[list[str]] = [
        ["TAX INVOICE", ""],
        [invoice.wording["invoice_number"], invoice.invoice_number],
        [invoice.wording["invoice_date"], invoice.formatted_date(invoice.invoice_date)],
    ]
    if invoice.show_due_date:
        rows.append([invoice.wording["due_date"], invoice.formatted_date(invoice.due_date)])
    rows.append([])

    for heading_key, name, address, gstin in (
        ("vendor_block", invoice.vendor_name, invoice.vendor_address, invoice.vendor_gstin),
        ("customer_block", invoice.customer_name, invoice.customer_address, invoice.customer_gstin),
    ):
        rows.append([f"{invoice.wording[heading_key]}:", ""])
        rows.append([name, ""])
        rows.extend([[address_line, ""] for address_line in address])
        rows.append([invoice.wording["gstin"], gstin])
        rows.append([])

    rows.append(["#", "Description", "HSN/SAC", "Qty", "Unit", "Rate", "Amount"])
    for index, item in enumerate(invoice.line_items, start=1):
        rows.append([
            str(index), item.description, item.hsn_sac, str(item.quantity), item.unit,
            group_digits(item.unit_price, invoice.indian_grouping),
            group_digits(item.taxable_value, invoice.indian_grouping),
        ])
    rows.append([])

    rows.append([invoice.wording["subtotal"], invoice.money(invoice.subtotal)])
    if invoice.interstate:
        rows.append([invoice.wording["igst"], invoice.money(invoice.igst_amount)])
    else:
        rows.append([invoice.wording["cgst"], invoice.money(invoice.cgst_amount)])
        rows.append([invoice.wording["sgst"], invoice.money(invoice.sgst_amount)])
    if invoice.show_tax_total:
        rows.append([invoice.wording["tax"], invoice.money(invoice.tax_amount)])
    rows.append([invoice.wording["total"], invoice.money(invoice.total_amount)])

    for row in rows:
        sheet.append(row)
    workbook.save(str(path))
    return path


def rasterise(pdf_path: Path, dpi: int = RASTER_DPI) -> np.ndarray:
    """First page of a PDF as a BGR array, ready for degradation."""
    import fitz

    with fitz.open(str(pdf_path)) as document:
        pixmap = document[0].get_pixmap(dpi=dpi)
        raw = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
    if raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)


def write_image(image: np.ndarray, path: Path, quality: int = 82) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        flags = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif suffix == ".webp":
        flags = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    else:
        flags = []
    if not cv2.imwrite(str(path), image, flags):
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def write_image_pdf(image: np.ndarray, path: Path) -> Path:
    """Wrap a degraded page image back into a PDF with no text layer — a
    scanner's output. This is the case that exercises the OCR fallback branch
    in text_extraction.extract_document, which a native PDF never reaches."""
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode page image for PDF")

    document = fitz.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_image(fitz.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT), stream=buffer.tobytes())
    document.save(str(path))
    document.close()
    return path


RASTER_SUFFIXES = {"jpg": ".jpg", "png": ".png", "webp": ".webp"}


def render_all(
    invoice: SynthInvoice,
    out_dir: Path,
    severity: str,
    formats: tuple[str, ...] | None = None,
    dpi: int = RASTER_DPI,
) -> list[RenderedDocument]:
    """Produce the requested formats for one invoice.

    All raster formats share a single degraded page image, so a difference
    between jpg/png/webp results is attributable to the codec alone rather
    than to a different noise draw. `formats` restricts the work: rendering a
    format nobody will read costs a rasterise-and-degrade pass, and skipping
    the raster branch entirely when none are wanted is the difference between
    a fast corpus pass and a slow one.
    """
    out_dir = Path(out_dir)
    wanted = set(formats) if formats is not None else set(FORMATS)
    documents: list[RenderedDocument] = []

    raster_wanted = wanted & (set(RASTER_SUFFIXES) | {"pdf_scan"})
    # The native PDF is the master copy every raster format is derived from,
    # so it is rendered whenever anything raster is wanted, even if the caller
    # did not ask for pdf_native itself.
    native_pdf = render_pdf(invoice, out_dir / f"{invoice.doc_id}.pdf") if (
        "pdf_native" in wanted or raster_wanted
    ) else None

    if "pdf_native" in wanted and native_pdf is not None:
        documents.append(RenderedDocument(invoice.doc_id, "pdf_native", native_pdf, "clean"))
    if "docx" in wanted:
        documents.append(RenderedDocument(invoice.doc_id, "docx", render_docx(invoice, out_dir / f"{invoice.doc_id}.docx"), "clean"))
    if "xlsx" in wanted:
        documents.append(RenderedDocument(invoice.doc_id, "xlsx", render_xlsx(invoice, out_dir / f"{invoice.doc_id}.xlsx"), "clean"))

    if not raster_wanted or native_pdf is None:
        return documents

    page_image = rasterise(native_pdf, dpi=dpi)
    # crc32, not hash(): PYTHONHASHSEED randomises str hashing per process,
    # which would make the same document degrade differently on every run.
    degraded = degrade(page_image, severity, seed=zlib.crc32(invoice.doc_id.encode()))

    for fmt, suffix in RASTER_SUFFIXES.items():
        if fmt in wanted:
            written = write_image(degraded, out_dir / f"{invoice.doc_id}{suffix}")
            documents.append(RenderedDocument(invoice.doc_id, fmt, written, severity))

    if "pdf_scan" in wanted:
        scan_pdf = write_image_pdf(degraded, out_dir / f"{invoice.doc_id}_scan.pdf")
        documents.append(RenderedDocument(invoice.doc_id, "pdf_scan", scan_pdf, severity))

    return documents
