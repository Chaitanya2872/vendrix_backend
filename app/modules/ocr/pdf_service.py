"""PDF handling: use the text layer where there is one, rasterise where there
is not — and decide that **per page**, not per document.

Per-page matters more than it sounds. A real invoice PDF is routinely mixed:
a generated first page with a perfect text layer, followed by a scanned
annexure or a photographed delivery note. Deciding once for the whole file
means either OCR-ing pages that did not need it — minutes of wasted compute
each — or reading nothing at all from the pages that did.

**One coordinate system.** PDF text coordinates are in points (72 per inch);
rasterised pages are in pixels at the render DPI. Everything downstream of
this module works in raster pixels, so native coordinates are scaled here on
the way out. Without that, a box from a text page and a box from a scanned
page would be in different units, and every geometric rule in the layout
stage would silently be wrong on half the pages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import settings

from .dto import SOURCE_NATIVE, OcrLine, OcrPage, OcrWord
from .exceptions import OcrPageFailed
from .geometry import BoundingBox

logger = logging.getLogger(__name__)

# A page needs at least this many characters of text layer before it is
# trusted as a text page. Well below a real invoice page (1,000+) and well
# above what a scanned page's stray artefacts produce (a page number, a
# stamp's embedded label) — the gap between those two populations is wide,
# which is what makes a simple threshold workable here.
MIN_NATIVE_CHARACTERS_PER_PAGE = 120

# A page whose area is this covered by raster images is a picture of a
# document, not a document. Measured directly rather than inferred from text
# sparsity: "there is a full-page image here" is the actual thing that makes
# a page a scan, and a page can be legitimately sparse (a terms-and-
# conditions page, a signature page) without being one.
SCAN_IMAGE_COVERAGE = 0.60

# On an image-dominated page, text must still cover this share before it is
# trusted. Below it, the text is furniture printed alongside the scan — a
# scanner's footer, a form field, a Bates stamp — and the invoice itself is
# pixels. Real invoice text pages measure several times this.
MIN_TEXT_AREA_RATIO_ON_SCAN = 0.02

# On a page with no raster content, this many characters is enough to trust
# the text layer however sparse it is — there are no pixels for OCR to read
# beyond the same glyphs. Below it the page may hold text drawn as vector
# paths, which OCR *can* read and the text layer cannot report.
MIN_CHARACTERS_WITHOUT_IMAGES = 20

PDF_POINTS_PER_INCH = 72.0


@dataclass
class PageDecision:
    """What this module decided to do with one page, and why.

    The reason is carried rather than logged and dropped because "why did
    this page get OCR'd" is the first question asked when a document takes
    twenty minutes, and reconstructing it afterwards means re-running.
    """

    page_number: int
    route: str                      # "native" | "raster"
    reason: str
    character_count: int = 0
    text_area_ratio: float = 0.0
    has_invisible_text: bool = False
    rotation: int = 0
    image_coverage: float = 0.0


@dataclass
class PdfPage:
    """A page prepared for the next stage: either already-read text, or an
    image that still needs OCR. Exactly one of the two is set."""

    page_number: int
    decision: PageDecision
    native: OcrPage | None = None
    image: np.ndarray | None = None

    @property
    def needs_ocr(self) -> bool:
        return self.native is None


@dataclass
class PdfAnalysis:
    pages: list[PdfPage] = field(default_factory=list)
    page_count: int = 0
    render_dpi: int = 300

    @property
    def native_page_count(self) -> int:
        return sum(1 for page in self.pages if not page.needs_ocr)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for page in self.pages if page.needs_ocr)

    @property
    def is_mixed(self) -> bool:
        return 0 < self.native_page_count < self.page_count

    def decisions(self) -> list[dict]:
        return [
            {
                "page_number": page.decision.page_number,
                "route": page.decision.route,
                "reason": page.decision.reason,
                "characters": page.decision.character_count,
                "text_area_ratio": round(page.decision.text_area_ratio, 5),
                "image_coverage": round(page.decision.image_coverage, 4),
                "rotation": page.decision.rotation,
            }
            for page in self.pages
        ]


def _open(path: str | Path):
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - PyMuPDF is a hard dependency
        raise OcrPageFailed("PyMuPDF is not installed; PDFs cannot be read.") from exc

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        raise OcrPageFailed(f"Unable to open PDF: {exc}") from exc

    if document.needs_pass:
        document.close()
        raise OcrPageFailed("This PDF is password-protected.")
    return document


def analyse(
    path: str | Path,
    render_dpi: int | None = None,
    min_characters: int = MIN_NATIVE_CHARACTERS_PER_PAGE,
    force_ocr: bool = False,
) -> PdfAnalysis:
    """Read every page, choosing the cheapest route that will actually work.

    `force_ocr` exists for the case where a PDF's text layer is present but
    known-bad — some scanning software embeds a garbage layer that is worse
    than re-reading the pixels. It is not the default because paying for OCR
    on a clean generated invoice is the single most expensive mistake this
    pipeline can make.
    """
    dpi = render_dpi or settings.ocr_render_dpi
    scale = dpi / PDF_POINTS_PER_INCH

    document = _open(path)
    pages: list[PdfPage] = []

    try:
        for index, page in enumerate(document):
            page_number = index + 1
            placed = _native_words(page, scale)
            characters = sum(len(item.word.text) for item in placed)
            area_ratio = _text_area_ratio(placed, page, scale)
            invisible = _has_invisible_text(page)
            image_coverage = _image_coverage(page)

            has_images = image_coverage > 0.01

            if force_ocr:
                decision = PageDecision(
                    page_number, "raster", "OCR forced by caller",
                    characters, area_ratio, invisible, page.rotation, image_coverage,
                )
            elif characters < min_characters and not has_images and characters >= MIN_CHARACTERS_WITHOUT_IMAGES:
                # Sparse text, but the page carries no raster content at all —
                # so there are no pixels holding anything the text layer does
                # not already state exactly. OCR could only re-read the same
                # glyphs, less accurately, for two minutes a page. A short
                # covering letter or a single-item invoice lands here.
                decision = PageDecision(
                    page_number, "native",
                    f"sparse ({characters} characters) but the page has no images, "
                    "so OCR could not find anything more",
                    characters, area_ratio, invisible, page.rotation, image_coverage,
                )
            elif characters < min_characters:
                decision = PageDecision(
                    page_number, "raster",
                    f"text layer holds only {characters} characters",
                    characters, area_ratio, invisible, page.rotation, image_coverage,
                )
            elif image_coverage >= SCAN_IMAGE_COVERAGE and area_ratio < MIN_TEXT_AREA_RATIO_ON_SCAN:
                # A full-page image with a sprinkle of text on top: the text
                # is the scanner's own furniture, and the invoice is pixels.
                decision = PageDecision(
                    page_number, "raster",
                    f"page is {image_coverage:.0%} covered by an image and its text covers "
                    f"only {area_ratio:.2%} of the page",
                    characters, area_ratio, invisible, page.rotation, image_coverage,
                )
            else:
                decision = PageDecision(
                    page_number, "native",
                    "usable text layer" + (" (from a prior OCR pass)" if invisible else ""),
                    characters, area_ratio, invisible, page.rotation, image_coverage,
                )

            if decision.route == "native":
                pages.append(PdfPage(page_number, decision, native=_build_native_page(page, placed, scale)))
            else:
                pages.append(PdfPage(page_number, decision, image=render_page(page, dpi)))

        analysis = PdfAnalysis(pages=pages, page_count=len(pages), render_dpi=dpi)
    finally:
        document.close()

    logger.info(
        "pdf.analysed pages=%d native=%d ocr=%d mixed=%s",
        analysis.page_count, analysis.native_page_count, analysis.ocr_page_count, analysis.is_mixed,
    )
    return analysis


@dataclass
class _PlacedWord:
    """A word plus the block/line it belongs to.

    The grouping keys travel with the word rather than being re-derived
    later: PyMuPDF's word list is filtered here (blank entries dropped), and
    a second pass that re-filtered independently would be one edit away from
    misaligning words with their own line numbers.
    """

    word: OcrWord
    block_no: int
    line_no: int


def _native_words(page, scale: float) -> list[_PlacedWord]:
    """Extract every word with its exact box, scaled into raster pixels.

    PyMuPDF reports a rectangle per word, so unlike the OCR path these boxes
    are exact rather than apportioned — which is what fixes the two-column
    seller/buyer block that a line-level text extractor collapses into one
    unusable string.
    """
    try:
        raw = page.get_text("words")
    except Exception as exc:
        logger.warning("pdf.word_extraction_failed page=%s error=%s", page.number + 1, exc)
        return []

    placed: list[_PlacedWord] = []
    for entry in raw:
        # (x0, y0, x1, y1, text, block_no, line_no, word_no)
        x0, y0, x1, y1, text = entry[0], entry[1], entry[2], entry[3], entry[4]
        if not str(text).strip():
            continue
        placed.append(
            _PlacedWord(
                word=OcrWord(
                    text=str(text),
                    box=BoundingBox(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
                    confidence=1.0,      # exact characters, not a recognition guess
                    approximate_box=False,
                ),
                block_no=int(entry[5]) if len(entry) > 5 else 0,
                line_no=int(entry[6]) if len(entry) > 6 else 0,
            )
        )
    return placed


def _build_native_page(page, placed: list[_PlacedWord], scale: float) -> OcrPage:
    """Assemble words into lines using PyMuPDF's own block/line structure.

    Its grouping is authoritative — it comes from the PDF's text objects —
    so re-deriving lines geometrically here would be strictly worse than
    using what the format already states.
    """
    grouped: dict[tuple[int, int], list[OcrWord]] = {}
    order: list[tuple[int, int]] = []
    for item in placed:
        key = (item.block_no, item.line_no)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item.word)

    lines: list[OcrLine] = []
    for key in order:
        members = sorted(grouped[key], key=lambda word: word.box.x0)
        text = " ".join(word.text for word in members).strip()
        if not text:
            continue
        box = BoundingBox.union(word.box for word in members)
        lines.append(
            OcrLine(
                text=text,
                box=box,
                confidence=1.0,
                polygon=[(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)],
                words=members,
                page_number=page.number + 1,
            )
        )

    lines.sort(key=lambda line: (round(line.box.y0, 1), line.box.x0))
    for index, line in enumerate(lines):
        line.line_index = index

    rect = page.rect
    return OcrPage(
        page_number=page.number + 1,
        width=rect.width * scale,
        height=rect.height * scale,
        lines=lines,
        source=SOURCE_NATIVE,
        rotation_applied=float(page.rotation or 0),
    )


def _text_area_ratio(placed: list[_PlacedWord], page, scale: float) -> float:
    if not placed:
        return 0.0
    page_area = (page.rect.width * scale) * (page.rect.height * scale)
    if page_area <= 0:
        return 0.0
    return sum(item.word.box.area for item in placed) / page_area


def _image_coverage(page) -> float:
    """Fraction of the page covered by placed raster images.

    Uses the union area of image placements rather than their sum, so a scan
    split into horizontal strips — which is how some scanner drivers emit a
    page — does not report 300% coverage. Overlapping placements are counted
    once via a coarse occupancy grid, which is cheaper and more robust than
    exact rectangle union for the precision this decision needs.
    """
    try:
        placements = page.get_image_info()
    except Exception:
        return 0.0
    if not placements:
        return 0.0

    page_rect = page.rect
    if page_rect.width <= 0 or page_rect.height <= 0:
        return 0.0

    resolution = 32  # 32x32 cells is ample for a >=60% threshold
    occupied: set[tuple[int, int]] = set()
    cell_width = page_rect.width / resolution
    cell_height = page_rect.height / resolution

    for placement in placements:
        bbox = placement.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        for column in range(resolution):
            cell_x = page_rect.x0 + (column + 0.5) * cell_width
            if not (x0 <= cell_x <= x1):
                continue
            for row in range(resolution):
                cell_y = page_rect.y0 + (row + 0.5) * cell_height
                if y0 <= cell_y <= y1:
                    occupied.add((column, row))

    return len(occupied) / (resolution * resolution)


def _has_invisible_text(page) -> bool:
    """Whether the page carries text drawn in invisible render mode.

    That is the signature of a searchable PDF produced by someone else's OCR.
    Its text is still worth using — it is free, and re-reading the pixels
    costs minutes — but it is a recognition result, not typed characters, so
    the confidence stage should know not to treat it as exact.
    """
    try:
        raw = page.get_text("rawdict")
    except Exception:
        return False
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                # Render mode 3 = fill none, stroke none: drawn but invisible.
                if span.get("type") == 3 or span.get("render_mode") == 3:
                    return True
    return False


def render_page(page, dpi: int) -> np.ndarray:
    """Rasterise one page into a BGR image for OCR.

    BGR rather than RGB because every consumer downstream is OpenCV, and a
    channel-order mismatch is the kind of bug that shows up as slightly worse
    OCR accuracy rather than as an error.
    """
    import cv2

    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    except Exception as exc:
        raise OcrPageFailed(f"Unable to render page: {exc}", page_number=page.number + 1) from exc

    buffer = np.frombuffer(pixmap.samples, dtype=np.uint8)
    image = buffer.reshape(pixmap.height, pixmap.width, pixmap.n)

    if pixmap.n == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if pixmap.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_all(path: str | Path, dpi: int | None = None) -> list[np.ndarray]:
    """Rasterise every page regardless of its text layer.

    For the review UI, which overlays boxes on a rendered page and therefore
    needs an image even for pages that were read natively.
    """
    dpi = dpi or settings.ocr_render_dpi
    document = _open(path)
    try:
        return [render_page(page, dpi) for page in document]
    finally:
        document.close()


def page_count(path: str | Path) -> int:
    document = _open(path)
    try:
        return document.page_count
    finally:
        document.close()
