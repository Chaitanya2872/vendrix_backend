"""Structured OCR output: the single shape every later stage consumes.

The point of this module is that **nothing downstream knows how the text was
obtained**. A native-text PDF page, a rasterised scan run through PaddleOCR,
and a spreadsheet row all arrive as an `OcrPage` full of `OcrLine`s carrying
boxes and confidences. Layout analysis, table detection and field extraction
are written once against that shape instead of once per input format.

That uniformity is also why confidence is mandatory rather than optional:
native extraction is exact, so it reports 1.0, and a consumer weighing a
candidate never has to branch on "did this come from OCR". The `source` field
is still recorded, because *reporting* to a human which pages needed OCR is
useful even when the algorithms do not care.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from .geometry import BoundingBox, Point

# How the text on a page was recovered. Kept as plain strings rather than an
# enum because they are written straight into JSON columns and API payloads.
SOURCE_NATIVE = "native"   # PDF text layer, exact characters
SOURCE_OCR = "ocr"         # PaddleOCR over a rasterised page
SOURCE_OFFICE = "office"   # docx/xlsx cell text, exact characters


@dataclass
class OcrWord:
    """One whitespace-delimited token with its own box.

    For native PDF pages these boxes are exact — PyMuPDF reports a rectangle
    per word. For OCR'd pages PaddleOCR detects *lines*, not words, so word
    boxes are apportioned across the line box by character width (see
    `split_line_into_words`). That approximation is good enough for the
    questions the layout stage asks of it (which column is this token in,
    what sits immediately right of this label) and is documented at the point
    where it is made rather than hidden here.
    """

    text: str
    box: BoundingBox
    confidence: float = 1.0
    approximate_box: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "approximate_box": self.approximate_box,
        }


@dataclass
class OcrLine:
    """A single detected text line."""

    text: str
    box: BoundingBox
    confidence: float = 1.0
    polygon: list[Point] = field(default_factory=list)
    words: list[OcrWord] = field(default_factory=list)
    page_number: int = 1
    line_index: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self, include_words: bool = False) -> dict:
        payload = {
            "text": self.text,
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "page_number": self.page_number,
            "line_index": self.line_index,
        }
        if include_words:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload


@dataclass
class OcrPage:
    """One page of a document, however that page was read."""

    page_number: int
    width: float
    height: float
    lines: list[OcrLine] = field(default_factory=list)
    source: str = SOURCE_OCR
    rotation_applied: float = 0.0
    preprocessing_applied: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Reading-order text. Every legacy consumer of this pipeline works
        from a flat string, so producing one from the structured form keeps
        those callers working unchanged while new code uses the geometry."""
        return "\n".join(line.text for line in self.lines if line.text.strip())

    @property
    def used_ocr(self) -> bool:
        return self.source == SOURCE_OCR

    @property
    def mean_confidence(self) -> float:
        scores = [line.confidence for line in self.lines if not line.is_empty]
        return round(mean(scores), 4) if scores else 0.0

    @property
    def low_confidence_lines(self) -> list[OcrLine]:
        return [line for line in self.lines if line.confidence < 0.80 and not line.is_empty]

    @property
    def character_count(self) -> int:
        return sum(len(line.text.strip()) for line in self.lines)

    def words(self) -> list[OcrWord]:
        return [word for line in self.lines for word in line.words]

    def to_dict(self, include_words: bool = False) -> dict:
        return {
            "page_number": self.page_number,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "source": self.source,
            "rotation_applied": self.rotation_applied,
            "preprocessing_applied": list(self.preprocessing_applied),
            "mean_confidence": self.mean_confidence,
            "line_count": len(self.lines),
            "lines": [line.to_dict(include_words=include_words) for line in self.lines],
        }


@dataclass
class OcrDocument:
    """Every page of one uploaded file."""

    pages: list[OcrPage] = field(default_factory=list)
    filename: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def used_ocr(self) -> bool:
        """True when *any* page needed OCR. A mixed PDF — text pages plus a
        scanned annexure — is an OCR document for reporting purposes, because
        the caveats that come with OCR apply to the result as a whole."""
        return any(page.used_ocr for page in self.pages)

    @property
    def mean_confidence(self) -> float:
        scores = [line.confidence for page in self.pages for line in page.lines if not line.is_empty]
        return round(mean(scores), 4) if scores else 0.0

    @property
    def character_count(self) -> int:
        return sum(page.character_count for page in self.pages)

    def lines(self) -> list[OcrLine]:
        return [line for page in self.pages for line in page.lines]

    def page(self, page_number: int) -> OcrPage | None:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_dict(self, include_words: bool = False) -> dict:
        return {
            "filename": self.filename,
            "page_count": self.page_count,
            "used_ocr": self.used_ocr,
            "mean_confidence": self.mean_confidence,
            "pages": [page.to_dict(include_words=include_words) for page in self.pages],
        }


def split_line_into_words(
    text: str,
    box: BoundingBox,
    confidence: float,
) -> list[OcrWord]:
    """Apportion a line's box across its whitespace-delimited tokens.

    PaddleOCR's detector emits one polygon per *line*, but geometric field
    matching needs to know where within that line a token sits — "the value is
    the token immediately right of the label" is meaningless at line
    granularity, and an invoice metadata grid puts three label/value pairs on
    one line.

    Width is apportioned by character count including the separating spaces,
    which assumes a roughly monospaced advance. For proportional fonts that is
    wrong in detail — an 'i' is narrower than a 'W' — but the error is bounded
    by the line height and never reorders tokens, so column assignment and
    left-right adjacency both survive it. Every box produced here is flagged
    `approximate_box=True` so a consumer that needs exact geometry (and a
    native-PDF page can give it) can tell the difference.
    """
    stripped = text.strip()
    if not stripped:
        return []

    tokens = stripped.split()
    if len(tokens) == 1:
        return [OcrWord(text=tokens[0], box=box, confidence=confidence, approximate_box=False)]

    total_characters = len(stripped)
    if total_characters == 0 or box.width <= 0:
        return [OcrWord(text=token, box=box, confidence=confidence, approximate_box=True) for token in tokens]

    character_width = box.width / total_characters
    words: list[OcrWord] = []
    cursor = 0  # character offset within the stripped line
    for token in tokens:
        start = stripped.find(token, cursor)
        if start < 0:  # defensive: token must be present, but never trust find()
            start = cursor
        end = start + len(token)
        words.append(
            OcrWord(
                text=token,
                box=BoundingBox(
                    box.x0 + start * character_width,
                    box.y0,
                    box.x0 + end * character_width,
                    box.y1,
                ),
                confidence=confidence,
                approximate_box=True,
            )
        )
        cursor = end
    return words
