"""The OCR stage's public surface.

Callers hand this module an image and get back an `OcrPage`: text lines in
reading order, each with a box, a polygon and a confidence. Nothing outside
this package imports `engine` directly.

The one non-obvious job here is **merging detections into lines**. PaddleOCR
detects text regions, and on an invoice a single visual line is routinely
split into several: `Invoice No:` and `INV-2026-0042` are separated by enough
whitespace that the detector treats them as two regions, as are the columns
of a borderless table. Leaving them split loses the "label then value"
adjacency that field extraction runs on; merging *everything* on a y-band
would glue a table's leftmost and rightmost columns into one string. So the
merge is gap-aware: fragments join only when the horizontal air between them
is small relative to the text height, which is the scale at which a space
character stops looking like a column boundary.

Fragments are kept individually addressable as words regardless, so a
consumer that wants the original granularity still has it.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Callable

import numpy as np

from app.core.config import settings

from . import engine
from .dto import (
    SOURCE_OCR,
    OcrDocument,
    OcrLine,
    OcrPage,
    OcrWord,
    split_line_into_words,
)
from .exceptions import OcrEngineUnavailable, OcrPageFailed
from .geometry import BoundingBox

logger = logging.getLogger(__name__)

# Two fragments on the same visual line are merged when the gap between them
# is under this multiple of their text height. ~1.5x is comfortably wider
# than an inter-word space in any normal face and comfortably narrower than
# the gutter between table columns, which is where the two cases separate.
MAX_MERGE_GAP_RATIO = 1.5

# Fragments must share this fraction of their height to be considered the
# same line at all. Below it they are stacked rows, not neighbours.
MIN_LINE_OVERLAP = 0.45

# Detections below `settings.ocr_min_confidence` are dropped as noise
# (stamps, logos, watermarks, scan artefacts): carrying them forward gives
# the field extractor plausible-looking garbage to match against. The
# configured default is deliberately low — a dropped line cannot be recovered
# downstream, while retained noise merely competes on score and loses.


def _merge_fragments(detections: list[engine.RawDetection]) -> list[tuple[str, BoundingBox, float, list[OcrWord]]]:
    """Group same-line detections into merged lines.

    Returns tuples rather than `OcrLine`s so the caller owns page numbering
    and line indexing — this function has no idea which page it is on.
    """
    if not detections:
        return []

    boxes = [BoundingBox.from_polygon(detection.polygon) for detection in detections]
    order = sorted(range(len(detections)), key=lambda index: (boxes[index].y0, boxes[index].x0))

    # Build bands greedily against a running extent, then split each band on
    # horizontal gaps. Doing it in that order matters: gap size is only
    # meaningful between fragments already known to share a line.
    bands: list[list[int]] = []
    extents: list[BoundingBox] = []
    for index in order:
        box = boxes[index]
        for position, extent in enumerate(extents):
            if box.vertical_overlap(extent) >= MIN_LINE_OVERLAP:
                bands[position].append(index)
                extents[position] = extent.merged_with(box)
                break
        else:
            bands.append([index])
            extents.append(box)

    merged: list[tuple[str, BoundingBox, float, list[OcrWord]]] = []
    for band in bands:
        band.sort(key=lambda index: boxes[index].x0)

        run: list[int] = []
        for index in band:
            if run:
                previous = boxes[run[-1]]
                current = boxes[index]
                height = max(previous.height, current.height, 1.0)
                if previous.horizontal_gap(current) > MAX_MERGE_GAP_RATIO * height:
                    merged.append(_assemble(run, detections, boxes))
                    run = []
            run.append(index)
        if run:
            merged.append(_assemble(run, detections, boxes))

    merged.sort(key=lambda item: (item[1].y0, item[1].x0))
    return merged


def _assemble(
    indices: list[int],
    detections: list[engine.RawDetection],
    boxes: list[BoundingBox],
) -> tuple[str, BoundingBox, float, list[OcrWord]]:
    """Combine a run of adjacent fragments into one line.

    Confidence is the *minimum* of the fragments, not the mean: a line whose
    amount was read at 0.42 is a 0.42-confidence line no matter how crisply
    its label was read, and averaging would hide exactly the case a reviewer
    needs to see.
    """
    texts = [detections[index].text.strip() for index in indices]
    text = " ".join(part for part in texts if part)
    box = BoundingBox.union(boxes[index] for index in indices)
    confidence = min(detections[index].confidence for index in indices)

    words: list[OcrWord] = []
    for index in indices:
        # Each fragment gets its own box apportioned across its own tokens,
        # so a merged line's words stay tied to the geometry they came from
        # instead of being smeared across the whole merged extent.
        words.extend(
            split_line_into_words(
                detections[index].text,
                boxes[index],
                detections[index].confidence,
            )
        )
    return text, box, confidence, words


def recognize_page(
    image: np.ndarray,
    page_number: int = 1,
    min_confidence: float | None = None,
    preprocessing_applied: list[str] | None = None,
    rotation_applied: float = 0.0,
) -> OcrPage:
    """OCR one page image into structured lines.

    `image` must already be in the state the caller wants recognised — the
    returned coordinates are in this image's pixel space, so preprocessing
    that changes geometry (rotation, cropping, rescaling) has to happen
    before this call, and is recorded on the page for traceability.
    """
    threshold = settings.ocr_min_confidence if min_confidence is None else min_confidence
    height, width = (image.shape[0], image.shape[1]) if image is not None and image.size else (0, 0)

    detections = engine.recognize(image, language=settings.ocr_language)

    kept = [detection for detection in detections if detection.confidence >= threshold]
    dropped = len(detections) - len(kept)
    if dropped:
        logger.info("ocr.low_confidence_dropped page=%s count=%d threshold=%.2f", page_number, dropped, threshold)

    lines: list[OcrLine] = []
    for line_index, (text, box, confidence, words) in enumerate(_merge_fragments(kept)):
        if not text.strip():
            continue
        lines.append(
            OcrLine(
                text=text,
                box=box,
                confidence=confidence,
                polygon=[(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)],
                words=words,
                page_number=page_number,
                line_index=line_index,
            )
        )

    page = OcrPage(
        page_number=page_number,
        width=float(width),
        height=float(height),
        lines=lines,
        source=SOURCE_OCR,
        rotation_applied=rotation_applied,
        preprocessing_applied=list(preprocessing_applied or []),
    )
    logger.info(
        "ocr.page_completed page=%s lines=%d characters=%d mean_confidence=%.3f",
        page_number, len(page.lines), page.character_count, page.mean_confidence,
    )
    return page


def recognize_image_bytes(raw: bytes, page_number: int = 1, **kwargs) -> OcrPage:
    """Decode and OCR a single raster image held in memory."""
    import cv2

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OcrPageFailed("Unable to decode image bytes", page_number=page_number)
    return recognize_page(image, page_number=page_number, **kwargs)


def recognize_pages(
    images: list[np.ndarray],
    first_page_number: int = 1,
    on_page: Callable[[int, int], None] | None = None,
    **kwargs,
) -> OcrDocument:
    """OCR a sequence of page images.

    `on_page(completed, total)` is called after each page. OCR is by far the
    slowest stage — roughly two minutes per page on this CPU build — so a
    caller that reports only "OCR finished" leaves a multi-page invoice
    looking hung for the entire wait.

    A page that fails is recorded as an empty page rather than aborting the
    document: one unreadable scan in a ten-page invoice should not cost the
    other nine.
    """
    pages: list[OcrPage] = []
    total = len(images)
    for offset, image in enumerate(images):
        page_number = first_page_number + offset
        try:
            pages.append(recognize_page(image, page_number=page_number, **kwargs))
        except OcrEngineUnavailable:
            raise  # operational: every remaining page fails identically
        except OcrPageFailed as exc:
            logger.warning("ocr.page_failed page=%s error=%s", page_number, exc)
            height, width = (image.shape[0], image.shape[1]) if image is not None and image.size else (0, 0)
            pages.append(OcrPage(page_number=page_number, width=float(width), height=float(height), lines=[]))
        if on_page is not None:
            on_page(offset + 1, total)
    return OcrDocument(pages=pages)


def image_fingerprint(image: np.ndarray) -> str:
    """Content hash of a page image, for caching and for de-duplicating
    repeated OCR of the same page across reprocessing runs."""
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def is_available() -> bool:
    return engine.is_available(settings.ocr_language)
