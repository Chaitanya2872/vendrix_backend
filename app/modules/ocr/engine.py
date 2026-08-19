"""PaddleOCR adapter: raw pixels in, text + polygons + confidences out.

This is the only module in the project that imports paddleocr. Everything
else talks to `service.py`, so swapping the engine — or running without one
in a test — touches one file.

Two things here are load-bearing and easy to get wrong:

**The oneDNN workaround.** PaddleX enables its oneDNN backend by default and
it crashes on this CPU build during text detection
(`ConvertPirAttribute2RuntimeAttribute NotImplementedError`). The environment
variable must be set *before* paddleocr or paddlex is imported anywhere in
the process, which is why it sits at module import time and why every import
of paddle in this project routes through here.

**The result shape is version-dependent.** PaddleOCR 3.x returns result
objects carrying `rec_texts` / `rec_scores` / `rec_polys`; 2.x returned a
nested list of `[polygon, (text, score)]`. `_normalise_result` accepts both,
because a paddle upgrade that silently changed the shape would otherwise turn
into empty extractions rather than an error.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from .exceptions import OcrEngineUnavailable, OcrPageFailed
from .geometry import Point

logger = logging.getLogger(__name__)

# Must precede any paddleocr/paddlex import in the process. See module docstring.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")


class RawDetection:
    """One detected text line straight from the engine, before any of this
    project's structure is imposed on it."""

    __slots__ = ("text", "polygon", "confidence")

    def __init__(self, text: str, polygon: list[Point], confidence: float) -> None:
        self.text = text
        self.polygon = polygon
        self.confidence = confidence

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RawDetection(text={self.text!r}, confidence={self.confidence:.3f})"


@lru_cache(maxsize=1)
def _engine(language: str = "en") -> Any:
    """The PaddleOCR instance, built once per process.

    Model loading costs seconds and holds hundreds of megabytes; a Celery
    worker that rebuilt it per task would spend most of its life loading. The
    cache is keyed on language so a future multi-language deployment gets one
    instance per language rather than thrashing a single slot.

    Document orientation and unwarping are left off here: they are
    preprocessing decisions this project makes explicitly and measurably in
    the preprocessing service, and having paddle silently apply its own would
    make the geometry it returns refer to an image the caller never sees.
    """
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # ImportError, but paddle also raises OSError on bad installs
        raise OcrEngineUnavailable(f"PaddleOCR could not be imported: {exc}") from exc

    try:
        return PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        raise OcrEngineUnavailable(f"PaddleOCR failed to initialise: {exc}") from exc


def is_available(language: str = "en") -> bool:
    """Whether OCR can run in this process, without raising.

    Used by callers that have a non-OCR path available and want to choose,
    rather than catch. Deliberately builds the engine — an import that
    succeeds but whose model files are missing would otherwise report
    available and fail on the first real page.
    """
    try:
        _engine(language)
        return True
    except OcrEngineUnavailable:
        return False


def _as_points(polygon: Any) -> list[Point]:
    """Coerce a polygon from numpy array / nested list / tuple into plain
    float pairs. Downstream code is stored, serialised and compared; numpy
    scalars leak into JSON encoders and equality checks in ways that only
    show up at the API boundary."""
    if polygon is None:
        return []
    if isinstance(polygon, np.ndarray):
        polygon = polygon.tolist()
    points: list[Point] = []
    for point in polygon:
        if isinstance(point, np.ndarray):
            point = point.tolist()
        if len(point) < 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return points


def _payload_of(page: Any) -> dict | None:
    """Pull the result mapping out of whatever PaddleOCR handed back.

    3.x result objects are dict-like *and* expose `.json`; preferring the
    mapping avoids paddle's JSON coercion pass, which is pure cost when we
    immediately convert to floats ourselves.
    """
    try:
        if "rec_texts" in page:
            return page  # type: ignore[return-value]
    except TypeError:
        pass  # not a mapping; fall through to the .json accessor

    data = getattr(page, "json", None)
    if isinstance(data, dict):
        return data.get("res", data)
    if isinstance(page, dict):
        return page.get("res", page)
    return None


def _normalise_result(result: Iterable[Any]) -> list[RawDetection]:
    """Flatten an engine result into `RawDetection`s, accepting both the 3.x
    and 2.x output shapes."""
    detections: list[RawDetection] = []

    for page in result:
        payload = _payload_of(page)

        if payload is not None and "rec_texts" in payload:
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            # rec_polys is the recognised subset of dt_polys, so it aligns
            # index-for-index with rec_texts; dt_polys does not once a
            # detection is dropped by the recogniser.
            polygons = list(payload.get("rec_polys") or [])
            if not polygons:
                polygons = list(payload.get("dt_polys") or [])

            for index, text in enumerate(texts):
                polygon = _as_points(polygons[index]) if index < len(polygons) else []
                if not polygon:
                    # A detection without geometry is unusable downstream and
                    # silently dropping it would look like an OCR miss.
                    logger.warning("ocr.detection_without_polygon text=%r", text[:40])
                    continue
                confidence = float(scores[index]) if index < len(scores) else 0.0
                detections.append(RawDetection(str(text), polygon, confidence))
            continue

        # PaddleOCR 2.x: [[polygon, (text, score)], ...]
        if isinstance(page, (list, tuple)):
            for entry in page:
                try:
                    polygon_raw, recognition = entry[0], entry[1]
                    text, score = recognition[0], recognition[1]
                except (IndexError, TypeError, KeyError):
                    continue
                polygon = _as_points(polygon_raw)
                if polygon:
                    detections.append(RawDetection(str(text), polygon, float(score)))
            continue

        logger.warning("ocr.unrecognised_result_shape type=%s", type(page).__name__)

    return detections


def recognize(image: np.ndarray, language: str = "en") -> list[RawDetection]:
    """Run OCR over one already-preprocessed page image.

    Preprocessing is the caller's job (see image_preprocessing_service): this
    function must stay a thin, predictable wrapper so that what the engine saw
    is exactly what the returned coordinates refer to.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise OcrPageFailed("Cannot run OCR on an empty image")

    engine = _engine(language)
    try:
        result = engine.predict(image)
    except Exception as exc:
        raise OcrPageFailed(f"PaddleOCR failed on this page: {exc}") from exc

    detections = _normalise_result(result)
    logger.debug("ocr.page_recognised detections=%d", len(detections))
    return detections
