"""OCR adapter: raw pixels in, text + polygons + confidences out.

This is the only module in the project that imports an OCR engine. Everything
else talks to `service.py` in terms of `RawDetection`, so swapping the engine
— or running without one in a test — touches one file. That was the point of
the seam, and it is what made the backend switch below a contained change.

**Two backends, same models.** Both run the PP-OCR detection and recognition
networks; they differ only in what executes the graph.

  `onnx`   — ONNX Runtime, via the `rapidocr` package. The default.
  `paddle` — PaddleOCR/PaddlePaddle. Kept as a fallback.

The default is ONNX because of a measured factor of seven. On the reference
CPU box, the same 71-line invoice page takes ~38s under paddle and ~5s under
ONNX Runtime, with identical line counts and higher mean confidence. The cost
under paddle is per-inference overhead — ~450ms for a single text line — and
nothing reachable from `PaddleOCR(...)` moves it: batching made it worse,
oneDNN crashes the detector on this build, and halving the input resolution
saved a tenth. See docs/ocr-performance.md for the full table.

Engine construction is also two orders cheaper (~1s against ~15-35s), which
matters because it is paid once per process and used to be paid inside the
first upload.

**The paddle oneDNN workaround.** PaddleX enables its oneDNN backend by
default and it crashes on this CPU build during text detection
(`ConvertPirAttribute2RuntimeAttribute NotImplementedError`). The environment
variable must be set *before* paddleocr or paddlex is imported anywhere in
the process, which is why it sits at module import time and why every import
of paddle in this project routes through here. It is set unconditionally
rather than only on the paddle path: it must land before the import, and by
the time a caller has chosen a backend it is too late.

**The paddle result shape is version-dependent.** PaddleOCR 3.x returns
result objects carrying `rec_texts` / `rec_scores` / `rec_polys`; 2.x
returned a nested list of `[polygon, (text, score)]`. `_normalise_paddle`
accepts both, because a paddle upgrade that silently changed the shape would
otherwise turn into empty extractions rather than an error.
"""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from .exceptions import OcrEngineUnavailable, OcrPageFailed
from .geometry import Point

logger = logging.getLogger(__name__)

# Must precede any paddleocr/paddlex import in the process. See module docstring.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
# PaddleX pings its model hosts on first construction to see whether a newer
# artifact exists. On a warm cache that check buys nothing and costs seconds
# of the first extraction; on a host with no outbound access it costs the
# connection timeout. The models are pinned by name, so staleness is a
# deployment decision, not something to discover at request time.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


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
def _paddle_engine(language: str = "en") -> Any:
    """The PaddleOCR instance, built once per process.

    Model loading costs seconds and holds hundreds of megabytes; a Celery
    worker that rebuilt it per task would spend most of its life loading. The
    cache is keyed on language so a future multi-language deployment gets one
    instance per language rather than thrashing a single slot.

    Document orientation and unwarping are left off here: they are
    preprocessing decisions this project makes explicitly and measurably in
    the preprocessing service, and having paddle silently apply its own would
    make the geometry it returns refer to an image the caller never sees.

    Model choice and batch size come from settings and are the two knobs that
    decide whether a page takes seconds or minutes on this CPU build. Both
    are measured in docs/ocr-performance.md rather than guessed.
    """
    from app.core.config import settings

    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # ImportError, but paddle also raises OSError on bad installs
        raise OcrEngineUnavailable(f"PaddleOCR could not be imported: {exc}") from exc

    options: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        # The single largest win on this build. The recogniser is invoked once
        # per detected text line — around 70 on a dense invoice — and running
        # them one at a time dominates the page. See docs/ocr-performance.md.
        "text_recognition_batch_size": settings.ocr_recognition_batch_size,
        "cpu_threads": settings.ocr_cpu_threads,
    }
    # Named models bypass paddle's `lang` defaulting, and paddle warns when
    # both are given, so only one of the two is ever passed.
    if settings.ocr_detection_model and settings.ocr_recognition_model:
        options["text_detection_model_name"] = settings.ocr_detection_model
        options["text_recognition_model_name"] = settings.ocr_recognition_model
    else:
        options["lang"] = language

    try:
        engine = PaddleOCR(**options)
    except Exception as exc:
        raise OcrEngineUnavailable(f"PaddleOCR failed to initialise: {exc}") from exc

    logger.info(
        "ocr.engine_ready detection=%s recognition=%s batch=%s threads=%s",
        options.get("text_detection_model_name", language),
        options.get("text_recognition_model_name", language),
        settings.ocr_recognition_batch_size,
        settings.ocr_cpu_threads,
    )
    return engine


@lru_cache(maxsize=1)
def _onnx_engine(language: str = "en") -> Any:
    """The RapidOCR (ONNX Runtime) instance, built once per process.

    `language` is accepted for signature parity with the paddle builder but
    is not used to select a model: the packaged PP-OCRv6 detection and
    recognition models are jointly trained on Latin script and Chinese, and
    read English invoices at ~0.99 mean confidence. Naming a per-language
    model here would be inventing a distinction the shipped artifacts do not
    make. `ocr_language` continues to select the model on the paddle path.

    Thread count is set through ONNX Runtime's own session options rather
    than a constructor argument, which is why it goes in as config keys.
    """
    from app.core.config import settings

    try:
        from rapidocr import RapidOCR
    except Exception as exc:  # ImportError, plus loader errors on bad installs
        raise OcrEngineUnavailable(
            f"rapidocr (ONNX backend) could not be imported: {exc}"
        ) from exc

    threads = settings.ocr_cpu_threads
    try:
        engine = RapidOCR(params={
            "Det.engine_cfg.onnxruntime.intra_op_num_threads": threads,
            "Rec.engine_cfg.onnxruntime.intra_op_num_threads": threads,
        })
    except Exception as exc:
        raise OcrEngineUnavailable(f"RapidOCR failed to initialise: {exc}") from exc

    logger.info("ocr.engine_ready backend=onnx threads=%s", threads)
    return engine


def _normalise_rapidocr(result: Any) -> list[RawDetection]:
    """Flatten a RapidOCR result into `RawDetection`s.

    `boxes` is an (N, 4, 2) array of quadrilaterals aligned index-for-index
    with `txts` and `scores`. An empty page yields None rather than empty
    arrays, which is why this checks before zipping.
    """
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None:
        return []

    detections: list[RawDetection] = []
    for index, text in enumerate(texts):
        polygon = _as_points(boxes[index]) if index < len(boxes) else []
        if not polygon:
            # A detection without geometry is unusable downstream, and
            # dropping it silently would look like an OCR miss.
            logger.warning("ocr.detection_without_polygon text=%r", str(text)[:40])
            continue
        confidence = float(scores[index]) if scores is not None and index < len(scores) else 0.0
        detections.append(RawDetection(str(text), polygon, confidence))
    return detections


def _engine(language: str = "en") -> Any:
    """The configured backend's engine, built once per process.

    Falls back to paddle when the ONNX backend is selected but `rapidocr` is
    not installed, so a deployment that has not yet picked up the new
    dependency keeps working — slowly, but working, which beats every
    document upload failing.
    """
    from app.core.config import settings

    if settings.ocr_backend == "paddle":
        return _paddle_engine(language)
    try:
        return _onnx_engine(language)
    except OcrEngineUnavailable as exc:
        logger.warning("ocr.onnx_unavailable falling_back_to_paddle error=%s", exc)
        return _paddle_engine(language)


def warm_up(language: str | None = None) -> bool:
    """Build the engine now so the first real document does not pay for it.

    Construction reads the model files and lays out the runtime's arenas —
    about a second under ONNX Runtime, and 15-35s under paddle. Either way it
    is a fixed cost per process, and paying it inside the first upload is what
    made a document that parses in seconds appear to take a minute. Called
    from application startup on a background thread; safe to call more than
    once, since the `lru_cache` makes every call after the first free.

    Returns whether OCR is usable, and never raises: a deployment with no
    OCR models should still serve every non-OCR route.
    """
    from app.core.config import settings

    started = time.monotonic()
    try:
        _engine(language or settings.ocr_language)
    except OcrEngineUnavailable as exc:
        logger.warning("ocr.warm_up_failed error=%s", exc)
        return False
    logger.info("ocr.warm_up_completed seconds=%.1f", time.monotonic() - started)
    return True


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


def _normalise_paddle(result: Iterable[Any]) -> list[RawDetection]:
    """Flatten a PaddleOCR result into `RawDetection`s, accepting both the
    3.x and 2.x output shapes."""
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
    from app.core.config import settings

    if image is None or getattr(image, "size", 0) == 0:
        raise OcrPageFailed("Cannot run OCR on an empty image")

    engine = _engine(language)

    # Which normaliser applies is decided by what was actually built, not by
    # what was configured: `_engine` falls back to paddle when the ONNX
    # backend is selected but unavailable, and reading the configured name
    # here would then parse a paddle result with the ONNX normaliser and
    # return nothing at all.
    if type(engine).__name__ == "RapidOCR":
        try:
            # Text-line orientation classification is off for the same reason
            # it is off on the paddle path: rotation is a preprocessing
            # decision this project makes explicitly and measurably.
            # `text_score` is passed so the project's own threshold stays
            # authoritative — RapidOCR would otherwise pre-filter at its own
            # default and drop lines `service.recognize_page` meant to keep.
            result = engine(image, use_cls=False, text_score=settings.ocr_min_confidence)
        except Exception as exc:
            raise OcrPageFailed(f"RapidOCR failed on this page: {exc}") from exc
        detections = _normalise_rapidocr(result)
    else:
        try:
            result = engine.predict(image)
        except Exception as exc:
            raise OcrPageFailed(f"PaddleOCR failed on this page: {exc}") from exc
        detections = _normalise_paddle(result)

    logger.debug("ocr.page_recognised detections=%d", len(detections))
    return detections
