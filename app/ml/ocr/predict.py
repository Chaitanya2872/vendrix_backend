"""Inference entry point: extracted document text in, field values out.

Loading is lazy and cached. The artifact is a few megabytes of sparse
coefficients and takes long enough to deserialise that doing it per document
would dominate the parse; doing it at import time would slow every process
that touches the invoices package, including ones that never parse anything.

A missing or unloadable artifact is not an error. `available()` returns False,
callers fall back to deterministic parsing, and the application keeps working —
an untrained checkout must not break invoice upload.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .extract import ExtractedField, confidence_from, extract_fields
from .features import contexts_from_lines
from .model import FieldLineModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "field_model.joblib"

_lock = threading.Lock()
_model: FieldLineModel | None = None
_load_attempted = False


def model_path() -> Path:
    from ...core.config import settings

    configured = getattr(settings, "ocr_field_model_path", None)
    return Path(configured) if configured else DEFAULT_MODEL_PATH


def load_model() -> FieldLineModel | None:
    """Return the cached model, loading it on first use. Returns None when no
    usable artifact exists."""
    global _model, _load_attempted

    if _model is not None or _load_attempted:
        return _model

    with _lock:
        # Re-check inside the lock: two request threads can arrive together.
        if _model is not None or _load_attempted:
            return _model
        _load_attempted = True
        path = model_path()
        if not path.exists():
            logger.info("ocr_model.artifact_absent path=%s falling_back=deterministic", path)
            return None
        try:
            _model = FieldLineModel.load(path)
            logger.info(
                "ocr_model.loaded path=%s trained_at=%s documents=%s",
                path, _model.metadata.trained_at, _model.metadata.document_count,
            )
        except Exception:
            logger.exception("ocr_model.load_failed path=%s falling_back=deterministic", path)
            _model = None
        return _model


def reset_cache() -> None:
    """Drop the cached model. For tests and for reloading after a retrain
    without restarting the process."""
    global _model, _load_attempted
    with _lock:
        _model, _load_attempted = None, False


def available() -> bool:
    return load_model() is not None


def predict_fields(text: str) -> dict[str, ExtractedField]:
    """Field values the model can recover from this document's text.

    Returns {} when no model is available, which callers should read as "no
    opinion" rather than "no fields present".
    """
    model = load_model()
    if model is None or not text.strip():
        return {}
    contexts = contexts_from_lines(text.splitlines())
    if not contexts:
        return {}
    try:
        return extract_fields(model, contexts)
    except Exception:
        # Inference must never take down an upload; the deterministic parser
        # has already produced a result by the time this runs.
        logger.exception("ocr_model.inference_failed lines=%d", len(contexts))
        return {}


def predict_with_confidence(text: str) -> tuple[dict[str, ExtractedField], float]:
    found = predict_fields(text)
    return found, confidence_from(found)
