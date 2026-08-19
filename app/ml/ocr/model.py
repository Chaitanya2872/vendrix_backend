"""The trainable field classifier and its persisted artifact.

A linear model over the sparse feature union from features.py. The choice is
deliberate rather than a placeholder:

  * The signal is overwhelmingly lexical and shape-based, and it is close to
    linearly separable in n-gram space. A gradient-boosted or neural model
    buys little on this feature set and costs a heavyweight dependency in a
    service whose other jobs are I/O bound.
  * Logistic regression emits calibrated per-class probabilities. That number
    is not decoration: it becomes `parsing_confidence`, which decides whether
    an invoice goes straight through or is queued for human review. A model
    that could only emit a hard label would force that decision to be made by
    something less informed.
  * Its weights are inspectable. When a field starts being mis-extracted in
    production, `top_features_for` shows which n-grams drove the call.

Class imbalance is severe by construction — most lines of most invoices are
OTHER — so the loss is class-weighted. Without that, predicting OTHER for
everything scores ~85% accuracy and finds nothing.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import LineContext, build_featureuniser

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 2


@dataclass
class ModelMetadata:
    """Everything needed to interpret a prediction six months from now."""

    format_version: int = MODEL_FORMAT_VERSION
    trained_at: str = ""
    document_count: int = 0
    line_count: int = 0
    formats: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class FieldLineModel:
    """Classifies each line of a document as the invoice field it carries."""

    def __init__(self, C: float = 4.0, max_iter: int = 2000, min_df: int = 2):
        self.pipeline = Pipeline([
            ("features", build_featureuniser(min_df=min_df)),
            ("classifier", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])
        self.metadata = ModelMetadata(hyperparameters={"C": C, "max_iter": max_iter, "min_df": min_df})

    # ── training ────────────────────────────────────────────────────────────

    def fit(self, contexts: list[LineContext], labels: list[str]) -> "FieldLineModel":
        if not contexts:
            raise ValueError("Cannot train on an empty corpus.")
        self.pipeline.fit(contexts, labels)
        self.metadata.trained_at = datetime.now(timezone.utc).isoformat()
        self.metadata.line_count = len(contexts)
        self.metadata.labels = list(self.pipeline.named_steps["classifier"].classes_)
        return self

    # ── inference ───────────────────────────────────────────────────────────

    @property
    def classes(self) -> np.ndarray:
        return self.pipeline.named_steps["classifier"].classes_

    def predict(self, contexts: list[LineContext]) -> list[str]:
        if not contexts:
            return []
        return list(self.pipeline.predict(contexts))

    def predict_proba(self, contexts: list[LineContext]) -> np.ndarray:
        if not contexts:
            return np.empty((0, len(self.classes)))
        return self.pipeline.predict_proba(contexts)

    def predict_with_confidence(self, contexts: list[LineContext]) -> list[tuple[str, float]]:
        """Label plus the probability assigned to it — the form the field
        extractor consumes, since it has to choose between several lines
        claiming the same field."""
        probabilities = self.predict_proba(contexts)
        classes = self.classes
        return [(classes[row.argmax()], float(row.max())) for row in probabilities]

    def probability_of(self, contexts: list[LineContext], label: str) -> np.ndarray:
        """Per-line probability of one specific label. Used when a field is
        missing from the top-1 predictions but the caller still wants the best
        available candidate rather than nothing."""
        classes = list(self.classes)
        if label not in classes:
            return np.zeros(len(contexts))
        return self.predict_proba(contexts)[:, classes.index(label)]

    # ── inspection ──────────────────────────────────────────────────────────

    def top_features_for(self, label: str, count: int = 15) -> list[tuple[str, float]]:
        """The n-grams and shape features pushing hardest towards a label.
        The reason to keep a linear model: when extraction goes wrong in
        production this answers "why" directly."""
        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        if label not in classes:
            return []
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        weights = classifier.coef_[classes.index(label)]
        order = np.argsort(weights)[::-1][:count]
        return [(str(names[index]), float(weights[index])) for index in order]

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": asdict(self.metadata)}, path, compress=3)
        logger.info("ocr_model.saved path=%s lines=%d", path, self.metadata.line_count)
        return path

    @classmethod
    def load(cls, path: Path) -> "FieldLineModel":
        import joblib

        payload = joblib.load(Path(path))
        stored_version = payload.get("metadata", {}).get("format_version")
        if stored_version != MODEL_FORMAT_VERSION:
            # Loading a pipeline whose feature layout no longer matches the
            # code would not error — it would silently produce nonsense.
            raise ValueError(
                f"Model artifact format version {stored_version} does not match "
                f"the expected {MODEL_FORMAT_VERSION}; retrain the model."
            )
        model = cls.__new__(cls)
        model.pipeline = payload["pipeline"]
        model.metadata = ModelMetadata(**payload["metadata"])
        return model
