"""Failure modes of the OCR stage.

These are distinct because they demand different responses. An unavailable
engine is an operational problem — the deployment is broken and every
document will fail the same way, so it must be loud. An unreadable page is a
data problem — this one file is bad, the service is fine, and the pipeline
should record the failure against the document and move on.
"""
from __future__ import annotations


class OcrError(Exception):
    """Base class for everything this module raises."""


class OcrEngineUnavailable(OcrError):
    """PaddleOCR could not be imported or initialised.

    Operational, not per-document: retrying the same page will fail
    identically until the deployment is fixed, so callers should surface this
    rather than swallow it into a per-document warning.
    """


class OcrPageFailed(OcrError):
    """A single page could not be recognised.

    Per-document: other pages of the same file, and other files, may still
    process normally.
    """

    def __init__(self, message: str, page_number: int | None = None) -> None:
        super().__init__(message)
        self.page_number = page_number
