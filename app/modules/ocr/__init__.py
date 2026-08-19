"""Structured OCR: pixels in, positioned text out.

Public surface is `service` (recognise pages), `dto` (the shapes it returns)
and `geometry` (how to reason about where things are). `engine` is internal —
it is the only module that imports paddle, and keeping it private is what
makes the engine replaceable.
"""
from .dto import OcrDocument, OcrLine, OcrPage, OcrWord
from .exceptions import OcrEngineUnavailable, OcrError, OcrPageFailed
from .geometry import BoundingBox

__all__ = [
    "BoundingBox",
    "OcrDocument",
    "OcrEngineUnavailable",
    "OcrError",
    "OcrLine",
    "OcrPage",
    "OcrPageFailed",
    "OcrWord",
]
