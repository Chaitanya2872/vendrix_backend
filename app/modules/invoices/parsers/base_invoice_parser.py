"""Parser abstraction. GenericInvoiceParser (in generic_invoice_parser.py) is
the only implementation today; vendor-specific parsers (e.g. for a supplier
whose layout the generic parser handles poorly) can subclass this later and
register themselves in `AVAILABLE_PARSERS` without touching the orchestration
service.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto import ParsedInvoiceResult
from .text_extraction import ExtractedDocument


class BaseInvoiceParser(ABC):
    name: str = "base"
    version: str = "1.0"

    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """Cheap heuristic check: does this document look like something
        this parser knows how to handle? The generic parser always returns
        True; vendor-specific parsers should look for a distinguishing
        marker (e.g. a vendor name or logo text) before claiming a document."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, extracted: ExtractedDocument) -> ParsedInvoiceResult:
        """Parse an already-extracted document (text + tables) into a
        ParsedInvoiceResult. Does not touch the filesystem or the database."""
        raise NotImplementedError
