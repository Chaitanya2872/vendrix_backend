"""Deterministic helpers for turning invoice-style money strings into Decimal.

Handles Indian lakh/crore grouping (₹1,25,000.00), currency prefixes
(INR / Rs. / ₹), and thousands separators. Never uses float for money.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_CURRENCY_PREFIXES = re.compile(
    r"^\s*(?:₹|Rs\.?|INR|USD|\$)\s*", re.IGNORECASE
)
_TRAILING_CURRENCY = re.compile(r"\s*(?:INR|Rs\.?|₹)\s*$", re.IGNORECASE)
# A bare number after stripping currency markers and separators.
_NUMERIC_TAIL = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def clean_amount_string(raw: str) -> str | None:
    """Strip currency symbols/words and separators, keep sign and decimals."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    text = _CURRENCY_PREFIXES.sub("", text)
    text = _TRAILING_CURRENCY.sub("", text)
    text = text.replace(",", "").strip()
    # Parenthesised amounts sometimes denote negative values, e.g. (500.00)
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    match = _NUMERIC_TAIL.search(text)
    if not match:
        return None
    value = match.group(0)
    return f"-{value}" if negative and not value.startswith("-") else value


def parse_amount(raw: str) -> Decimal | None:
    """Parse a monetary string (any of the supported invoice formats) to Decimal.

    Returns None rather than raising when the string isn't a usable number —
    callers are expected to treat that as "not found" and add a warning.
    """
    cleaned = clean_amount_string(raw)
    if cleaned is None:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def find_first_amount(text: str) -> Decimal | None:
    """Convenience: find and parse the first money-like token in free text."""
    match = re.search(
        r"(?:₹|Rs\.?|INR)?\s*-?\d[\d,]*(?:\.\d+)?", text, re.IGNORECASE
    )
    if not match:
        return None
    return parse_amount(match.group(0))


def approx_equal(a: Decimal | None, b: Decimal | None, tolerance: Decimal = Decimal("1.00")) -> bool:
    """Compare two monetary Decimals within a configurable rounding tolerance."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance
