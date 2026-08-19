"""Label-driven extraction of invoice header fields from raw document text.

Every extractor here works the same way: find a label (one of several known
aliases), then look at the text immediately after it on the same line (or the
next non-empty line) for the value. This is intentionally conservative —
it will not do lookups against a database of vendor formats.
"""
from __future__ import annotations

import re
from datetime import date

from dateutil import parser as dateutil_parser

from .money_utils import parse_amount
from .gst_utils import attribute_gstins, find_pans

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{10}\b")

_INVOICE_NUMBER_LABELS = (
    "tax invoice no", "invoice number", "invoice no", "invoice #",
    "inv no", "inv #", "bill no", "document no",
)
_INVOICE_DATE_LABELS = ("invoice date", "bill date", "date of invoice")
_DUE_DATE_LABELS = ("due date", "payment due date")
_PO_NUMBER_LABELS = ("po number", "po no", "purchase order number", "purchase order no")
_PO_DATE_LABELS = ("po date", "purchase order date")
_PLACE_OF_SUPPLY_LABELS = ("place of supply",)
_PAYMENT_TERMS_LABELS = ("payment terms", "terms of payment")

# label -> DTO field name, ordered so more specific labels are tried first
_AMOUNT_LABELS: list[tuple[str, str]] = [
    ("taxable value", "taxable_amount"),
    ("taxable amount", "taxable_amount"),
    ("sub total", "subtotal"),
    ("subtotal", "subtotal"),
    ("discount", "discount_amount"),
    ("round off", "round_off"),
    ("total tax", "tax_amount"),
    ("tax amount", "tax_amount"),
    ("amount payable", "total_amount"),
    ("net payable", "total_amount"),
    ("grand total", "total_amount"),
    ("invoice total", "total_amount"),
    ("total amount", "total_amount"),
    ("amount paid", "amount_paid"),
    ("amount due", "amount_due"),
    ("balance due", "amount_due"),
]

_CGST_PATTERN = re.compile(r"c\.?gst[^\d\n]{0,15}?(\d+(?:\.\d+)?)\s*%?[^\d\n]{0,15}?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_SGST_PATTERN = re.compile(r"s\.?gst[^\d\n]{0,15}?(\d+(?:\.\d+)?)\s*%?[^\d\n]{0,15}?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_IGST_PATTERN = re.compile(r"i\.?gst[^\d\n]{0,15}?(\d+(?:\.\d+)?)\s*%?[^\d\n]{0,15}?([\d,]+(?:\.\d+)?)", re.IGNORECASE)

_CURRENCY_HINTS = {"₹": "INR", "rs.": "INR", "rs": "INR", "inr": "INR", "$": "USD", "usd": "USD"}

_DATE_TOKEN = re.compile(
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}"
)


def _line_containing(text: str, label: str) -> str | None:
    pattern = re.compile(re.escape(label), re.IGNORECASE)
    for line in text.splitlines():
        if pattern.search(line):
            return line
    return None


def _value_after_label(text: str, label: str) -> str | None:
    """Return the substring on the same line following the label (after any
    ':' / '-' separator), or, if the line only contains the label, the next
    non-empty line."""
    lines = text.splitlines()
    pattern = re.compile(re.escape(label), re.IGNORECASE)
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        remainder = line[match.end():].lstrip(" :\t-#").strip()
        if remainder:
            return remainder
        # value likely on the next line
        for next_line in lines[i + 1:i + 3]:
            if next_line.strip():
                return next_line.strip()
    return None


def extract_invoice_number(text: str) -> str | None:
    for label in _INVOICE_NUMBER_LABELS:
        value = _value_after_label(text, label)
        if value:
            # Take the leading token: letters/digits/-/_// , stop at whitespace
            # that's followed by another label-like word (e.g. "Date:").
            match = re.match(r"[A-Za-z0-9/_\-]+", value)
            if match:
                return match.group(0)
    return None


def _parse_date_token(token: str) -> date | None:
    try:
        parsed = dateutil_parser.parse(token, dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError):
        return None
    return parsed.date()


def _extract_date(text: str, labels: tuple[str, ...]) -> date | None:
    for label in labels:
        value = _value_after_label(text, label)
        if not value:
            continue
        match = _DATE_TOKEN.search(value)
        candidate = match.group(0) if match else value
        parsed = _parse_date_token(candidate)
        if parsed:
            return parsed
    return None


def extract_invoice_date(text: str) -> date | None:
    return _extract_date(text, _INVOICE_DATE_LABELS)


def extract_due_date(text: str) -> date | None:
    return _extract_date(text, _DUE_DATE_LABELS)


def extract_po_number(text: str) -> str | None:
    for label in _PO_NUMBER_LABELS:
        value = _value_after_label(text, label)
        if value:
            match = re.match(r"[A-Za-z0-9/_\-]+", value)
            if match:
                return match.group(0)
    return None


def extract_po_date(text: str) -> date | None:
    return _extract_date(text, _PO_DATE_LABELS)


def extract_place_of_supply(text: str) -> str | None:
    for label in _PLACE_OF_SUPPLY_LABELS:
        value = _value_after_label(text, label)
        if value:
            return re.split(r"\s{2,}|\t", value)[0].strip()
    return None


def extract_payment_terms(text: str) -> str | None:
    for label in _PAYMENT_TERMS_LABELS:
        value = _value_after_label(text, label)
        if value:
            return re.split(r"\s{2,}|\t", value)[0].strip()
    return None


def extract_currency(text: str) -> str | None:
    lowered = text.lower()
    for hint, code in _CURRENCY_HINTS.items():
        if hint in lowered:
            return code
    return None


def extract_header_amounts(text: str) -> dict:
    """Contextual amount extraction: walk labels most-specific-first so that,
    e.g., 'Taxable Value' doesn't get mistaken for a generic 'Total' line, and
    a field already found isn't overwritten by a later, less specific label."""
    found: dict[str, object] = {}
    for label, field_name in _AMOUNT_LABELS:
        if field_name in found:
            continue
        value = _value_after_label(text, label)
        if value is None:
            continue
        amount = parse_amount(value)
        if amount is not None:
            found[field_name] = amount
    return found


def extract_gst_breakdown(text: str) -> dict:
    result: dict[str, object] = {}
    for pattern, key in ((_CGST_PATTERN, "cgst_amount"), (_SGST_PATTERN, "sgst_amount"), (_IGST_PATTERN, "igst_amount")):
        match = pattern.search(text)
        if match:
            amount = parse_amount(match.group(2))
            if amount is not None:
                result[key] = amount
    return result


def _extract_party_block(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        block = _value_after_label(text, label)
        if block:
            return block
    return None


def extract_parties(text: str) -> tuple[dict, dict, list[str]]:
    """Extract vendor and customer name/address/email/phone plus GSTIN/PAN,
    returning (vendor_dict, customer_dict, warnings)."""
    from .gst_utils import VENDOR_LABELS, CUSTOMER_LABELS  # reuse label lists

    vendor_name = _extract_party_block(text, VENDOR_LABELS)
    customer_name = _extract_party_block(text, CUSTOMER_LABELS)

    vendor_gstin, customer_gstin, gstin_warnings = attribute_gstins(text)

    pans = find_pans(text)
    vendor_pan = pans[0] if pans else None
    customer_pan = pans[1] if len(pans) > 1 else None

    emails = EMAIL_PATTERN.findall(text)
    phones = PHONE_PATTERN.findall(text)

    vendor = {
        "name": vendor_name,
        "gstin": vendor_gstin,
        "pan": vendor_pan,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
    }
    customer = {
        "name": customer_name,
        "gstin": customer_gstin,
        "pan": customer_pan,
        "email": emails[1] if len(emails) > 1 else None,
        "phone": phones[1] if len(phones) > 1 else None,
    }
    return vendor, customer, gstin_warnings
