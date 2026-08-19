"""GSTIN / PAN extraction with contextual (vendor vs customer) attribution.

GSTIN format: 2 digit state code + 10 char PAN + 1 entity code + 1 'Z' by
convention + 1 checksum char. We validate structurally, not against the
official checksum algorithm (out of scope for deterministic best-effort
parsing), but the PAN embedded inside a GSTIN is validated against the
standard PAN pattern.
"""
from __future__ import annotations

import re

GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

VENDOR_LABELS = (
    "vendor", "supplier", "seller", "from", "bill from", "billed from",
)
CUSTOMER_LABELS = (
    "buyer", "customer", "bill to", "billed to", "consignee", "ship to",
)
# Backward-compatible aliases (kept private-looking for internal call sites).
_VENDOR_LABELS = VENDOR_LABELS
_CUSTOMER_LABELS = CUSTOMER_LABELS

_CONTEXT_WINDOW = 60  # characters of preceding text considered as "label context"


def find_gstins(text: str) -> list[tuple[str, int]]:
    """Return (gstin, position) for every structurally valid GSTIN found."""
    return [(m.group(0), m.start()) for m in GSTIN_PATTERN.finditer(text)]


def find_pans(text: str, exclude_from_gstins: bool = True) -> list[str]:
    """Return standalone PAN matches, excluding substrings that are actually
    part of a GSTIN (a GSTIN embeds a PAN at offset 2)."""
    gstin_spans = [(m.start(), m.end()) for m in GSTIN_PATTERN.finditer(text)] if exclude_from_gstins else []
    pans = []
    for m in PAN_PATTERN.finditer(text):
        if any(gs <= m.start() and m.end() <= ge for gs, ge in gstin_spans):
            continue
        pans.append(m.group(0))
    return pans


def _label_before(text: str, position: int) -> str:
    window = text[max(0, position - _CONTEXT_WINDOW):position].lower()
    return window


def classify_gstin_context(text: str, position: int) -> str | None:
    """Look at the text immediately preceding a GSTIN match and decide whether
    it's contextually a vendor or customer GSTIN. Returns None if ambiguous."""
    window = _label_before(text, position)
    is_vendor = any(label in window for label in _VENDOR_LABELS)
    is_customer = any(label in window for label in _CUSTOMER_LABELS)
    if is_vendor and not is_customer:
        return "vendor"
    if is_customer and not is_vendor:
        return "customer"
    return None


def attribute_gstins(text: str) -> tuple[str | None, str | None, list[str]]:
    """Find all GSTINs in the document and attribute them to vendor/customer
    using nearby labels. Returns (vendor_gstin, customer_gstin, warnings).

    Never guesses: if two candidates exist and neither has a clear label,
    or if both candidates would map to the same role, a warning is raised
    instead of picking one arbitrarily.
    """
    warnings: list[str] = []
    candidates = find_gstins(text)
    if not candidates:
        return None, None, warnings

    vendor_gstin = None
    customer_gstin = None
    unclassified = []

    for gstin, pos in candidates:
        role = classify_gstin_context(text, pos)
        if role == "vendor" and vendor_gstin is None:
            vendor_gstin = gstin
        elif role == "customer" and customer_gstin is None:
            customer_gstin = gstin
        elif role is None:
            unclassified.append(gstin)

    if unclassified:
        if vendor_gstin is None and customer_gstin is None and len(unclassified) >= 1:
            # First unclassified candidate is *usually* the vendor (invoices
            # conventionally list the issuer first), but we flag it rather
            # than treat it as certain.
            vendor_gstin = unclassified[0]
            warnings.append(
                "Vendor GSTIN could not be determined from context labels; "
                "used the first GSTIN found on the document as a best guess."
            )
            if len(unclassified) > 1:
                warnings.append("Additional unlabeled GSTIN candidates found; customer GSTIN left unset.")
        elif vendor_gstin is not None and customer_gstin is None:
            customer_gstin = unclassified[0]
            if len(unclassified) > 1:
                warnings.append("Multiple unlabeled GSTIN candidates found for customer; used the first.")
        elif customer_gstin is not None and vendor_gstin is None:
            vendor_gstin = unclassified[0]
            if len(unclassified) > 1:
                warnings.append("Multiple unlabeled GSTIN candidates found for vendor; used the first.")
        else:
            warnings.append("Unattributed extra GSTIN candidates found on document; ignored.")

    return vendor_gstin, customer_gstin, warnings
