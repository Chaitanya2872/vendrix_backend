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
from .gst_utils import GSTIN_PATTERN, attribute_gstins, find_gstins, find_pans

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
    r"\d{1,2}[/\-.][A-Za-z]{3,9}\.?[/\-.]\d{2,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}"
)


def _line_containing(text: str, label: str) -> str | None:
    pattern = _label_pattern(label)
    for line in text.splitlines():
        if pattern.search(line):
            return line
    return None


def _value_after_label(text: str, label: str) -> str | None:
    """Return the substring on the same line following the label (after any
    ':' / '-' separator), or, if the line only contains the label, the next
    non-empty line."""
    lines = text.splitlines()
    pattern = _label_pattern(label)
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


def _label_pattern(label: str) -> re.Pattern[str]:
    """Match a complete label, not the same letters inside another label."""
    return re.compile(rf"(?<!\w){re.escape(label)}(?![\w'])", re.IGNORECASE)


def _nearby_values(text: str, label: str, distance: int = 4) -> list[str]:
    """Candidate values on the label line and the few lines below it."""
    lines = text.splitlines()
    pattern = _label_pattern(label)
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        values = []
        remainder = line[match.end():].lstrip(" :\t-#.").strip()
        if remainder:
            values.append(remainder)
        values.extend(candidate.strip() for candidate in lines[index + 1:index + 1 + distance]
                      if candidate.strip())
        return values
    return []


def extract_invoice_number(text: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    for label in _INVOICE_NUMBER_LABELS:
        for distance, value in enumerate(_nearby_values(text, label, distance=6)):
            for candidate in re.findall(r"[A-Za-z0-9]+(?:[/_-][A-Za-z0-9]+)+", value):
                if any(char.isalpha() for char in candidate) and any(char.isdigit() for char in candidate):
                    if _DATE_TOKEN.fullmatch(candidate) is not None or len(candidate) > 30:
                        continue
                    lowered = candidate.lower()
                    if lowered.startswith(("hyderabad-", "telangana-", "gujarat-")):
                        continue
                    score = candidate.count("/") * 4 + candidate.count("-") - distance
                    candidates.append((score, candidate))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


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
    for label in (*_INVOICE_DATE_LABELS, "dated"):
        for value in _nearby_values(text, label, distance=6):
            for match in _DATE_TOKEN.finditer(value):
                parsed = _parse_date_token(match.group(0))
                if parsed:
                    return parsed
    return None


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


_MONEY_TOKEN = re.compile(r"(?:\(-\)|[-−])?\s*(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?")


def _amounts_on_line(line: str) -> list:
    amounts = []
    for match in _MONEY_TOKEN.finditer(line):
        token = match.group(0).strip().replace("(-)", "-").replace("−", "-")
        # Percentages are rates, not money.
        if line[match.end():].lstrip().startswith("%"):
            continue
        if re.fullmatch(r"\d{8,}", token):
            continue  # HSN/SAC, GSTIN fragments and acknowledgement numbers
        value = parse_amount(token)
        if value is not None:
            amounts.append(value)
    return amounts


def extract_tally_summary(text: str) -> dict:
    """Read a Tally-style summary whose values may precede their labels."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def near(index: int, radius: int = 2) -> list:
        values = []
        for candidate in lines[max(0, index - radius):index + radius + 1]:
            values.extend(_amounts_on_line(candidate))
        return values

    found: dict[str, object] = {}
    total_candidates = []
    for index, line in enumerate(lines):
        normalized = normalise_label = re.sub(r"[^a-z]+", " ", line.lower()).strip()
        if normalized == "total" or normalized.startswith("total nos") or normalized.startswith("total rs"):
            total_candidates.extend(near(index))
    if total_candidates:
        found["total_amount"] = max(total_candidates)

    for field, token in (("cgst_amount", "cgst"), ("sgst_amount", "sgst"), ("igst_amount", "igst")):
        candidates: list[tuple[int, int, object]] = []
        for index, line in enumerate(lines):
            words = re.sub(r"[^a-z]+", " ", line.lower()).split()
            if token in words and "taxable" not in words:
                for candidate_index in range(max(0, index - 2), min(len(lines), index + 3)):
                    for value in _amounts_on_line(lines[candidate_index]):
                        if value > 1:
                            direction = 0 if candidate_index > index else 1
                            candidates.append((abs(candidate_index - index), direction, value))
        if candidates:
            same_line = [value for distance, _, value in candidates if distance == 0]
            immediate_after = [value for distance, direction, value in candidates
                               if distance == 1 and direction == 0]
            immediate_before = [value for distance, direction, value in candidates
                                if distance == 1 and direction == 1]
            if same_line:
                found[field] = max(same_line)
            elif immediate_after:
                found[field] = max(immediate_after)
            elif immediate_before:
                found[field] = max(immediate_before)
            else:
                closest = min(distance for distance, _, _ in candidates)
                found[field] = max(value for distance, _, value in candidates if distance == closest)

    round_candidates = []
    for index, line in enumerate(lines):
        if "round off" in line.lower() or "roundoff" in line.lower():
            round_candidates.extend(value for value in near(index, 1) if abs(value) < 1)
    if round_candidates:
        found["round_off"] = min(round_candidates, key=abs)

    components = [found[name] for name in ("cgst_amount", "sgst_amount", "igst_amount") if name in found]
    if components:
        found["tax_amount"] = sum(components)
    if "total_amount" in found and "tax_amount" in found:
        found["subtotal"] = found["total_amount"] - found["tax_amount"] - found.get("round_off", 0)
        found["taxable_amount"] = found["subtotal"]
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


_COMPANY_NAME = re.compile(
    r"\b(?:p(?:vt|rivate)\.?\s+l(?:td|imited)\.?|llp|limited|technologies|"
    r"innovations|industries|enterprises|corporation|company|co\.?)\b",
    re.IGNORECASE,
)


def _company_names_before_gstins(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    names: list[str] = []
    for index, line in enumerate(lines):
        if not GSTIN_PATTERN.search(line.upper()):
            continue
        window = lines[max(0, index - 15):index]
        name = next((candidate for candidate in reversed(window)
                     if _COMPANY_NAME.search(candidate) and not GSTIN_PATTERN.search(candidate.upper())), None)
        if name is None:
            name = next((candidate for candidate in window
                         if candidate == candidate.upper()
                         and len(re.findall(r"[A-Z]{2,}", candidate)) >= 2
                         and not any(label in candidate.lower() for label in (
                             "tax invoice", "invoice no", "delivery note", "state name",
                         ))), None)
        if name and name not in names:
            names.append(name)
    return names


def _looks_like_party_name(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return bool(_COMPANY_NAME.search(value)) and not any(token in lowered for token in (
        "order no", "reference no", "invoice no", "dated", "delivery note",
    ))


_LEGAL_NAME_END = re.compile(
    r"^(.+?\b(?:LLP|PVT\.?\s+L.?D\.?|PRIVATE\s+LIMITED|LIMITED))\b",
    re.IGNORECASE,
)


def _clean_party_name(value: str | None) -> str | None:
    if not value:
        return value
    cleaned = re.sub(r"^\s*(?:buyer\s*)?(?:\(bill to\)\s*)+", "", value,
                     flags=re.IGNORECASE).strip(" :-")
    cleaned = re.sub(r"\s*-?\s*\(\d{2}-\d{2}\)\s*$", "", cleaned).strip()
    match = _LEGAL_NAME_END.search(cleaned)
    company = match.group(1).strip() if match else cleaned
    return re.sub(r"\bPVT\.?\s+L.?D\.?\b", "PVT LTD", company, flags=re.IGNORECASE)


def extract_parties(text: str) -> tuple[dict, dict, list[str]]:
    """Extract vendor and customer name/address/email/phone plus GSTIN/PAN,
    returning (vendor_dict, customer_dict, warnings)."""
    from .gst_utils import VENDOR_LABELS, CUSTOMER_LABELS  # reuse label lists

    vendor_name = _extract_party_block(text, VENDOR_LABELS)
    customer_name = _extract_party_block(text, CUSTOMER_LABELS)

    nearby_company_names = _company_names_before_gstins(text)
    if not _looks_like_party_name(vendor_name) and nearby_company_names:
        vendor_name = nearby_company_names[0]
    if not _looks_like_party_name(customer_name) and len(nearby_company_names) > 1:
        customer_name = nearby_company_names[1]

    vendor_name = _clean_party_name(vendor_name)
    customer_name = _clean_party_name(customer_name)

    vendor_gstin, customer_gstin, gstin_warnings = attribute_gstins(text)

    # Once both named party blocks are present, document order is stronger
    # evidence than a character window that can cross into the next block.
    # Supplier GSTIN is conventionally first, customer second; later unique
    # values are ship-from/warehouse registrations.
    ordered_gstins = list(dict.fromkeys(value for value, _ in find_gstins(text)))
    if vendor_name and customer_name and len(ordered_gstins) >= 2:
        vendor_gstin, customer_gstin = ordered_gstins[:2]
        gstin_warnings = []

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
    if vendor["name"] and vendor["gstin"] and customer["name"] and customer["gstin"]:
        gstin_warnings = [warning for warning in gstin_warnings if not warning.startswith((
            "Vendor GSTIN could not be determined",
            "GSTIN roles could not be determined",
        ))]
    return vendor, customer, gstin_warnings
