"""Turn line predictions into typed field values.

The classifier answers "which field does this line carry". This module answers
"and what is the value", which is a separate problem with a separate failure
mode — the model can point at exactly the right line and still yield nothing
usable if OCR turned the amount into noise. Keeping the two apart is what lets
the evaluation attribute a miss to the model or to the OCR engine.

Two behaviours worth knowing about:

* Candidates are tried in probability order, and a candidate whose value will
  not parse is skipped rather than accepted empty. The second-best line for
  TOTAL_AMOUNT is worth more than the best line with nothing extractable on
  it.
* A value is looked for on the *next* line when the predicted line has none.
  OCR of a right-aligned totals block routinely emits the label and the amount
  as two lines, and refusing to look one line down would lose every total on
  every scanned invoice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .corpus import DATE_TOKEN, GSTIN_TOKEN, MONEY_TOKEN
from .features import LineContext
from .labels import SINGLE_VALUE_LABELS, SPEC_BY_LABEL

# Below this probability a line is not considered a candidate for a field at
# all. Set low deliberately: the cost of a wrong value here is a correction in
# a review screen, while the cost of a missing value is a field the user has
# to type from scratch.
MIN_FIELD_PROBABILITY = 0.20

# How many candidate lines to try per field before giving up.
MAX_CANDIDATES = 4

# Fields whose presence defines a usable extraction. Confidence is reported
# against these rather than against all fields, because a document legitimately
# missing a due date should not be scored as a partial failure.
CORE_FIELDS = (
    "INVOICE_NUMBER", "INVOICE_DATE", "VENDOR_NAME",
    "VENDOR_GSTIN", "SUBTOTAL", "TOTAL_AMOUNT",
)

# GSTIN character classes by position: 2 digits, 5 letters, 4 digits, 1 letter,
# 1 alphanumeric, a literal 'Z', 1 alphanumeric.
_GSTIN_SHAPE = "dd" + "aaaaa" + "dddd" + "a" + "x" + "Z" + "x"

# OCR confusions, applied only where the position tells us which class the
# character must belong to. Repairing blind would corrupt valid identifiers.
_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
                           "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
                           "6": "G", "4": "A", "7": "T"})

# A block heading to strip off a value line ("Bill To:", "Seller - ").
# The hyphen form requires whitespace on both sides: without that,
# "Coromandel Agro-Exports" parses as the heading "Coromandel Agro" and the
# party name comes back as "Exports".
_HEADING = re.compile(r"^\s*[A-Za-z][A-Za-z /]{2,24}(?:\s*:\s*|\s+-\s+)")
_TRAILING_LABEL = re.compile(r"\s*[:\-]\s*$")


@dataclass
class ExtractedField:
    label: str
    value: object
    raw_text: str
    confidence: float
    line_index: int


def repair_gstin(candidate: str) -> str | None:
    """Fix position-appropriate OCR confusions in a 15-character token and
    return it if the result is a structurally valid GSTIN."""
    from ...modules.invoices.parsers.gst_utils import GSTIN_PATTERN

    token = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if len(token) != 15:
        return None
    if GSTIN_PATTERN.fullmatch(token):
        return token

    repaired = []
    for character, expected in zip(token, _GSTIN_SHAPE):
        if expected == "d":
            repaired.append(character.translate(_TO_DIGIT))
        elif expected == "a":
            repaired.append(character.translate(_TO_ALPHA))
        elif expected == "Z":
            repaired.append("Z")
        else:
            repaired.append(character)
    result = "".join(repaired)
    return result if GSTIN_PATTERN.fullmatch(result) else None


def _parse_money(text: str) -> Decimal | None:
    """Last money token on the line. Invoice totals read "Label ... amount",
    so the last token is the value and any earlier one is a rate or a
    quantity."""
    from ...modules.invoices.parsers.money_utils import parse_amount

    tokens = MONEY_TOKEN.findall(text)
    if not tokens:
        return None
    return parse_amount(tokens[-1])


def _parse_date(text: str) -> date | None:
    from dateutil import parser as dateutil_parser

    for token in DATE_TOKEN.findall(text):
        try:
            # Indian invoices are day-first; the ISO shape is unambiguous
            # either way, so this only decides ambiguous dd/mm vs mm/dd.
            return dateutil_parser.parse(token, dayfirst=True).date()
        except (ValueError, OverflowError, TypeError):
            continue
    return None


def _parse_gstin(text: str) -> str | None:
    upper = text.upper()
    for token in GSTIN_TOKEN.findall(upper):
        repaired = repair_gstin(token)
        if repaired:
            return repaired
    # OCR sometimes breaks a GSTIN with a space; retry on the whole line with
    # separators removed before giving up.
    collapsed = re.sub(r"[^A-Z0-9]", "", upper)
    for start in range(0, max(0, len(collapsed) - 14)):
        repaired = repair_gstin(collapsed[start:start + 15])
        if repaired:
            return repaired
    return None


def _parse_identifier(text: str) -> str | None:
    """An invoice number: whatever follows the label on the line, else the
    most identifier-shaped token present."""
    after_label = _HEADING.sub("", text).strip()
    if after_label and after_label != text.strip():
        candidate = after_label.split()[0] if after_label.split() else ""
        if any(character.isdigit() for character in candidate) and len(candidate) >= 3:
            return candidate.strip(".,;")
    tokens = re.findall(r"\b[A-Z0-9][A-Z0-9/\-]{2,}\b", text.upper())
    scored = [token for token in tokens if any(character.isdigit() for character in token)]
    if not scored:
        return None
    return max(scored, key=len).strip(".,;")


def _parse_text(text: str) -> str | None:
    """A party name: drop a leading block heading ("Seller:", "Bill To -")
    and keep the rest."""
    stripped = _TRAILING_LABEL.sub("", _HEADING.sub("", text)).strip()
    return stripped or None


_PARSERS = {
    "money": _parse_money,
    "date": _parse_date,
    "gstin": _parse_gstin,
    "identifier": _parse_identifier,
    "text": _parse_text,
}


def _value_from(context: LineContext, kind: str) -> tuple[object, str] | None:
    """Parse a value of `kind` from this line, falling back to the following
    line for value kinds that OCR commonly separates from their label."""
    parser = _PARSERS[kind]
    value = parser(context.text)
    if value is not None:
        return value, context.text
    if kind in {"money", "date", "gstin", "identifier"} and context.next_text:
        value = parser(context.next_text)
        if value is not None:
            return value, context.next_text
    if kind == "text" and not _parse_text(context.text) and context.next_text:
        value = parser(context.next_text)
        if value is not None:
            return value, context.next_text
    return None


def extract_fields(model, contexts: list[LineContext]) -> dict[str, ExtractedField]:
    """Run the model over a document's lines and reduce them to one value per
    field."""
    if not contexts:
        return {}

    probabilities = model.predict_proba(contexts)
    classes = list(model.classes)
    found: dict[str, ExtractedField] = {}
    claimed: set[int] = set()

    for label in SINGLE_VALUE_LABELS:
        if label not in classes:
            continue
        kind = SPEC_BY_LABEL[label].kind
        column = probabilities[:, classes.index(label)]
        ranked = sorted(range(len(contexts)), key=lambda index: -column[index])

        for line_index in ranked[:MAX_CANDIDATES]:
            probability = float(column[line_index])
            if probability < MIN_FIELD_PROBABILITY:
                break
            # One line rarely carries two different header fields; letting a
            # second field claim an already-used line is how a subtotal ends
            # up duplicated into the total.
            if line_index in claimed:
                continue
            parsed = _value_from(contexts[line_index], kind)
            if parsed is None:
                continue
            value, raw_text = parsed
            found[label] = ExtractedField(label, value, raw_text, round(probability, 4), line_index)
            claimed.add(line_index)
            break

    return found


def confidence_from(found: dict[str, ExtractedField]) -> float:
    """Document-level confidence: how much of the core field set was recovered,
    weighted by how sure the model was of each. Reported as
    `parsing_confidence`, which is what decides straight-through versus review.
    """
    if not CORE_FIELDS:
        return 0.0
    total = sum(found[label].confidence for label in CORE_FIELDS if label in found)
    return round(min(total / len(CORE_FIELDS), 1.0), 3)
