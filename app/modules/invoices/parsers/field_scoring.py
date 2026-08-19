"""Candidate generation and scoring: how a field's value is actually chosen.

The parser this replaces used first-match-wins over an ordered label list.
That is exact when the first match is right and silently wrong when it is
not — and on an invoice the word `Total` appears in the table header, in a
line item, and three times in the summary block, so "the first one" is a coin
flip.

Here, every plausible reading becomes a *candidate*, each candidate is scored
on independent evidence, and the highest score wins. The margin between the
winner and the runner-up is kept, because a field won by a hair is exactly
what a human should look at — and no first-match scheme can tell you that
such a field exists.

Evidence, all of it cheap and none of it vendor-specific:

  **label quality** — how well the text names the field (lexicon.py)
  **relation**      — inline beats to-the-right beats below beats positional
  **proximity**     — nearer values are more likely to belong to the label
  **type validity** — a date field's value must parse as a date
  **OCR confidence**— a value the recogniser doubted is a doubtful value
  **region prior**  — a grand total belongs in the summary block, not the
                      item table; a line item's amount is not the invoice's

Nothing here knows any vendor's layout. It knows what invoices are like.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dataclass_field
from datetime import date
from decimal import Decimal
from typing import Any

from app.modules.ocr.geometry import BoundingBox
from app.modules.ocr.layout_service import (
    REGION_FOOTER,
    REGION_HEADER,
    REGION_SUMMARY,
    REGION_TABLE,
    KeyValuePair,
    PageLayout,
)

from . import lexicon
from .gst_utils import GSTIN_PATTERN, PAN_PATTERN
from .money_utils import parse_amount

logger = logging.getLogger(__name__)

# How much each relation is trusted, before any other evidence.
RELATION_WEIGHT = {
    "same-token": 1.00,   # `Invoice No: INV-42` — unambiguous
    "right": 0.92,        # the dominant metadata layout
    "below": 0.84,        # column-headed grids and stacked blocks
    "positional": 0.55,   # no label at all; inferred from where it sits
}

# Region priors per field, as multipliers. Absent means neutral (1.0).
# These encode what an invoice *is*, not what any vendor's looks like.
REGION_PRIORS: dict[str, dict[str, float]] = {
    "total_amount":    {REGION_SUMMARY: 1.30, REGION_TABLE: 0.45, REGION_HEADER: 0.7},
    "subtotal":        {REGION_SUMMARY: 1.25, REGION_TABLE: 0.50},
    "taxable_amount":  {REGION_SUMMARY: 1.25, REGION_TABLE: 0.50},
    "tax_amount":      {REGION_SUMMARY: 1.25, REGION_TABLE: 0.50},
    "cgst_amount":     {REGION_SUMMARY: 1.25, REGION_TABLE: 0.55},
    "sgst_amount":     {REGION_SUMMARY: 1.25, REGION_TABLE: 0.55},
    "igst_amount":     {REGION_SUMMARY: 1.25, REGION_TABLE: 0.55},
    "round_off":       {REGION_SUMMARY: 1.25},
    "amount_due":      {REGION_SUMMARY: 1.20},
    "invoice_number":  {REGION_HEADER: 1.20, REGION_TABLE: 0.5, REGION_FOOTER: 0.6},
    "invoice_date":    {REGION_HEADER: 1.15, REGION_TABLE: 0.5},
    "due_date":        {REGION_HEADER: 1.10, REGION_TABLE: 0.5},
    "bank_account_number": {REGION_FOOTER: 1.25},
    "bank_ifsc":       {REGION_FOOTER: 1.25},
    "upi_id":          {REGION_FOOTER: 1.25},
}

# Proximity decay: a value this many label-heights away scores half.
PROXIMITY_HALF_LIFE = 14.0

# A candidate whose value does not parse as its field's type is dropped, not
# merely penalised. An invoice date of "Terms" is not a weak reading of a
# date; it is not a date.
DROP_ON_TYPE_FAILURE = True

_DATE_TOKEN = re.compile(
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}"
)
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9/_\-]{2,}")
_MONEY_TOKEN = re.compile(r"-?(?:₹|Rs\.?|INR|\$)?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?")


@dataclass
class Candidate:
    """One plausible reading of one field."""

    field: str
    raw_value: str
    parsed_value: Any
    relation: str
    box: BoundingBox
    page_number: int = 1
    label_text: str | None = None
    label_score: float = 0.0
    ocr_confidence: float = 1.0
    region: str | None = None
    distance: float = 0.0
    score: float = 0.0
    quality: float = 0.0
    reasons: list[str] = dataclass_field(default_factory=list)

    def explain(self) -> dict:
        """Everything that went into the score.

        Persisted with the extraction so a reviewer asking "why did it pick
        that" gets an answer, and so a threshold can be retuned against real
        decisions rather than guesses.
        """
        return {
            "field": self.field,
            "value": self.raw_value,
            "relation": self.relation,
            "label": self.label_text,
            "label_score": round(self.label_score, 3),
            "ocr_confidence": round(self.ocr_confidence, 3),
            "region": self.region,
            "distance": round(self.distance, 1),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "page_number": self.page_number,
            "box": self.box.to_dict() if self.box else None,
        }


@dataclass
class Resolution:
    """The chosen candidate for one field, and how safely it was chosen."""

    field: str
    winner: Candidate
    runner_up: Candidate | None = None
    alternatives: int = 0

    @property
    def margin(self) -> float:
        """How far clear the winner was, in [0, 1].

        A field won by a hair is the one a human should check, and this is
        the only number in the system that can say so. It feeds directly into
        the confidence score.
        """
        if self.runner_up is None or self.winner.score <= 0:
            return 1.0
        return max(0.0, (self.winner.score - self.runner_up.score) / self.winner.score)

    @property
    def contested(self) -> bool:
        return self.margin < 0.15


# --- value typing ----------------------------------------------------------


def parse_value(kind: str, text: str) -> Any:
    """Turn a value string into a typed value, or None if it is not one.

    Returning None is the type check: a field's candidate survives only if
    its value is actually of that field's type.
    """
    if kind == "money":
        return _parse_money(text)
    if kind == "date":
        return _parse_date(text)
    if kind == "gstin":
        return _parse_gstin(text)
    if kind == "identifier":
        return _parse_identifier(text)
    if kind == "percent":
        return _parse_percent(text)
    return text.strip() or None


def _parse_money(text: str) -> Decimal | None:
    """The amount on a line, which is the *last* money token, not the first.

    `CGST 9%    1782.00` contains two numbers, and the first one is a rate.
    Taking the first match records the tax rate as the tax amount — a wrong
    value that looks entirely plausible in a review screen, which is the
    worst kind. Percentages are excluded outright, and the rightmost
    surviving token wins because that is where invoices put the amount.
    """
    if lexicon.is_negative_label(text):
        return None

    candidates = [
        match for match in _MONEY_TOKEN.finditer(text)
        if not text[match.end():match.end() + 1].strip().startswith("%")
    ]
    if not candidates:
        return None
    return parse_amount(candidates[-1].group(0))


def _parse_date(text: str) -> date | None:
    from dateutil import parser as dateutil_parser

    match = _DATE_TOKEN.search(text)
    if not match:
        return None
    try:
        # dayfirst: the deployment is India-only, where dd/mm/yyyy is
        # universal. An ambiguous 03/04/2026 is 3 April, not 4 March.
        parsed = dateutil_parser.parse(match.group(0), dayfirst=True, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None
    # A date decades out is a misread, not a date. Invoices are not dated
    # 1907, and a two-digit year misparse is the usual cause.
    if not (1990 <= parsed.year <= 2100):
        return None
    return parsed.date()


def _parse_gstin(text: str) -> str | None:
    match = GSTIN_PATTERN.search(text.upper().replace(" ", ""))
    return match.group(0) if match else None


def _parse_identifier(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = _IDENTIFIER_TOKEN.search(stripped)
    if not match:
        return None
    value = match.group(0)
    # An identifier that is only punctuation-and-separators is not one.
    return value if any(character.isalnum() for character in value) else None


def _parse_percent(text: str) -> Decimal | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return Decimal(match.group(1)) if match else None


# --- candidate generation --------------------------------------------------


def candidates_from_pairs(
    pairs: list[KeyValuePair],
    layout: PageLayout,
    ocr_confidence_by_box: dict | None = None,
) -> list[Candidate]:
    """Turn every geometric label→value pair into typed field candidates.

    One pair can produce candidates for several fields: `Total` names both
    `total_amount` and, less plausibly, `tax_amount`. Both are generated, and
    the score decides — which is the entire point of not stopping at the
    first match.
    """
    generated: list[Candidate] = []
    for pair in pairs:
        matches = lexicon.match_any(pair.label)
        if not matches:
            continue
        region = _region_of(layout, pair.label_box)

        for match in matches:
            spec = lexicon.FIELD_BY_NAME[match.field]
            parsed = parse_value(spec.kind, pair.value)
            if parsed is None and DROP_ON_TYPE_FAILURE:
                continue
            generated.append(
                Candidate(
                    field=match.field,
                    raw_value=pair.value,
                    parsed_value=parsed,
                    relation=pair.relation,
                    box=pair.value_box,
                    page_number=pair.page_number,
                    label_text=pair.label,
                    label_score=match.score,
                    region=region,
                    distance=pair.distance,
                    ocr_confidence=_confidence_for(pair.value_box, ocr_confidence_by_box),
                )
            )
    return generated


def candidates_from_position(layout: PageLayout) -> list[Candidate]:
    """Generate label-free candidates from where values sit.

    Two invoice regularities carry real information even with no label at
    all: the grand total is the bottom-most money in the summary block, and a
    GSTIN is a GSTIN wherever it appears. These score low by construction —
    they are a fallback for documents whose labels OCR destroyed, not a
    competitor to a clean label match.
    """
    generated: list[Candidate] = []

    # Every summary region, not just the first: a summary block split by
    # whitespace becomes several blocks, and looking only at the first finds
    # the subtotal and stops before reaching the grand total.
    summary_lines = [line for region in layout.regions if region.kind == REGION_SUMMARY
                     for line in region.lines]
    reason = "bottom-most amount in the summary block"

    if not summary_lines:
        # No summary region — which happens precisely when OCR mangled the
        # summary labels past matching, i.e. exactly when this fallback is
        # needed. Fall back to geometry alone: amounts in the lower-right of
        # the page, below anything that looked like a table.
        summary_lines = _lower_right_lines(layout)
        reason = "bottom-most amount in the lower-right of the page"

    money_lines = [
        (line, _parse_money(line.text))
        for line in summary_lines
        if _parse_money(line.text) is not None
    ]
    if money_lines:
        # Bottom-most, then largest: summary blocks read downward to the
        # grand total, and where two sit at the same height the larger is
        # the total and the smaller is the round-off.
        line, value = max(money_lines, key=lambda item: (item[0].box.y0, item[1]))
        generated.append(
            Candidate(
                field="total_amount", raw_value=line.text, parsed_value=value,
                relation="positional", box=line.box, page_number=line.page_number,
                label_text=None, label_score=0.0,
                region=_region_of(layout, line.box),
                ocr_confidence=line.confidence,
                reasons=[reason],
            )
        )

    all_lines = [line for block in layout.blocks for line in block.lines]
    for line in all_lines:
        found = GSTIN_PATTERN.search(line.text.upper().replace(" ", ""))
        if found:
            generated.append(
                Candidate(
                    field="gstin", raw_value=found.group(0), parsed_value=found.group(0),
                    relation="positional", box=line.box, page_number=line.page_number,
                    label_score=0.0, region=_region_of(layout, line.box),
                    ocr_confidence=line.confidence,
                    reasons=["structurally valid GSTIN"],
                )
            )

    return generated


def _lower_right_lines(layout: PageLayout) -> list:
    """Lines in the lower-right quadrant, below any detected item table.

    Where an invoice puts its totals when you cannot read its labels. Weak
    evidence, which is why candidates built from it carry the positional
    relation and score accordingly — but far better than returning nothing
    for a scan whose summary wording OCR destroyed.
    """
    table = layout.region(REGION_TABLE)
    floor = table.box.y1 if table is not None else layout.height * 0.5
    return [
        line
        for block in layout.blocks
        for line in block.lines
        if line.box.y0 >= floor and line.box.center_x >= layout.width * 0.45
    ]


def _region_of(layout: PageLayout, box: BoundingBox) -> str | None:
    for region in layout.regions:
        if region.box.contains(box, tolerance=4.0) or region.box.iou(box) > 0.2:
            return region.kind
    return None


def _confidence_for(box: BoundingBox, lookup: dict | None) -> float:
    if not lookup:
        return 1.0
    return float(lookup.get((round(box.x0), round(box.y0)), 1.0))


# --- scoring ---------------------------------------------------------------


def score_candidate(candidate: Candidate) -> float:
    """Combine the evidence into a ranking score in [0, 1], and separately a
    quality score used as the basis for confidence.

    The two are **not** the same, and conflating them mis-scores the most
    ordinary invoice there is:

      **score** ranks candidates against each other. Proximity and region
      priors belong here — given two readings of `Total`, the nearer one in
      the summary block is the better bet.

      **quality** says how much to believe the winner once it has won. It
      excludes proximity and region priors, because those are statements
      about *rivalry*, not about correctness. An amount right-aligned 250px
      from its label is not less likely to be right; that is simply what a
      summary block looks like. Charging it for the distance sent perfectly
      extracted invoices to manual review.

    Both are multiplicative: each factor is a probability-like belief, and a
    candidate that fails badly on one should not be rescued by the others.
    An amount read at 0.2 OCR confidence is a bad candidate however perfect
    its label was.
    """
    reasons: list[str] = list(candidate.reasons)

    relation = RELATION_WEIGHT.get(candidate.relation, 0.5)
    reasons.append(f"relation={candidate.relation} ({relation:.2f})")

    # A positional candidate has no label; give it a floor so it is not
    # multiplied to nothing, but keep it well below any real label match.
    label = candidate.label_score if candidate.label_score > 0 else 0.35
    reasons.append(f"label={label:.2f}")

    proximity = 1.0
    if candidate.distance > 0:
        height = max(candidate.box.height, 1.0)
        proximity = 0.5 ** (candidate.distance / (height * PROXIMITY_HALF_LIFE))
        proximity = max(proximity, 0.35)
        reasons.append(f"proximity={proximity:.2f}")

    ocr = max(0.2, min(1.0, candidate.ocr_confidence))
    if ocr < 0.8:
        reasons.append(f"low OCR confidence ({ocr:.2f})")

    prior = REGION_PRIORS.get(candidate.field, {}).get(candidate.region or "", 1.0)
    if prior != 1.0:
        reasons.append(f"region={candidate.region} ({prior:.2f})")

    candidate.reasons = reasons
    candidate.quality = min(1.0, relation * label * ocr)
    candidate.score = min(1.0, relation * label * proximity * ocr * prior)
    return candidate.score


def resolve(candidates: list[Candidate]) -> dict[str, Resolution]:
    """Pick one candidate per field, keeping the runner-up.

    Candidates proposing the same value for the same field are merged and
    mildly boosted before the comparison: two independent routes agreeing is
    corroboration, and treating them as rivals would make an agreed value
    look contested.
    """
    for candidate in candidates:
        score_candidate(candidate)

    by_field: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_field.setdefault(candidate.field, []).append(candidate)

    resolutions: dict[str, Resolution] = {}
    for field_name, group in by_field.items():
        merged = _merge_agreeing(group)
        merged.sort(key=lambda item: item.score, reverse=True)
        resolutions[field_name] = Resolution(
            field=field_name,
            winner=merged[0],
            runner_up=merged[1] if len(merged) > 1 else None,
            alternatives=len(merged) - 1,
        )
        if resolutions[field_name].contested:
            logger.info(
                "extraction.contested field=%s winner=%r (%.3f) runner_up=%r (%.3f)",
                field_name, merged[0].raw_value, merged[0].score,
                merged[1].raw_value, merged[1].score,
            )
    return resolutions


def _merge_agreeing(group: list[Candidate]) -> list[Candidate]:
    """Collapse candidates that propose the same value, boosting agreement."""
    by_value: dict[str, list[Candidate]] = {}
    for candidate in group:
        by_value.setdefault(str(candidate.parsed_value), []).append(candidate)

    merged: list[Candidate] = []
    for members in by_value.values():
        members.sort(key=lambda item: item.score, reverse=True)
        best = members[0]
        if len(members) > 1:
            # Diminishing: three routes agreeing is barely better than two,
            # and unbounded growth would let a repeated weak reading beat a
            # single strong one.
            boost = min(1.15, 1.0 + 0.05 * (len(members) - 1))
            best.score = min(1.0, best.score * boost)
            best.reasons.append(f"{len(members)} independent readings agree")
        merged.append(best)
    return merged


def extract_fields(
    layout: PageLayout,
    ocr_confidence_by_box: dict | None = None,
) -> dict[str, Resolution]:
    """Full field extraction for one page's layout."""
    generated = candidates_from_pairs(layout.pairs, layout, ocr_confidence_by_box)
    generated.extend(candidates_from_position(layout))
    resolved = resolve(generated)
    logger.info(
        "extraction.resolved fields=%d contested=%d",
        len(resolved), sum(1 for item in resolved.values() if item.contested),
    )
    return resolved
