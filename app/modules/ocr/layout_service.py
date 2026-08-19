"""Layout analysis: turn a bag of positioned lines into a described page.

This is the stage that makes extraction template-free. A label-matching
parser working on flat text can only ask "what characters follow the word
`Total`". This stage lets the next one ask the questions that actually
identify a value on an invoice:

  - what sits immediately to the right of this label, across 400px of
    whitespace or a table cell boundary?
  - which column is this token in, and what is the header above that column?
  - is this line in the summary block at the bottom right, or is it a line
    item in the middle of the table?

None of those depend on a vendor's wording or layout, which is why they
generalise where a template does not.

Three products come out of it:

  **Columns** — vertical bands the page's text organises into, found from
  whitespace, not assumed.
  **Regions** — typed areas (header, parties, item table, summary, footer)
  located by position and content, not by fixed coordinates.
  **Key-value pairs** — labels bound to their values geometrically.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .dto import OcrLine, OcrPage, OcrWord
from .geometry import BoundingBox

logger = logging.getLogger(__name__)

# --- region kinds ----------------------------------------------------------

REGION_HEADER = "HEADER"          # title, logo band, vendor identity
REGION_PARTIES = "PARTIES"        # bill-to / ship-to / seller blocks
REGION_METADATA = "METADATA"      # invoice number, dates, PO — usually a grid
REGION_TABLE = "ITEM_TABLE"       # the line-item table
REGION_SUMMARY = "SUMMARY"        # subtotal, taxes, grand total
REGION_FOOTER = "FOOTER"          # bank details, terms, signature
REGION_BODY = "BODY"              # anything not confidently typed

# A label is bound to a value at most this far away, as a multiple of the
# label's own height. Beyond it the "nearest value to the right" is usually
# in a different column entirely — an unbound label is a better outcome than
# a confidently wrong binding.
MAX_PAIR_DISTANCE_RATIO = 12.0

# ...or this fraction of the page width, whichever is greater, for values to
# the right. Right-aligned metadata sits most of a page away from its label,
# which a height-based limit alone rejects.
MAX_PAIR_WIDTH_RATIO = 0.6

# Gap between text bands, as a multiple of median line height, above which
# the page is considered to have moved to a new block.
BLOCK_GAP_RATIO = 1.8

# A column gutter must be at least this fraction of the page width.
MIN_GUTTER_RATIO = 0.02

_LABEL_ENDING = re.compile(r"[:：\-–—]\s*$")
_MOSTLY_DIGITS = re.compile(r"^[\s\d.,()\-+/₹$€£%]*$")


@dataclass
class TextBlock:
    """A run of lines separated from its neighbours by vertical whitespace."""

    lines: list[OcrLine]
    box: BoundingBox
    column_index: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Column:
    """A vertical band the page's text organises into."""

    index: int
    x0: float
    x1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def contains(self, box: BoundingBox) -> bool:
        return self.x0 <= box.center_x <= self.x1


@dataclass
class Region:
    kind: str
    box: BoundingBox
    lines: list[OcrLine] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "box": self.box.to_dict(),
            "line_count": len(self.lines),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class KeyValuePair:
    """A label bound to its value, with the geometry that justified it."""

    label: str
    value: str
    label_box: BoundingBox
    value_box: BoundingBox
    relation: str                # "right" | "below" | "same-token"
    page_number: int = 1
    distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "relation": self.relation,
            "page_number": self.page_number,
            "label_box": self.label_box.to_dict(),
            "value_box": self.value_box.to_dict(),
        }


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    pairs: list[KeyValuePair] = field(default_factory=list)

    def region(self, kind: str) -> Region | None:
        for region in self.regions:
            if region.kind == kind:
                return region
        return None

    def lines_in(self, kind: str) -> list[OcrLine]:
        region = self.region(kind)
        return list(region.lines) if region else []

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "column_count": len(self.columns),
            "regions": [region.to_dict() for region in self.regions],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


# --- structure -------------------------------------------------------------


def _median_line_height(lines: list[OcrLine]) -> float:
    heights = sorted(line.box.height for line in lines if line.box.height > 0)
    if not heights:
        return 1.0
    return heights[len(heights) // 2]


def detect_columns(lines: list[OcrLine], page_width: float) -> list[Column]:
    """Find vertical bands from whitespace in the horizontal ink profile.

    Derived from the page rather than assumed, because column positions are
    exactly the thing that varies between vendors. A gutter counts only when
    it is wide enough to be deliberate — narrow gaps are word spacing, and
    treating them as columns shatters every line into fragments.
    """
    if not lines or page_width <= 0:
        return [Column(0, 0.0, page_width)]

    resolution = 400
    cell = page_width / resolution
    occupancy = [False] * resolution
    for line in lines:
        start = max(0, int(line.box.x0 / cell))
        end = min(resolution - 1, int(line.box.x1 / cell))
        for index in range(start, end + 1):
            occupancy[index] = True

    minimum_gutter_cells = max(2, int((MIN_GUTTER_RATIO * page_width) / cell))

    columns: list[Column] = []
    index = 0
    column_index = 0
    while index < resolution:
        if not occupancy[index]:
            index += 1
            continue
        start = index
        gap_run = 0
        while index < resolution:
            if occupancy[index]:
                gap_run = 0
                index += 1
                continue
            gap_run += 1
            if gap_run >= minimum_gutter_cells:
                break
            index += 1
        end = index - gap_run
        columns.append(Column(column_index, start * cell, (end + 1) * cell))
        column_index += 1
        index += 1

    return columns or [Column(0, 0.0, page_width)]


def detect_blocks(lines: list[OcrLine]) -> list[TextBlock]:
    """Split lines into blocks wherever the vertical gap widens.

    Blocks are what make a seller address distinguishable from the buyer
    address that follows it: the two are separated by whitespace, and nothing
    else about them differs.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (line.box.y0, line.box.x0))
    line_height = _median_line_height(ordered)
    threshold = line_height * BLOCK_GAP_RATIO

    blocks: list[TextBlock] = []
    current: list[OcrLine] = [ordered[0]]
    for previous, line in zip(ordered, ordered[1:]):
        if line.box.y0 - previous.box.y1 > threshold:
            blocks.append(TextBlock(current, BoundingBox.union(item.box for item in current)))
            current = []
        current.append(line)
    if current:
        blocks.append(TextBlock(current, BoundingBox.union(item.box for item in current)))
    return blocks


# --- key/value binding -----------------------------------------------------


def _looks_like_label(text: str) -> bool:
    """Whether a fragment reads as a field label rather than a value.

    A trailing colon is the strongest signal and the most common; failing
    that, a short run of words that is not itself mostly digits. Deliberately
    permissive — this only decides what to *attempt* to bind, and a wrong
    guess costs one unused candidate, while a missed label costs a field.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _LABEL_ENDING.search(stripped):
        return True
    if _MOSTLY_DIGITS.match(stripped):
        return False
    return len(stripped.split()) <= 5


def _clean_label(text: str) -> str:
    return _LABEL_ENDING.sub("", text.strip()).strip()


def extract_pairs(page: OcrPage) -> list[KeyValuePair]:
    """Bind labels to values by geometry.

    Three relations, tried in order of how reliably they hold on invoices:

    1. **same-token** — `Invoice No: INV-42` inside one detected fragment.
       Unambiguous, so it wins outright.
    2. **right** — the value sits in the same horizontal band, further right.
       The dominant layout for metadata.
    3. **below** — the value sits in the same vertical band, underneath.
       How column-headed grids and stacked address blocks are laid out.

    Nothing here knows any invoice vocabulary. Which of these pairs is the
    invoice number is the *next* stage's problem; this stage's job is to make
    sure that pair exists to be chosen.
    """
    words = [word for line in page.lines for word in line.words]

    # Word level runs first and owns every label whose colon is followed by
    # whitespace — which is nearly all of them. Its boxes are tighter than a
    # whole-line box, and it is the only pass that can recover two pairs from
    # one line. The line-level passes below then fill in only what it missed,
    # rather than producing a second, coarser copy of what it already found.
    pairs: list[KeyValuePair] = _word_level_pairs(words, page.page_number)
    pairs.extend(_aligned_value_pairs(page))
    covered = [pair.label_box for pair in pairs]

    def already_bound(box: BoundingBox) -> bool:
        return any(box.iou(existing) > 0.05 or box.contains(existing) for existing in covered)

    for line in page.lines:
        text = line.text.strip()
        if not text or already_bound(line.box):
            continue

        # A label glued to its value with no space — `Invoice No:INV-42` —
        # which word-level splitting cannot see.
        inline = _split_inline(text)
        if inline is not None:
            label, value = inline
            pairs.append(
                KeyValuePair(
                    label=label, value=value,
                    label_box=line.box, value_box=line.box,
                    relation="same-token", page_number=page.page_number, distance=0.0,
                )
            )
            covered.append(line.box)
            continue

        if not _looks_like_label(text):
            continue

        neighbour = (
            _nearest_to_the_right(line, page.lines, page.width)
            or _nearest_below(line, page.lines, page.height)
        )
        if neighbour is None:
            continue
        candidate, relation = neighbour
        pairs.append(
            KeyValuePair(
                label=_clean_label(text), value=candidate.text.strip(),
                label_box=line.box, value_box=candidate.box,
                relation=relation, page_number=page.page_number,
                distance=line.box.distance_to(candidate.box),
            )
        )
        covered.append(line.box)

    return pairs


def _split_inline(text: str) -> tuple[str, str] | None:
    """Split `Label: value` where both sit in one fragment.

    Only splits on the *first* separator, and only when both sides are
    non-empty: `Total: 1,200.00` splits, while a bare `Notes:` does not, and
    a time like `14:30` does not because its left side is digits.
    """
    match = re.match(r"^([^:：]{2,40}?)\s*[:：]\s*(.+)$", text)
    if not match:
        return None
    label, value = match.group(1).strip(), match.group(2).strip()
    if not label or not value:
        return None
    if _MOSTLY_DIGITS.match(label):
        return None  # a time, a ratio, a page marker — not a label
    return label, value


def _horizontal_limit(label: OcrLine, page_width: float) -> float:
    """How far right a value may sit and still belong to this label.

    Scaled to the page, not to the text height: right-aligning a value in the
    metadata column of an A4 invoice routinely puts it most of a page away
    from its label, and a height-based limit rejects the commonest layout
    there is. Binding across rows is prevented by `is_right_of` requiring
    vertical overlap, not by this distance — so a generous limit here is
    safe, and "nearest wins" still keeps a closer candidate ahead of a
    farther one.
    """
    height_based = label.box.height * MAX_PAIR_DISTANCE_RATIO
    return max(height_based, page_width * MAX_PAIR_WIDTH_RATIO) if page_width > 0 else height_based


def _nearest_to_the_right(
    label: OcrLine, lines: list[OcrLine], page_width: float = 0.0
) -> tuple[OcrLine, str] | None:
    best: OcrLine | None = None
    best_distance = float("inf")
    limit = _horizontal_limit(label, page_width)
    for candidate in lines:
        if candidate is label or not candidate.text.strip():
            continue
        if not candidate.box.is_right_of(label.box):
            continue
        distance = label.box.distance_to(candidate.box)
        if distance <= limit and distance < best_distance:
            best, best_distance = candidate, distance
    return (best, "right") if best is not None else None


def _nearest_below(
    label: OcrLine, lines: list[OcrLine], page_height: float = 0.0
) -> tuple[OcrLine, str] | None:
    """Nearest line directly beneath the label.

    Kept to a few line-heights, unlike the horizontal case: a value stacked
    under its header is always immediately under it, and a generous vertical
    reach would bind a header to whatever happens to be further down the same
    column — usually a line item.
    """
    best: OcrLine | None = None
    best_distance = float("inf")
    limit = label.box.height * MAX_PAIR_DISTANCE_RATIO
    del page_height  # reserved: vertical reach is intentionally not page-scaled
    for candidate in lines:
        if candidate is label or not candidate.text.strip():
            continue
        if not candidate.box.is_below(label.box):
            continue
        distance = label.box.distance_to(candidate.box)
        if distance <= limit and distance < best_distance:
            best, best_distance = candidate, distance
    return (best, "below") if best is not None else None


def _word_level_pairs(words: list[OcrWord], page_number: int) -> list[KeyValuePair]:
    """Bind labels that end mid-line, which line-level binding cannot see.

    A metadata row reading `Invoice Date: 14/08/2026    Due Date: 13/09/2026`
    is one line. At line granularity it yields at most one pair; at word
    granularity both survive. This is the concrete payoff of carrying word
    boxes rather than only line boxes.
    """
    pairs: list[KeyValuePair] = []
    for index, word in enumerate(words):
        if not word.text.strip().endswith((":", "：")):
            continue
        # Walk back over the label's other words: `Invoice Date:` is two.
        label_words = [word]
        for previous in reversed(words[max(0, index - 3):index]):
            if previous.box.vertical_overlap(word.box) < 0.5:
                break
            if previous.text.strip().endswith((":", "：")):
                break
            if _MOSTLY_DIGITS.match(previous.text.strip()):
                break
            label_words.insert(0, previous)

        value_words: list[OcrWord] = []
        for following in words[index + 1:index + 6]:
            if following.box.vertical_overlap(word.box) < 0.5:
                break
            if following.text.strip().endswith((":", "：")):
                break  # the next label has started
            if value_words and _looks_like_new_label(following, value_words[-1]):
                break
            value_words.append(following)

        if not value_words:
            continue

        label_box = BoundingBox.union(item.box for item in label_words)
        value_box = BoundingBox.union(item.box for item in value_words)
        pairs.append(
            KeyValuePair(
                label=_clean_label(" ".join(item.text for item in label_words)),
                value=" ".join(item.text for item in value_words).strip(),
                label_box=label_box, value_box=value_box,
                relation="right", page_number=page_number,
                distance=label_box.distance_to(value_box),
            )
        )
    return pairs


_NUMERIC_TOKEN = re.compile(r"^\(?-?(?:₹|Rs\.?|INR|\$)?\s*\d[\d,]*(?:\.\d+)?\)?%?$", re.IGNORECASE)
_PERCENT_TOKEN = re.compile(r"^\(?\d+(?:\.\d+)?\s*%\)?$")


def _aligned_value_pairs(page: OcrPage) -> list[KeyValuePair]:
    """Bind `Label            1,234.00` — a label and a value on one line,
    separated by alignment whitespace rather than a colon.

    This is how every summary block is laid out, and it carries the fields
    that matter most: subtotal, each GST component, round-off, grand total.
    Colon-based binding never sees it, and treating the whole line as a label
    is worse than not binding at all — the line then gets bound to whatever
    sits *below* it, which is the next summary row, so the grand total ends
    up recorded as the round-off.
    """
    pairs: list[KeyValuePair] = []
    for line in page.lines:
        words = [word for word in line.words if word.text.strip()]
        if len(words) < 2:
            continue

        value_word = words[-1]
        if not _NUMERIC_TOKEN.match(value_word.text.strip()):
            continue

        # The value must sit in its own column, not merely be the last word
        # of a sentence that happens to end in a number.
        gap = words[-2].box.horizontal_gap(value_word.box)
        if gap < value_word.box.height * 1.2:
            continue

        # Strip trailing rate tokens: `CGST 9%   1782.00` labels the amount
        # `CGST`, and keeping the `9%` would stop the label matching.
        label_words = list(words[:-1])
        while label_words and (
            _PERCENT_TOKEN.match(label_words[-1].text.strip())
            or _NUMERIC_TOKEN.match(label_words[-1].text.strip())
        ):
            label_words.pop()
        if not label_words:
            continue

        label_box = BoundingBox.union(word.box for word in label_words)
        pairs.append(
            KeyValuePair(
                label=_clean_label(" ".join(word.text for word in label_words)),
                value=value_word.text.strip(),
                label_box=label_box,
                value_box=value_word.box,
                relation="right",
                page_number=page.page_number,
                distance=label_box.distance_to(value_word.box),
            )
        )
    return pairs


def _looks_like_new_label(word: OcrWord, previous: OcrWord) -> bool:
    """Whether a wide gap means the value has ended and something else begun."""
    return previous.box.horizontal_gap(word.box) > previous.box.height * 2.5


# --- region typing ---------------------------------------------------------

_TABLE_HEADER_HINTS = (
    "description", "particulars", "item", "qty", "quantity", "rate", "amount",
    "hsn", "sac", "unit", "price", "taxable",
)
_SUMMARY_HINTS = (
    "sub total", "subtotal", "grand total", "total", "cgst", "sgst", "igst",
    "tax", "round off", "amount payable", "net payable", "balance due",
)
_PARTY_HINTS = (
    "bill to", "billed to", "ship to", "shipped to", "buyer", "consignee",
    "customer", "sold by", "seller", "supplier", "vendor", "sold to",
)
_FOOTER_HINTS = (
    "bank", "account", "ifsc", "terms", "conditions", "declaration",
    "signatory", "jurisdiction", "amount in words", "upi",
)


def _hint_score(text: str, hints: tuple[str, ...]) -> float:
    lowered = text.lower()
    return sum(1 for hint in hints if hint in lowered)


def classify_regions(page: OcrPage, blocks: list[TextBlock]) -> list[Region]:
    """Type each block by what it contains and where it sits.

    Content first, position as the tie-breaker. Position alone fails on the
    many invoices that put the summary block on the left, or the parties
    below the table; content alone fails on the word `Total`, which appears
    in the table header, in a line item, and in the summary.
    """
    if not blocks:
        return []

    page_height = page.height or max((block.box.y1 for block in blocks), default=1.0)
    regions: list[Region] = []

    table_index = _locate_item_table(blocks)

    for index, block in enumerate(blocks):
        text = block.text
        relative_y = block.box.center_y / page_height if page_height else 0.0

        if index == table_index:
            regions.append(Region(REGION_TABLE, block.box, block.lines, confidence=0.9))
            continue

        party_score = _hint_score(text, _PARTY_HINTS)
        summary_score = _hint_score(text, _SUMMARY_HINTS)
        footer_score = _hint_score(text, _FOOTER_HINTS)

        if footer_score >= 2 and relative_y > 0.55:
            kind, confidence = REGION_FOOTER, min(0.9, 0.5 + 0.15 * footer_score)
        elif summary_score >= 2 and (table_index is None or index > table_index):
            kind, confidence = REGION_SUMMARY, min(0.9, 0.5 + 0.15 * summary_score)
        elif party_score >= 1 and relative_y < 0.6:
            kind, confidence = REGION_PARTIES, min(0.9, 0.5 + 0.2 * party_score)
        elif relative_y < 0.2:
            kind, confidence = REGION_HEADER, 0.6
        elif _is_metadata_grid(block):
            kind, confidence = REGION_METADATA, 0.6
        else:
            kind, confidence = REGION_BODY, 0.3

        regions.append(Region(kind, block.box, block.lines, confidence=confidence))

    return regions


def _locate_item_table(blocks: list[TextBlock]) -> int | None:
    """Find the block holding the line-item table, by its header row.

    Anchored on the header rather than on "the biggest block" or "the middle
    of the page": the header is the one part of a line-item table every
    vendor has, whatever they call the columns.
    """
    best_index: int | None = None
    best_score = 0.0
    for index, block in enumerate(blocks):
        # Every line, not just the block's first few. Where the page has
        # uniform line spacing there is no gap for `detect_blocks` to split
        # on, so the whole invoice arrives as one block and the table header
        # sits well down inside it — a first-three-lines window finds nothing
        # and the line items are lost.
        for line in block.lines:
            score = _hint_score(line.text, _TABLE_HEADER_HINTS)
            if score >= 3 and score > best_score:
                best_index, best_score = index, score
    return best_index


def _is_metadata_grid(block: TextBlock) -> bool:
    """Whether a block looks like a label/value grid rather than prose.

    Measured by how many of its lines carry a colon: a metadata grid is
    mostly `Label: value`, while an address block or a terms paragraph is
    mostly not.
    """
    if not block.lines:
        return False
    with_colon = sum(1 for line in block.lines if ":" in line.text)
    return with_colon / len(block.lines) >= 0.5


def analyse_page(page: OcrPage) -> PageLayout:
    """Full layout analysis for one page."""
    lines = [line for line in page.lines if line.text.strip()]
    blocks = detect_blocks(lines)
    columns = detect_columns(lines, page.width)

    for block in blocks:
        for column in columns:
            if column.contains(block.box):
                block.column_index = column.index
                break

    layout = PageLayout(
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        blocks=blocks,
        columns=columns,
        regions=classify_regions(page, blocks),
        pairs=extract_pairs(page),
    )
    logger.info(
        "layout.analysed page=%s blocks=%d columns=%d regions=%s pairs=%d",
        page.page_number, len(blocks), len(columns),
        [region.kind for region in layout.regions], len(layout.pairs),
    )
    return layout
