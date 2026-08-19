"""Table structure recovery, for tables that came from pixels.

The existing table parser reads pdfplumber's cell grid, which exists only for
native PDFs. A scanned or photographed invoice arrives here as positioned
text and nothing else, and its line items are the single most valuable thing
on the page. This module rebuilds the grid.

Two strategies, tried in order of how much evidence they rest on:

**Ruled** — the table is drawn with lines. Morphological opening with a long
horizontal kernel keeps only horizontal rules; the same with a vertical
kernel keeps verticals; their intersections are the cell corners. When a
vendor draws their table, this is near-exact and costs one pass over the
image.

**Borderless** — no rules, so the columns must be inferred from where the
text *is not*. Columns are the vertical whitespace channels that run the full
height of the table; rows are y-clusters of text. Weaker evidence, so it is
validated before being believed: a candidate grid that does not put a number
in its last column on most rows is rejected rather than returned.

Both produce the same `DetectedTable`, so the header-mapping and row-typing
logic downstream is written once — and is the same logic the existing
pdfplumber path already uses, which is why `HEADER_ALIASES` is imported from
there rather than restated.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .dto import OcrLine, OcrPage
from .geometry import BoundingBox

logger = logging.getLogger(__name__)

METHOD_RULED = "ruled"
METHOD_BORDERLESS = "borderless"

# A rule must span this fraction of the table's width (or height) to count.
# Below it, the "rule" is an underline, a strikethrough, or the edge of a logo.
MIN_RULE_SPAN_RATIO = 0.30

# A whitespace channel must be this wide, relative to the median character
# width of the table's text, before it is treated as a column boundary.
MIN_COLUMN_GAP_CHARS = 2.5

# ...and must be clear on at least this fraction of the table's rows. A gap
# that only appears on two rows is a coincidence of two short cells, not a
# column boundary.
MIN_COLUMN_GAP_ROW_COVERAGE = 0.75

# A borderless candidate must put a number in its rightmost column on at
# least this fraction of its body rows. Every invoice line item ends in an
# amount; a grid that does not is a misdetected paragraph.
MIN_NUMERIC_TAIL_RATIO = 0.6

_NUMBER = re.compile(r"\d")
_MOSTLY_NUMERIC = re.compile(r"^[\s\d.,()\-+/₹$€£%]+$")


@dataclass
class TableCell:
    text: str
    box: BoundingBox
    row: int
    column: int
    confidence: float = 1.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class DetectedTable:
    cells: list[TableCell] = field(default_factory=list)
    box: BoundingBox | None = None
    row_count: int = 0
    column_count: int = 0
    method: str = METHOD_BORDERLESS
    page_number: int = 1
    header_row: int | None = None

    def grid(self) -> list[list[str]]:
        """Row-major text grid, in the shape the existing header-alias parser
        already consumes — so line-item parsing is not reimplemented for
        scans."""
        rows = [["" for _ in range(self.column_count)] for _ in range(self.row_count)]
        for cell in self.cells:
            if 0 <= cell.row < self.row_count and 0 <= cell.column < self.column_count:
                existing = rows[cell.row][cell.column]
                rows[cell.row][cell.column] = f"{existing} {cell.text}".strip() if existing else cell.text
        return rows

    def row(self, index: int) -> list[TableCell]:
        return sorted(
            (cell for cell in self.cells if cell.row == index),
            key=lambda cell: cell.column,
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "page_number": self.page_number,
            "rows": self.row_count,
            "columns": self.column_count,
            "header_row": self.header_row,
            "box": self.box.to_dict() if self.box else None,
        }


# --- ruled tables ----------------------------------------------------------


def find_rules(image: np.ndarray) -> tuple[list[BoundingBox], list[BoundingBox]]:
    """Locate drawn horizontal and vertical rules.

    Morphological opening with a long thin kernel keeps only runs of ink that
    are long in one direction — which is exactly what a rule is and what text
    is not. Kernel length is derived from the image size rather than fixed,
    so the same code works on a 150-DPI fax and a 600-DPI scan.
    """
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2
    )
    height, width = binary.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 30)))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    return _rule_boxes(horizontal, horizontal=True), _rule_boxes(vertical, horizontal=False)


def _rule_boxes(mask: np.ndarray, horizontal: bool) -> list[BoundingBox]:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        span = w if horizontal else h
        if span < 20:
            continue
        boxes.append(BoundingBox(float(x), float(y), float(x + w), float(y + h)))
    return boxes


def grid_from_rules(
    horizontals: list[BoundingBox],
    verticals: list[BoundingBox],
    tolerance: float = 8.0,
) -> tuple[list[float], list[float]] | None:
    """Reduce rule boxes to the y positions of rows and x positions of columns.

    Rules are merged when they sit within `tolerance` of each other: a drawn
    line is several pixels thick and often broken into segments by the text
    crossing it, so the raw contours are many more than the table has lines.
    """
    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    table_width = max(box.x1 for box in horizontals) - min(box.x0 for box in horizontals)
    table_height = max(box.y1 for box in verticals) - min(box.y0 for box in verticals)

    row_positions = _merge_positions(
        [box.center_y for box in horizontals if box.width >= table_width * MIN_RULE_SPAN_RATIO],
        tolerance,
    )
    column_positions = _merge_positions(
        [box.center_x for box in verticals if box.height >= table_height * MIN_RULE_SPAN_RATIO],
        tolerance,
    )

    if len(row_positions) < 2 or len(column_positions) < 2:
        return None
    return row_positions, column_positions


def _merge_positions(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    merged = [[ordered[0]]]
    for value in ordered[1:]:
        if value - merged[-1][-1] <= tolerance:
            merged[-1].append(value)
        else:
            merged.append([value])
    return [float(np.mean(group)) for group in merged]


def table_from_grid(
    page: OcrPage,
    row_positions: list[float],
    column_positions: list[float],
) -> DetectedTable:
    """Place the page's text into a known cell grid."""
    cells: list[TableCell] = []
    box = BoundingBox(column_positions[0], row_positions[0], column_positions[-1], row_positions[-1])

    for line in page.lines:
        if not line.text.strip():
            continue
        for word in line.words:
            row = _band_index(word.box.center_y, row_positions)
            column = _band_index(word.box.center_x, column_positions)
            if row is None or column is None:
                continue
            cells.append(
                TableCell(
                    text=word.text, box=word.box, row=row, column=column,
                    confidence=word.confidence,
                )
            )

    return _merge_cell_words(
        cells,
        row_count=len(row_positions) - 1,
        column_count=len(column_positions) - 1,
        box=box,
        method=METHOD_RULED,
        page_number=page.page_number,
    )


def _band_index(value: float, boundaries: list[float]) -> int | None:
    for index in range(len(boundaries) - 1):
        if boundaries[index] <= value <= boundaries[index + 1]:
            return index
    return None


def _merge_cell_words(
    word_cells: list[TableCell],
    row_count: int,
    column_count: int,
    box: BoundingBox | None,
    method: str,
    page_number: int,
) -> DetectedTable:
    """Combine per-word cells into one cell per grid position."""
    grouped: dict[tuple[int, int], list[TableCell]] = {}
    for cell in word_cells:
        grouped.setdefault((cell.row, cell.column), []).append(cell)

    cells: list[TableCell] = []
    for (row, column), members in grouped.items():
        members.sort(key=lambda cell: (round(cell.box.y0, 1), cell.box.x0))
        cells.append(
            TableCell(
                text=" ".join(member.text for member in members).strip(),
                box=BoundingBox.union(member.box for member in members),
                row=row,
                column=column,
                # Weakest word wins, for the same reason a merged OCR line
                # takes its worst fragment's score: an amount read poorly is
                # a poorly-read cell however crisp its neighbours were.
                confidence=min(member.confidence for member in members),
            )
        )

    table = DetectedTable(
        cells=sorted(cells, key=lambda cell: (cell.row, cell.column)),
        box=box,
        row_count=row_count,
        column_count=column_count,
        method=method,
        page_number=page_number,
    )
    table.header_row = _locate_header_row(table)
    return table


# --- borderless tables -----------------------------------------------------


def _row_bands(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group lines into table rows by vertical overlap."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: line.box.y0)
    bands: list[list[OcrLine]] = [[ordered[0]]]
    extents: list[BoundingBox] = [ordered[0].box]
    for line in ordered[1:]:
        if line.box.vertical_overlap(extents[-1]) >= 0.4:
            bands[-1].append(line)
            extents[-1] = extents[-1].merged_with(line.box)
        else:
            bands.append([line])
            extents.append(line.box)
    for band in bands:
        band.sort(key=lambda line: line.box.x0)
    return bands


def find_column_boundaries(lines: list[OcrLine], box: BoundingBox) -> list[float]:
    """Infer column edges from vertical whitespace that runs the table's height.

    A gap counts only if it is clear on most rows. A gap present on two rows
    is two short cells lining up by chance; a real column boundary is clear on
    nearly every row, because that is what makes it a column.
    """
    words = [word for line in lines for word in line.words if word.text.strip()]
    if not words or box.width <= 0:
        return []

    median_character = float(np.median([
        word.box.width / max(len(word.text), 1) for word in words
    ]))
    minimum_gap = max(median_character * MIN_COLUMN_GAP_CHARS, box.width * 0.01)

    resolution = 600
    cell = box.width / resolution
    bands = _row_bands(lines)
    if not bands:
        return []

    # For each horizontal position, how many rows have ink there.
    coverage = np.zeros(resolution, dtype=np.int32)
    for band in bands:
        occupied = np.zeros(resolution, dtype=bool)
        for line in band:
            for word in line.words:
                if not word.text.strip():
                    continue
                start = max(0, int((word.box.x0 - box.x0) / cell))
                end = min(resolution - 1, int((word.box.x1 - box.x0) / cell))
                occupied[start:end + 1] = True
        coverage += occupied.astype(np.int32)

    clear_threshold = len(bands) * (1.0 - MIN_COLUMN_GAP_ROW_COVERAGE)
    is_gap = coverage <= clear_threshold

    boundaries: list[float] = [box.x0]
    index = 0
    while index < resolution:
        if not is_gap[index]:
            index += 1
            continue
        start = index
        while index < resolution and is_gap[index]:
            index += 1
        gap_width = (index - start) * cell
        # Ignore the margins: whitespace before the first column and after
        # the last is not a boundary between anything.
        if gap_width >= minimum_gap and start > 0 and index < resolution:
            boundaries.append(box.x0 + ((start + index) / 2) * cell)
    boundaries.append(box.x1)
    return boundaries


def detect_borderless(page: OcrPage, lines: list[OcrLine]) -> DetectedTable | None:
    """Rebuild a table that was never drawn, from where its text sits."""
    lines = [line for line in lines if line.text.strip()]
    if len(lines) < 2:
        return None

    box = BoundingBox.union(line.box for line in lines)
    boundaries = find_column_boundaries(lines, box)
    if len(boundaries) < 3:  # fewer than two columns is not a table
        return None

    bands = _row_bands(lines)
    word_cells: list[TableCell] = []
    for row_index, band in enumerate(bands):
        for line in band:
            for word in line.words:
                if not word.text.strip():
                    continue
                column = _band_index(word.box.center_x, boundaries)
                if column is None:
                    continue
                word_cells.append(
                    TableCell(word.text, word.box, row_index, column, word.confidence)
                )

    table = _merge_cell_words(
        word_cells,
        row_count=len(bands),
        column_count=len(boundaries) - 1,
        box=box,
        method=METHOD_BORDERLESS,
        page_number=page.page_number,
    )

    if not _looks_like_a_line_item_table(table):
        logger.info(
            "table.borderless_rejected page=%s rows=%d columns=%d reason=no_numeric_tail",
            page.page_number, table.row_count, table.column_count,
        )
        return None
    return table


def _looks_like_a_line_item_table(table: DetectedTable) -> bool:
    """Validate a borderless candidate before believing it.

    Borderless detection works from weak evidence, and the failure mode is
    confidently returning a paragraph of prose as a five-column table. Every
    invoice line item ends in an amount, so requiring a numeric rightmost
    column on most body rows separates the two cheaply.
    """
    if table.row_count < 2 or table.column_count < 2:
        return False

    body_rows = [
        index for index in range(table.row_count)
        if table.header_row is None or index > table.header_row
    ]
    if not body_rows:
        return False

    last_column = table.column_count - 1
    with_numeric_tail = 0
    for index in body_rows:
        tail = [cell for cell in table.row(index) if cell.column == last_column]
        if tail and _MOSTLY_NUMERIC.match(tail[0].text.strip()) and _NUMBER.search(tail[0].text):
            with_numeric_tail += 1

    return with_numeric_tail / len(body_rows) >= MIN_NUMERIC_TAIL_RATIO


# --- header detection ------------------------------------------------------


def _header_aliases() -> dict[str, list[str]]:
    """The alias table the pdfplumber path already uses.

    Imported rather than restated so a new column wording is added in one
    place and both paths benefit; imported lazily to keep this module free of
    an invoice-module dependency at import time.
    """
    from app.modules.invoices.parsers.invoice_table_parser import HEADER_ALIASES

    return HEADER_ALIASES


def _locate_header_row(table: DetectedTable) -> int | None:
    """Find the row whose cells read as column headings.

    Searched only in the first few rows: a table's header is at its top, and
    scanning further finds the word 'Amount' in a line item's description.
    """
    aliases = _header_aliases()
    flattened = {alias for values in aliases.values() for alias in values}

    best_row: int | None = None
    best_score = 0
    for index in range(min(table.row_count, 4)):
        cells = table.row(index)
        if not cells:
            continue
        score = sum(
            1 for cell in cells
            if any(alias == cell.text.strip().lower() or alias in cell.text.strip().lower()
                   for alias in flattened)
        )
        if score >= 2 and score > best_score:
            best_row, best_score = index, score
    return best_row


def map_columns(table: DetectedTable) -> dict[int, str]:
    """Map column indices to canonical field names using the header row.

    Detected, never assumed by position: column order is the most arbitrary
    thing about a vendor's table, and a fixed index is a template by another
    name.
    """
    if table.header_row is None:
        return {}

    aliases = _header_aliases()
    mapping: dict[int, str] = {}
    for cell in table.row(table.header_row):
        normalised = re.sub(r"\s+", " ", cell.text.replace("\n", " ")).strip().lower()
        if not normalised:
            continue
        for canonical, candidates in aliases.items():
            if canonical in mapping.values():
                continue
            if normalised in candidates or any(candidate == normalised for candidate in candidates):
                mapping[cell.column] = canonical
                break
        else:
            # Fall back to substring matching, for headers carrying units:
            # "Rate (₹)", "Amount in INR", "Qty."
            for canonical, candidates in aliases.items():
                if canonical in mapping.values():
                    continue
                if any(candidate in normalised for candidate in candidates):
                    mapping[cell.column] = canonical
                    break
    return mapping


# --- entry point -----------------------------------------------------------


def extract_tables(
    page: OcrPage,
    region_lines: list[OcrLine] | None = None,
    image: np.ndarray | None = None,
) -> list[DetectedTable]:
    """Recover every table on a page, preferring drawn structure over inferred.

    `region_lines` narrows the search to the item-table region the layout
    stage identified. Passing the whole page also works but is weaker: header
    and footer text widens the inferred column boundaries and drags the row
    bands out of alignment.
    """
    lines = region_lines if region_lines is not None else page.lines
    tables: list[DetectedTable] = []

    if image is not None and image.size:
        try:
            horizontals, verticals = find_rules(image)
            grid = grid_from_rules(horizontals, verticals)
            if grid is not None:
                rows, columns = grid
                ruled = table_from_grid(page, rows, columns)
                if ruled.cells:
                    logger.info(
                        "table.ruled_detected page=%s rows=%d columns=%d",
                        page.page_number, ruled.row_count, ruled.column_count,
                    )
                    return [ruled]
        except Exception as exc:
            # A failed rule detection must fall through to the borderless
            # path, not lose the table entirely.
            logger.warning("table.rule_detection_failed page=%s error=%s", page.page_number, exc)

    borderless = detect_borderless(page, lines)
    if borderless is not None:
        logger.info(
            "table.borderless_detected page=%s rows=%d columns=%d",
            page.page_number, borderless.row_count, borderless.column_count,
        )
        tables.append(borderless)
    return tables


def continues(previous: DetectedTable, current: DetectedTable, tolerance: float = 0.15) -> bool:
    """Whether `current` is the continuation of `previous` onto a new page.

    Judged by column geometry rather than by a repeated header: many vendors
    repeat the header on page two and many do not, but the columns of a table
    that spans pages line up, because it is the same table.
    """
    if previous.column_count != current.column_count:
        return False
    if previous.box is None or current.box is None:
        return False
    width = max(previous.box.width, current.box.width)
    if width <= 0:
        return False
    return abs(previous.box.x0 - current.box.x0) / width <= tolerance and \
        abs(previous.box.x1 - current.box.x1) / width <= tolerance
