"""Geometric primitives shared by the OCR, layout and table stages.

Everything downstream of OCR reasons about *where* text sits, not just what
it says: a label 400px to the right of its value is still that value's label,
and a line-string match can never know that. This module is the vocabulary
for those questions.

Coordinates are always in pixels of the page image, origin top-left, y
growing downward — the convention both OpenCV and PaddleOCR use. Native-PDF
extraction produces points, not pixels, so the PDF service scales them to the
same raster space before building boxes; keeping one coordinate system means
the layout stage never has to ask which pipeline produced a box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle. Frozen because boxes are identity for a
    piece of text: mutating one in place would silently invalidate any index
    built over it."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # Normalise inverted input rather than rejecting it. Rotated polygons
        # and native-PDF coordinate flips both produce x1 < x0 legitimately,
        # and every consumer here assumes x0 <= x1.
        if self.x1 < self.x0:
            low, high = self.x1, self.x0
            object.__setattr__(self, "x0", low)
            object.__setattr__(self, "x1", high)
        if self.y1 < self.y0:
            low, high = self.y1, self.y0
            object.__setattr__(self, "y0", low)
            object.__setattr__(self, "y1", high)

    # --- basic measurements -------------------------------------------------

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    # --- construction -------------------------------------------------------

    @classmethod
    def from_polygon(cls, points: Sequence[Point]) -> "BoundingBox":
        """Axis-aligned hull of a detection polygon. PaddleOCR returns four
        corners that are *not* axis-aligned on skewed text; the hull is what
        every consumer here wants, and the original polygon is kept alongside
        for callers that care about the true shape."""
        if not points:
            raise ValueError("Cannot build a bounding box from an empty polygon")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def union(cls, boxes: Iterable["BoundingBox"]) -> "BoundingBox":
        boxes = list(boxes)
        if not boxes:
            raise ValueError("Cannot union an empty sequence of boxes")
        return cls(
            min(box.x0 for box in boxes),
            min(box.y0 for box in boxes),
            max(box.x1 for box in boxes),
            max(box.y1 for box in boxes),
        )

    def merged_with(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox.union((self, other))

    def scaled(self, factor_x: float, factor_y: float | None = None) -> "BoundingBox":
        """Rescale into another raster resolution. Used when a page is OCR'd
        at one DPI but displayed at another — the review UI overlays boxes on
        a browser-rendered page whose size it chose, not ours."""
        factor_y = factor_x if factor_y is None else factor_y
        return BoundingBox(
            self.x0 * factor_x, self.y0 * factor_y,
            self.x1 * factor_x, self.y1 * factor_y,
        )

    # --- relationships ------------------------------------------------------

    def vertical_overlap(self, other: "BoundingBox") -> float:
        """Overlapping height as a fraction of the *shorter* box's height.

        Shorter, not taller: a small superscript sitting inside a tall line
        belongs to that line, and dividing by the tall box would score that
        near zero. This is the primary test for "are these two fragments on
        the same text line".
        """
        overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        if overlap <= 0:
            return 0.0
        shorter = min(self.height, other.height)
        return overlap / shorter if shorter > 0 else 0.0

    def horizontal_overlap(self, other: "BoundingBox") -> float:
        """Overlapping width as a fraction of the narrower box's width. The
        column-alignment counterpart of vertical_overlap — this is how a value
        is recognised as sitting *under* its header."""
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        if overlap <= 0:
            return 0.0
        narrower = min(self.width, other.width)
        return overlap / narrower if narrower > 0 else 0.0

    def intersection_area(self, other: "BoundingBox") -> float:
        width = min(self.x1, other.x1) - max(self.x0, other.x0)
        height = min(self.y1, other.y1) - max(self.y0, other.y0)
        return width * height if width > 0 and height > 0 else 0.0

    def iou(self, other: "BoundingBox") -> float:
        intersection = self.intersection_area(other)
        if intersection <= 0:
            return 0.0
        return intersection / (self.area + other.area - intersection)

    def contains(self, other: "BoundingBox", tolerance: float = 2.0) -> bool:
        return (
            self.x0 - tolerance <= other.x0
            and self.y0 - tolerance <= other.y0
            and self.x1 + tolerance >= other.x1
            and self.y1 + tolerance >= other.y1
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def horizontal_gap(self, other: "BoundingBox") -> float:
        """Signed horizontal distance: positive when `other` sits to the right
        with clear air between them, negative when they overlap."""
        if other.x0 >= self.x1:
            return other.x0 - self.x1
        if self.x0 >= other.x1:
            return -(self.x0 - other.x1)
        return -min(self.x1, other.x1) + max(self.x0, other.x0)

    def vertical_gap(self, other: "BoundingBox") -> float:
        """Signed vertical distance: positive when `other` sits below."""
        if other.y0 >= self.y1:
            return other.y0 - self.y1
        if self.y0 >= other.y1:
            return -(self.y0 - other.y1)
        return -min(self.y1, other.y1) + max(self.y0, other.y0)

    def is_right_of(self, other: "BoundingBox", min_vertical_overlap: float = 0.4) -> bool:
        """Same text band, further right. The pairing rule for `Label: value`
        laid out across a page with whitespace or a table cell boundary in
        between."""
        return self.x0 >= other.center_x and self.vertical_overlap(other) >= min_vertical_overlap

    def is_below(self, other: "BoundingBox", min_horizontal_overlap: float = 0.3) -> bool:
        """Same column, further down. The pairing rule for a header stacked
        above its value, which is how most invoice metadata grids are laid
        out."""
        return self.y0 >= other.center_y and self.horizontal_overlap(other) >= min_horizontal_overlap

    def distance_to(self, other: "BoundingBox") -> float:
        """Edge-to-edge Euclidean distance; 0 when the boxes touch or overlap.
        Used as the tie-breaker when several candidates satisfy the same
        directional rule — the nearest one usually owns the label."""
        dx = max(0.0, max(self.x0 - other.x1, other.x0 - self.x1))
        dy = max(0.0, max(self.y0 - other.y1, other.y0 - self.y1))
        return (dx * dx + dy * dy) ** 0.5

    # --- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Serialised for the API as x/y/width/height rather than corners:
        that is what a CSS-positioned overlay in the review UI consumes, and
        converting on the client would put the same arithmetic in two places."""
        return {
            "x": round(self.x0, 2),
            "y": round(self.y0, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "BoundingBox":
        x = float(payload["x"])
        y = float(payload["y"])
        return cls(x, y, x + float(payload["width"]), y + float(payload["height"]))


def cluster_by_vertical_overlap(
    boxes: Sequence[BoundingBox],
    min_overlap: float = 0.5,
) -> list[list[int]]:
    """Group box indices into text lines by vertical overlap.

    Returns indices rather than boxes so the caller can carry its own payload
    (text, confidence, source word) alongside without this function needing to
    know about it.

    The sweep is greedy over y0-sorted boxes and compares each box against the
    *running union* of its group. Comparing against only the first member
    breaks on lines with mixed font sizes: a tall leading capital and a small
    trailing superscript may not overlap each other while both clearly belong
    to the line between them.
    """
    order = sorted(range(len(boxes)), key=lambda index: (boxes[index].y0, boxes[index].x0))
    groups: list[list[int]] = []
    group_extent: list[BoundingBox] = []

    for index in order:
        box = boxes[index]
        placed = False
        for position, extent in enumerate(group_extent):
            if box.vertical_overlap(extent) >= min_overlap:
                groups[position].append(index)
                group_extent[position] = extent.merged_with(box)
                placed = True
                break
        if not placed:
            groups.append([index])
            group_extent.append(box)

    # Left-to-right within a line, top-to-bottom between lines: reading order.
    for group in groups:
        group.sort(key=lambda index: boxes[index].x0)
    groups.sort(key=lambda group: min(boxes[index].y0 for index in group))
    return groups
