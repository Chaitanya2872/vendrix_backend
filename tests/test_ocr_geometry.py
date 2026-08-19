"""Geometry primitives.

These are tested hard because every later stage's correctness rests on them:
if `is_right_of` is wrong, field extraction pairs the wrong value with the
label and the error surfaces three stages away as an inexplicably wrong
total.
"""
from app.modules.ocr.geometry import BoundingBox, cluster_by_vertical_overlap


def box(x0, y0, x1, y1):
    return BoundingBox(x0, y0, x1, y1)


class TestNormalisation:
    def test_inverted_coordinates_are_swapped_not_silently_kept(self):
        # A rotated polygon and a flipped PDF coordinate space both produce
        # x1 < x0 legitimately; every consumer assumes x0 <= x1.
        normalised = box(100, 80, 20, 10)
        assert (normalised.x0, normalised.x1) == (20, 100)
        assert (normalised.y0, normalised.y1) == (10, 80)
        assert normalised.width == 80
        assert normalised.height == 70

    def test_already_ordered_coordinates_are_untouched(self):
        ordered = box(10, 20, 30, 50)
        assert (ordered.x0, ordered.y0, ordered.x1, ordered.y1) == (10, 20, 30, 50)


class TestConstruction:
    def test_from_polygon_takes_the_axis_aligned_hull_of_skewed_text(self):
        # PaddleOCR returns four corners that are not axis-aligned when text
        # is skewed; the hull is what the layout stage works in.
        skewed = BoundingBox.from_polygon([(10, 12), (100, 8), (102, 30), (12, 34)])
        assert (skewed.x0, skewed.y0, skewed.x1, skewed.y1) == (10, 8, 102, 34)

    def test_union_spans_every_box(self):
        assert BoundingBox.union([box(10, 10, 20, 20), box(50, 5, 60, 30)]) == box(10, 5, 60, 30)

    def test_scaled_converts_between_render_resolutions(self):
        assert box(10, 20, 30, 40).scaled(2.0) == box(20, 40, 60, 80)
        assert box(10, 20, 30, 40).scaled(2.0, 0.5) == box(20, 10, 60, 20)


class TestSameLineDetection:
    def test_small_text_inside_a_tall_line_counts_as_the_same_line(self):
        # Denominator is the shorter box: a superscript sitting inside a tall
        # line belongs to it, and dividing by the taller box scores near zero.
        tall = box(0, 0, 100, 40)
        superscript = box(110, 2, 120, 12)
        assert tall.vertical_overlap(superscript) == 1.0

    def test_stacked_rows_do_not_overlap(self):
        assert box(0, 0, 100, 20).vertical_overlap(box(0, 25, 100, 45)) == 0.0

    def test_partial_overlap_is_a_fraction(self):
        assert box(0, 0, 10, 20).vertical_overlap(box(20, 10, 30, 30)) == 0.5


class TestDirectionalRules:
    def test_value_far_to_the_right_of_its_label_is_still_paired(self):
        # The case a line-string match can never handle: 400px of whitespace
        # between "Invoice No:" and its value.
        label = box(50, 100, 150, 120)
        value = box(550, 102, 700, 122)
        assert value.is_right_of(label)
        assert not label.is_right_of(value)

    def test_a_value_on_a_different_row_is_not_to_the_right(self):
        label = box(50, 100, 150, 120)
        next_row = box(550, 200, 700, 220)
        assert not next_row.is_right_of(label)

    def test_value_under_its_header_is_paired(self):
        header = box(400, 100, 500, 120)
        value = box(410, 130, 490, 150)
        assert value.is_below(header)
        assert not header.is_below(value)

    def test_a_value_in_the_neighbouring_column_is_not_below(self):
        header = box(400, 100, 500, 120)
        other_column = box(700, 130, 800, 150)
        assert not other_column.is_below(header)


class TestDistances:
    def test_horizontal_gap_is_positive_for_clear_air_and_negative_when_overlapping(self):
        assert box(0, 0, 100, 20).horizontal_gap(box(150, 0, 200, 20)) == 50
        assert box(0, 0, 100, 20).horizontal_gap(box(90, 0, 200, 20)) < 0

    def test_distance_is_zero_for_touching_boxes(self):
        assert box(0, 0, 100, 20).distance_to(box(100, 0, 200, 20)) == 0.0

    def test_distance_is_diagonal_when_boxes_share_no_axis(self):
        assert box(0, 0, 10, 10).distance_to(box(13, 14, 20, 20)) == 5.0

    def test_iou_of_identical_boxes_is_one_and_of_disjoint_boxes_is_zero(self):
        assert box(0, 0, 10, 10).iou(box(0, 0, 10, 10)) == 1.0
        assert box(0, 0, 10, 10).iou(box(50, 50, 60, 60)) == 0.0


class TestSerialisation:
    def test_round_trips_through_the_api_representation(self):
        original = box(12.5, 30.25, 112.5, 50.25)
        assert BoundingBox.from_dict(original.to_dict()) == original

    def test_serialises_as_position_and_size_for_css_overlays(self):
        assert box(10, 20, 60, 45).to_dict() == {"x": 10.0, "y": 20.0, "width": 50.0, "height": 25.0}


class TestClustering:
    def test_groups_boxes_into_reading_order_lines(self):
        boxes = [
            box(300, 100, 400, 118),   # line 1, right
            box(50, 200, 150, 218),    # line 2, left
            box(50, 102, 150, 120),    # line 1, left
        ]
        assert cluster_by_vertical_overlap(boxes) == [[2, 0], [1]]

    def test_mixed_font_sizes_on_one_line_stay_together(self):
        # Compared against the running union, not the first member: the tall
        # capital and the small trailing mark need not overlap each other.
        boxes = [box(0, 0, 30, 40), box(35, 10, 100, 32), box(105, 26, 120, 38)]
        assert cluster_by_vertical_overlap(boxes) == [[0, 1, 2]]

    def test_empty_input_yields_no_groups(self):
        assert cluster_by_vertical_overlap([]) == []
