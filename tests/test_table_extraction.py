"""Table structure recovery from positioned text and from drawn rules.

The line-item table is the most valuable thing on an invoice and the thing a
scan destroys most completely. These tests build tables at explicit
coordinates — the way real vendors lay them out — and check both that a real
table is recovered and, just as importantly, that prose is not mistaken for
one.
"""
import cv2
import numpy as np
import pytest

from app.modules.ocr import table_extraction_service as tables
from app.modules.ocr.dto import SOURCE_OCR, OcrLine, OcrPage, split_line_into_words
from app.modules.ocr.geometry import BoundingBox
from app.modules.ocr.table_extraction_service import (
    METHOD_BORDERLESS,
    METHOD_RULED,
    detect_borderless,
    extract_tables,
    map_columns,
)

CHARACTER_WIDTH = 12.0
LINE_HEIGHT = 30.0


def cell_line(text, x, y):
    box = BoundingBox(x, y, x + len(text) * CHARACTER_WIDTH, y + LINE_HEIGHT)
    return OcrLine(text=text, box=box, confidence=0.95, words=split_line_into_words(text, box, 0.95))


def row(columns, y):
    """One table row: (text, x) pairs laid out at fixed column positions."""
    return [cell_line(text, x, y) for text, x in columns]


def page_of(lines, width=2480.0, height=3508.0):
    ordered = sorted(lines, key=lambda line: (line.box.y0, line.box.x0))
    for index, line in enumerate(ordered):
        line.line_index = index
    return OcrPage(page_number=1, width=width, height=height, lines=ordered, source=SOURCE_OCR)


COLUMNS = [200, 900, 1250, 1550, 1950]


def invoice_table_lines():
    """A four-row borderless line-item table with well-separated columns."""
    lines = []
    lines += row([("Description", COLUMNS[0]), ("HSN", COLUMNS[1]), ("Qty", COLUMNS[2]),
                  ("Rate", COLUMNS[3]), ("Amount", COLUMNS[4])], 900)
    lines += row([("Steel Fabrication", COLUMNS[0]), ("7308", COLUMNS[1]), ("10", COLUMNS[2]),
                  ("1500.00", COLUMNS[3]), ("15000.00", COLUMNS[4])], 960)
    lines += row([("Welding Services", COLUMNS[0]), ("9988", COLUMNS[1]), ("4", COLUMNS[2]),
                  ("1200.00", COLUMNS[3]), ("4800.00", COLUMNS[4])], 1020)
    lines += row([("Site Supervision", COLUMNS[0]), ("9983", COLUMNS[1]), ("2", COLUMNS[2]),
                  ("3000.00", COLUMNS[3]), ("6000.00", COLUMNS[4])], 1080)
    return lines


class TestBorderlessDetection:
    def test_a_borderless_table_is_recovered(self):
        lines = invoice_table_lines()

        table = detect_borderless(page_of(lines), lines)

        assert table is not None
        assert table.method == METHOD_BORDERLESS
        assert table.row_count == 4
        assert table.column_count == 5

    def test_cells_land_in_the_right_row_and_column(self):
        lines = invoice_table_lines()

        grid = detect_borderless(page_of(lines), lines).grid()

        assert grid[1] == ["Steel Fabrication", "7308", "10", "1500.00", "15000.00"]
        assert grid[3] == ["Site Supervision", "9983", "2", "3000.00", "6000.00"]

    def test_a_multi_word_description_stays_in_one_cell(self):
        # Word-level placement with per-cell merging: the words of a
        # description must not scatter across the columns to their right.
        lines = invoice_table_lines()

        grid = detect_borderless(page_of(lines), lines).grid()

        assert grid[1][0] == "Steel Fabrication"
        assert grid[1][1] == "7308"

    def test_the_header_row_is_located(self):
        lines = invoice_table_lines()

        table = detect_borderless(page_of(lines), lines)

        assert table.header_row == 0

    def test_cell_confidence_takes_the_weakest_word(self):
        lines = invoice_table_lines()
        # Simulate one badly-read amount.
        for line in lines:
            if line.text == "15000.00":
                line.words[0].confidence = 0.41

        table = detect_borderless(page_of(lines), lines)

        amount = [cell for cell in table.row(1) if cell.column == 4][0]
        assert amount.confidence == pytest.approx(0.41)


class TestFalsePositives:
    def test_a_paragraph_of_prose_is_not_returned_as_a_table(self):
        """The borderless failure mode: confidently returning body text as a
        five-column table. Every line item ends in an amount; prose does not."""
        lines = [
            cell_line("These goods remain the property of the seller until", 200, 900),
            cell_line("payment has been received in full and cleared funds", 200, 960),
            cell_line("are available in the seller's nominated bank account", 200, 1020),
        ]

        assert detect_borderless(page_of(lines), lines) is None

    def test_a_single_line_is_not_a_table(self):
        lines = [cell_line("Grand Total 23364.00", 200, 900)]

        assert detect_borderless(page_of(lines), lines) is None

    def test_an_address_block_is_not_a_table(self):
        lines = [
            cell_line("Beta Constructions Private Limited", 200, 500),
            cell_line("14 Marine Drive", 200, 560),
            cell_line("Mumbai 400001", 200, 620),
        ]

        assert detect_borderless(page_of(lines), lines) is None

    def test_a_two_column_layout_without_amounts_is_rejected(self):
        lines = []
        lines += row([("Payment Terms", 200), ("Net thirty days", 1200)], 900)
        lines += row([("Delivery Terms", 200), ("Ex works Pune", 1200)], 960)
        lines += row([("Warranty", 200), ("Twelve months", 1200)], 1020)

        assert detect_borderless(page_of(lines), lines) is None


class TestColumnInference:
    def test_a_gap_present_on_only_one_row_is_not_a_column_boundary(self):
        """Two short cells lining up by chance is not a column. A real
        boundary is clear on nearly every row, because that is what makes
        it one."""
        lines = invoice_table_lines()
        # One row whose description happens to have an internal gap.
        lines += row([("Extra", COLUMNS[0]), ("Item", 500), ("1", COLUMNS[2]),
                      ("100.00", COLUMNS[3]), ("100.00", COLUMNS[4])], 1140)

        table = detect_borderless(page_of(lines), lines)

        assert table.column_count == 5, "the one-row gap must not create a sixth column"

    def test_narrow_word_spacing_does_not_split_a_description(self):
        lines = invoice_table_lines()

        table = detect_borderless(page_of(lines), lines)

        assert table.grid()[1][0] == "Steel Fabrication"

    def test_a_table_whose_columns_are_close_together_still_resolves(self):
        lines = []
        tight = [200, 700, 900, 1100, 1350]
        lines += row([("Item", tight[0]), ("HSN", tight[1]), ("Qty", tight[2]),
                      ("Rate", tight[3]), ("Amount", tight[4])], 900)
        lines += row([("Bolts", tight[0]), ("7318", tight[1]), ("50", tight[2]),
                      ("12.00", tight[3]), ("600.00", tight[4])], 960)
        lines += row([("Nuts", tight[0]), ("7318", tight[1]), ("50", tight[2]),
                      ("8.00", tight[3]), ("400.00", tight[4])], 1020)

        table = detect_borderless(page_of(lines), lines)

        assert table is not None
        assert table.column_count == 5


class TestHeaderMapping:
    def test_columns_map_to_canonical_fields(self):
        lines = invoice_table_lines()

        mapping = map_columns(detect_borderless(page_of(lines), lines))

        assert mapping[0] == "description"
        assert mapping[1] == "hsn_sac"
        assert mapping[2] == "quantity"
        assert mapping[3] == "unit_price"
        assert mapping[4] == "total_amount"

    @pytest.mark.parametrize(
        "headers",
        [
            ["Particulars", "SAC", "Quantity", "Price", "Total"],
            ["Item Description", "HSN Code", "Qty", "Unit Price", "Net Amount"],
            ["Product", "HSN", "Units", "Rate", "Line Total"],
        ],
    )
    def test_mapping_survives_different_vendor_wordings(self, headers):
        """No template: three vendors' column names, one mapping."""
        lines = row([(headers[index], COLUMNS[index]) for index in range(5)], 900)
        lines += row([("Steel Fabrication", COLUMNS[0]), ("7308", COLUMNS[1]), ("10", COLUMNS[2]),
                      ("1500.00", COLUMNS[3]), ("15000.00", COLUMNS[4])], 960)
        lines += row([("Welding", COLUMNS[0]), ("9988", COLUMNS[1]), ("4", COLUMNS[2]),
                      ("1200.00", COLUMNS[3]), ("4800.00", COLUMNS[4])], 1020)

        mapping = map_columns(detect_borderless(page_of(lines), lines))

        assert mapping[0] == "description"
        assert mapping[4] == "total_amount"

    def test_a_header_carrying_units_still_maps(self):
        lines = row([("Description", COLUMNS[0]), ("HSN/SAC", COLUMNS[1]), ("Qty.", COLUMNS[2]),
                     ("Rate (Rs)", COLUMNS[3]), ("Amount in INR", COLUMNS[4])], 900)
        lines += row([("Steel Fabrication", COLUMNS[0]), ("7308", COLUMNS[1]), ("10", COLUMNS[2]),
                      ("1500.00", COLUMNS[3]), ("15000.00", COLUMNS[4])], 960)
        lines += row([("Welding", COLUMNS[0]), ("9988", COLUMNS[1]), ("4", COLUMNS[2]),
                      ("1200.00", COLUMNS[3]), ("4800.00", COLUMNS[4])], 1020)

        mapping = map_columns(detect_borderless(page_of(lines), lines))

        assert mapping[4] == "total_amount"
        assert mapping[3] == "unit_price"

    def test_no_header_row_yields_no_mapping_rather_than_a_guess(self):
        # Positional guessing is a template by another name.
        lines = []
        lines += row([("Steel", COLUMNS[0]), ("7308", COLUMNS[1]), ("10", COLUMNS[2]),
                      ("1500.00", COLUMNS[3]), ("15000.00", COLUMNS[4])], 960)
        lines += row([("Welding", COLUMNS[0]), ("9988", COLUMNS[1]), ("4", COLUMNS[2]),
                      ("1200.00", COLUMNS[3]), ("4800.00", COLUMNS[4])], 1020)

        table = detect_borderless(page_of(lines), lines)

        assert table.header_row is None
        assert map_columns(table) == {}


class TestRuledTables:
    def ruled_image(self, rows=4, columns=5, width=1200, height=400):
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        row_positions = [int(index * height / rows) for index in range(rows + 1)]
        column_positions = [int(index * width / columns) for index in range(columns + 1)]
        for y in row_positions:
            cv2.line(image, (0, min(y, height - 1)), (width - 1, min(y, height - 1)), (0, 0, 0), 2)
        for x in column_positions:
            cv2.line(image, (min(x, width - 1), 0), (min(x, width - 1), height - 1), (0, 0, 0), 2)
        return image, row_positions, column_positions

    def test_drawn_rules_are_found(self):
        image, _, _ = self.ruled_image()

        horizontals, verticals = tables.find_rules(image)

        assert len(horizontals) >= 4
        assert len(verticals) >= 4

    def test_a_grid_is_recovered_from_the_rules(self):
        image, row_positions, column_positions = self.ruled_image()

        grid = tables.grid_from_rules(*tables.find_rules(image))

        assert grid is not None
        recovered_rows, recovered_columns = grid
        assert len(recovered_rows) == len(row_positions)
        assert len(recovered_columns) == len(column_positions)

    def test_text_is_placed_into_the_recovered_grid(self):
        image, row_positions, column_positions = self.ruled_image()
        # Put text in the centre of each cell of the second row.
        y = (row_positions[1] + row_positions[2]) / 2
        lines = [
            cell_line(text, (column_positions[index] + column_positions[index + 1]) / 2 - 20, y - 15)
            for index, text in enumerate(["Steel", "7308", "10", "1500.00", "15000.00"])
        ]

        rows, columns = tables.grid_from_rules(*tables.find_rules(image))
        table = tables.table_from_grid(page_of(lines, width=1200, height=400), rows, columns)

        assert table.method == METHOD_RULED
        assert table.grid()[1][0] == "Steel"
        assert table.grid()[1][4] == "15000.00"

    def test_an_image_with_no_rules_yields_no_grid(self):
        blank = np.full((400, 1200, 3), 255, dtype=np.uint8)

        assert tables.grid_from_rules(*tables.find_rules(blank)) is None

    def test_a_short_underline_is_not_treated_as_a_rule(self):
        # An underline under a heading spans a fraction of the table width;
        # treating it as a row boundary shatters the grid.
        image = np.full((400, 1200, 3), 255, dtype=np.uint8)
        cv2.line(image, (100, 50), (260, 50), (0, 0, 0), 2)
        cv2.line(image, (100, 120), (260, 120), (0, 0, 0), 2)

        assert tables.grid_from_rules(*tables.find_rules(image)) is None


class TestStrategySelection:
    def test_drawn_structure_is_preferred_over_inference(self):
        image = np.full((400, 1200, 3), 255, dtype=np.uint8)
        for y in (0, 100, 200, 300, 399):
            cv2.line(image, (0, y), (1199, y), (0, 0, 0), 2)
        for x in (0, 240, 480, 720, 960, 1199):
            cv2.line(image, (x, 0), (x, 399), (0, 0, 0), 2)
        lines = [cell_line(text, x + 20, 130) for text, x in
                 zip(["Steel", "7308", "10", "1500.00", "15000.00"], (0, 240, 480, 720, 960))]

        found = extract_tables(page_of(lines, width=1200, height=400), image=image)

        assert found and found[0].method == METHOD_RULED

    def test_borderless_inference_is_used_when_nothing_is_drawn(self):
        lines = invoice_table_lines()

        found = extract_tables(page_of(lines), region_lines=lines, image=None)

        assert found and found[0].method == METHOD_BORDERLESS

    def test_a_failed_rule_detection_falls_through_rather_than_losing_the_table(self):
        lines = invoice_table_lines()
        # A blank image finds no rules; the table must still come back.
        found = extract_tables(
            page_of(lines), region_lines=lines,
            image=np.full((3508, 2480, 3), 255, dtype=np.uint8),
        )

        assert found and found[0].method == METHOD_BORDERLESS

    def test_a_page_with_no_table_returns_nothing(self):
        lines = [cell_line("Thank you for your business.", 200, 900)]

        assert extract_tables(page_of(lines), region_lines=lines) == []


class TestMultiPageContinuation:
    def test_a_table_continuing_on_the_next_page_is_recognised(self):
        first = invoice_table_lines()
        second = row([("Painting Works", COLUMNS[0]), ("9954", COLUMNS[1]), ("6", COLUMNS[2]),
                      ("800.00", COLUMNS[3]), ("4800.00", COLUMNS[4])], 300)
        second += row([("Cleaning", COLUMNS[0]), ("9985", COLUMNS[1]), ("3", COLUMNS[2]),
                       ("500.00", COLUMNS[3]), ("1500.00", COLUMNS[4])], 360)
        second += row([("Disposal", COLUMNS[0]), ("9994", COLUMNS[1]), ("1", COLUMNS[2]),
                       ("900.00", COLUMNS[3]), ("900.00", COLUMNS[4])], 420)

        first_table = detect_borderless(page_of(first), first)
        second_table = detect_borderless(page_of(second), second)

        assert tables.continues(first_table, second_table)

    def test_an_unrelated_table_is_not_treated_as_a_continuation(self):
        first = invoice_table_lines()
        narrow = [200, 500, 800]
        second = row([("Tax", narrow[0]), ("Rate", narrow[1]), ("Amount", narrow[2])], 300)
        second += row([("CGST", narrow[0]), ("9%", narrow[1]), ("1782.00", narrow[2])], 360)
        second += row([("SGST", narrow[0]), ("9%", narrow[1]), ("1782.00", narrow[2])], 420)

        first_table = detect_borderless(page_of(first), first)
        second_table = detect_borderless(page_of(second), second)

        assert second_table is not None
        assert not tables.continues(first_table, second_table)
