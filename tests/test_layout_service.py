"""Layout analysis.

Pages here are built by placing text at explicit coordinates, so each test
states a layout a real vendor uses and asserts the analysis survives it.
That is the whole claim of this stage: it works without knowing the vendor.
"""
import pytest

from app.modules.ocr import layout_service
from app.modules.ocr.dto import SOURCE_NATIVE, OcrLine, OcrPage, split_line_into_words
from app.modules.ocr.geometry import BoundingBox
from app.modules.ocr.layout_service import (
    REGION_FOOTER,
    REGION_PARTIES,
    REGION_SUMMARY,
    REGION_TABLE,
    analyse_page,
)

CHARACTER_WIDTH = 11.0
LINE_HEIGHT = 26.0


def line(text, x, y, index=0):
    box = BoundingBox(x, y, x + len(text) * CHARACTER_WIDTH, y + LINE_HEIGHT)
    return OcrLine(
        text=text, box=box, confidence=1.0,
        words=split_line_into_words(text, box, 1.0),
        page_number=1, line_index=index,
    )


def page(lines, width=2480.0, height=3508.0):
    for index, item in enumerate(lines):
        item.line_index = index
    return OcrPage(
        page_number=1, width=width, height=height,
        lines=sorted(lines, key=lambda item: (item.box.y0, item.box.x0)),
        source=SOURCE_NATIVE,
    )


def pair_for(layout, label_fragment):
    matches = [
        candidate for candidate in layout.pairs
        if label_fragment.lower() in candidate.label.lower()
    ]
    return matches[0] if matches else None


class TestKeyValueBinding:
    def test_a_label_and_value_on_one_line_are_split(self):
        layout = analyse_page(page([line("Invoice No: INV-2026-0042", 200, 300)]))

        found = pair_for(layout, "Invoice No")
        assert found is not None
        assert found.value == "INV-2026-0042"

    def test_a_label_glued_to_its_value_with_no_space_is_still_split(self):
        # Word-level splitting cannot see this one, so the line-level
        # fallback has to catch it.
        layout = analyse_page(page([line("Invoice No:INV-2026-0042", 200, 300)]))

        found = pair_for(layout, "Invoice No")
        assert found is not None
        assert found.value == "INV-2026-0042"
        assert found.relation == "same-token"

    def test_a_line_yields_only_one_pair_per_label(self):
        """Word-level and line-level passes must not both claim the same
        label — a duplicate looks like corroboration to a scoring stage."""
        layout = analyse_page(page([line("Invoice No: INV-2026-0042", 200, 300)]))

        assert len([p for p in layout.pairs if p.label.lower() == "invoice no"]) == 1

    def test_a_value_far_to_the_right_of_its_label_is_bound(self):
        """The case flat-text matching cannot handle: 900px of whitespace
        between the label and its value."""
        layout = analyse_page(page([
            line("Invoice Number", 200, 300),
            line("INV-2026-0042", 1400, 300),
        ]))

        found = pair_for(layout, "Invoice Number")
        assert found is not None
        assert found.value == "INV-2026-0042"
        assert found.relation == "right"

    def test_a_value_stacked_under_its_header_is_bound(self):
        layout = analyse_page(page([
            line("Invoice Date", 1600, 300),
            line("14/08/2026", 1600, 332),
        ]))

        found = pair_for(layout, "Invoice Date")
        assert found is not None
        assert found.value == "14/08/2026"
        assert found.relation == "below"

    def test_two_label_value_pairs_on_one_line_both_survive(self):
        """A metadata row is one line. At line granularity only one pair can
        come out of it; word boxes are what make both recoverable."""
        layout = analyse_page(page([
            line("Invoice Date: 14/08/2026        Due Date: 13/09/2026", 200, 300),
        ]))

        invoice_date = pair_for(layout, "Invoice Date")
        due_date = pair_for(layout, "Due Date")
        assert invoice_date is not None and "14/08/2026" in invoice_date.value
        assert due_date is not None and "13/09/2026" in due_date.value

    def test_a_label_is_not_bound_to_a_value_on_the_far_side_of_the_page(self):
        # An unbound label beats a confidently wrong binding: the next stage
        # can fall back, but it cannot un-believe a value it was handed.
        layout = analyse_page(page([
            line("Notes", 200, 300),
            line("99,999.00", 2300, 3400),
        ]))

        found = pair_for(layout, "Notes")
        assert found is None or found.value != "99,999.00"

    def test_a_time_is_not_mistaken_for_a_label_value_pair(self):
        layout = analyse_page(page([line("Generated at 14:30 hrs", 200, 300)]))

        assert not any(candidate.label.strip() == "14" for candidate in layout.pairs)

    def test_a_bare_heading_with_no_value_produces_no_pair(self):
        layout = analyse_page(page([line("Terms and Conditions:", 200, 3000)]))

        found = pair_for(layout, "Terms and Conditions")
        assert found is None or found.value


class TestColumns:
    def test_a_two_column_party_block_is_split_into_two_columns(self):
        """The layout that defeats a line-level text extractor: seller and
        buyer side by side, collapsed into one string by flat extraction."""
        layout = analyse_page(page([
            line("Sold By: Alpha Steel Works", 150, 400),
            line("Bill To: Beta Constructions", 1400, 400),
            line("GSTIN: 29AAAAA0000A1Z5", 150, 432),
            line("GSTIN: 27BBBBB1111B2Z6", 1400, 432),
        ]))

        assert len(layout.columns) >= 2
        left = [column for column in layout.columns if column.x1 < 1300]
        right = [column for column in layout.columns if column.x0 > 1300]
        assert left and right

    def test_each_party_keeps_its_own_gstin(self):
        layout = analyse_page(page([
            line("Sold By: Alpha Steel Works", 150, 400),
            line("Bill To: Beta Constructions", 1400, 400),
            line("GSTIN: 29AAAAA0000A1Z5", 150, 432),
            line("GSTIN: 27BBBBB1111B2Z6", 1400, 432),
        ]))

        gstin_pairs = [candidate for candidate in layout.pairs if candidate.label.upper() == "GSTIN"]
        assert len(gstin_pairs) == 2
        values = {candidate.value for candidate in gstin_pairs}
        assert values == {"29AAAAA0000A1Z5", "27BBBBB1111B2Z6"}
        # And they are distinguishable by position, which is what lets the
        # next stage decide which belongs to the seller.
        left = min(gstin_pairs, key=lambda candidate: candidate.label_box.x0)
        assert left.value == "29AAAAA0000A1Z5"

    def test_a_single_column_page_reports_one_column(self):
        layout = analyse_page(page([
            line("TAX INVOICE", 200, 200),
            line("A single narrow column of text here", 200, 260),
            line("And another line beneath it", 200, 320),
        ]))

        assert len(layout.columns) == 1

    def test_word_spacing_is_not_mistaken_for_a_column_gutter(self):
        layout = analyse_page(page([
            line("Steel fabrication work as per approved drawing", 200, 400),
        ]))

        assert len(layout.columns) == 1


class TestBlocks:
    def test_blocks_split_where_vertical_whitespace_widens(self):
        layout = analyse_page(page([
            line("Sold By: Alpha Steel Works", 150, 400),
            line("Plot 42, Industrial Estate", 150, 432),
            # A clear gap, then a new block.
            line("Bill To: Beta Constructions", 150, 700),
            line("14 Marine Drive", 150, 732),
        ]))

        assert len(layout.blocks) >= 2
        first, second = layout.blocks[0], layout.blocks[1]
        assert "Alpha Steel Works" in first.text
        assert "Beta Constructions" in second.text

    def test_consecutive_lines_stay_in_one_block(self):
        layout = analyse_page(page([
            line("Alpha Steel Works", 150, 400),
            line("Plot 42, Industrial Estate", 150, 432),
            line("Pune 411001", 150, 464),
        ]))

        assert len(layout.blocks) == 1
        assert len(layout.blocks[0].lines) == 3


class TestRegionTyping:
    @pytest.fixture
    def invoice(self):
        return page([
            line("ALPHA STEEL WORKS", 900, 120),
            line("TAX INVOICE", 1050, 180),

            line("Bill To: Beta Constructions Pvt Ltd", 150, 500),
            line("14 Marine Drive, Mumbai 400001", 150, 532),
            line("GSTIN: 27BBBBB1111B2Z6", 150, 564),

            line("Description        HSN     Qty     Rate      Amount", 150, 900),
            line("Steel Fabrication  7308     10  1500.00   15000.00", 150, 940),
            line("Welding Services   9988      4  1200.00    4800.00", 150, 980),

            line("Sub Total                              19800.00", 1300, 1300),
            line("CGST 9%                                 1782.00", 1300, 1340),
            line("SGST 9%                                 1782.00", 1300, 1380),
            line("Grand Total                            23364.00", 1300, 1420),

            line("Bank: State Bank of India", 150, 2400),
            line("Account No: 1234567890  IFSC: SBIN0001234", 150, 2440),
            line("Terms and Conditions apply. Subject to Pune jurisdiction.", 150, 2480),
        ])

    def test_the_item_table_is_located_by_its_header_row(self, invoice):
        layout = analyse_page(invoice)

        table = layout.region(REGION_TABLE)
        assert table is not None
        assert "Steel Fabrication" in table.text
        assert "HSN" in table.text

    def test_the_summary_block_is_separated_from_the_table(self, invoice):
        layout = analyse_page(invoice)

        summary = layout.region(REGION_SUMMARY)
        assert summary is not None
        assert "Grand Total" in summary.text
        # The word "Amount" appears in the table header too; position is what
        # separates the two, which is why content alone is not enough.
        assert "Steel Fabrication" not in summary.text

    def test_the_party_block_is_typed(self, invoice):
        layout = analyse_page(invoice)

        parties = layout.region(REGION_PARTIES)
        assert parties is not None
        assert "Beta Constructions" in parties.text

    def test_the_footer_is_typed(self, invoice):
        layout = analyse_page(invoice)

        footer = layout.region(REGION_FOOTER)
        assert footer is not None
        assert "IFSC" in footer.text or "Bank" in footer.text

    def test_every_line_lands_in_exactly_one_region(self, invoice):
        layout = analyse_page(invoice)

        assigned = [line_item for region in layout.regions for line_item in region.lines]
        assert len(assigned) == len([item for item in invoice.lines if item.text.strip()])
        assert len({id(item) for item in assigned}) == len(assigned)

    def test_an_empty_page_yields_no_regions_rather_than_raising(self):
        layout = analyse_page(page([]))

        assert layout.regions == []
        assert layout.pairs == []


class TestTableIndependenceFromWording:
    @pytest.mark.parametrize(
        "header",
        [
            "Description        HSN     Qty     Rate      Amount",
            "Particulars        SAC     Quantity  Price   Total",
            "Item Description   HSN Code  Units   Unit Price  Net Amount",
        ],
    )
    def test_the_table_is_found_whatever_the_vendor_calls_the_columns(self, header):
        """The claim this whole stage rests on: no vendor-specific template.
        Three vendors' wordings, one anchor — a row of column-ish words."""
        layout = analyse_page(page([
            line("TAX INVOICE", 1050, 180),
            line(header, 150, 900),
            line("Steel Fabrication  7308     10  1500.00   15000.00", 150, 940),
        ]))

        table = layout.region(REGION_TABLE)
        assert table is not None
        assert "Steel Fabrication" in table.text

    def test_a_page_with_no_table_reports_none(self):
        layout = analyse_page(page([
            line("TAX INVOICE", 1050, 180),
            line("This document confirms receipt of payment in full.", 150, 900),
        ]))

        assert layout.region(REGION_TABLE) is None
