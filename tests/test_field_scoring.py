"""Candidate scoring: how a field's value is chosen when several compete.

The word `Total` appears on an invoice in the table header, in a line item,
and three times in the summary block. First-match-wins picks whichever the
reading order reaches first, which is a coin flip. These tests set up exactly
those competitions and pin which reading wins and why.
"""
from datetime import date
from decimal import Decimal

import pytest

from app.modules.invoices.parsers import field_scoring
from app.modules.invoices.parsers.field_scoring import (
    Candidate,
    extract_fields,
    parse_value,
    resolve,
)
from app.modules.ocr.dto import SOURCE_OCR, OcrLine, OcrPage, split_line_into_words
from app.modules.ocr.geometry import BoundingBox
from app.modules.ocr.layout_service import analyse_page

CHARACTER_WIDTH = 12.0
LINE_HEIGHT = 30.0


def line(text, x, y, confidence=0.98):
    box = BoundingBox(x, y, x + len(text) * CHARACTER_WIDTH, y + LINE_HEIGHT)
    return OcrLine(text=text, box=box, confidence=confidence,
                   words=split_line_into_words(text, box, confidence))


def page_of(lines, width=2480.0, height=3508.0):
    ordered = sorted(lines, key=lambda item: (item.box.y0, item.box.x0))
    for index, item in enumerate(ordered):
        item.line_index = index
    return OcrPage(page_number=1, width=width, height=height, lines=ordered, source=SOURCE_OCR)


def full_invoice_page():
    """A complete invoice with every ambiguity a real one has."""
    return page_of([
        line("ALPHA STEEL WORKS", 900, 120),
        line("TAX INVOICE", 1050, 200),

        line("Invoice No: INV-2026-0042", 150, 400),
        line("Invoice Date: 14/08/2026", 150, 450),
        line("Due Date: 13/09/2026", 150, 500),

        line("Bill To: Beta Constructions Pvt Ltd", 1400, 400),
        line("GSTIN: 27BBBBB1111B2Z6", 1400, 450),

        line("Description        HSN     Qty     Rate      Amount", 150, 900),
        line("Steel Fabrication  7308     10  1500.00   15000.00", 150, 960),
        line("Welding Services   9988      4  1200.00    4800.00", 150, 1020),

        line("Sub Total                              19800.00", 1300, 1400),
        line("CGST 9%                                 1782.00", 1300, 1450),
        line("SGST 9%                                 1782.00", 1300, 1500),
        line("Round Off                                  -0.40", 1300, 1550),
        line("Grand Total                            23363.60", 1300, 1600),
        line("Amount in Words: Twenty Three Thousand Only", 150, 1680),
    ])


@pytest.fixture
def resolved():
    return extract_fields(analyse_page(full_invoice_page()))


class TestValueTyping:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("15000.00", Decimal("15000.00")),
            ("1,92,407.04", Decimal("192407.04")),      # Indian lakh grouping
            ("192,407.04", Decimal("192407.04")),       # Western grouping
            ("Rs. 1,500.00", Decimal("1500.00")),
            ("₹ 1,500.00", Decimal("1500.00")),
            ("(500.00)", Decimal("-500.00")),           # parenthesised negative
            ("-0.40", Decimal("-0.40")),
        ],
    )
    def test_money_in_every_format_an_invoice_uses(self, text, expected):
        assert parse_value("money", text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("14/08/2026", date(2026, 8, 14)),
            ("14-08-2026", date(2026, 8, 14)),
            ("2026-08-14", date(2026, 8, 14)),
            ("14 Aug 2026", date(2026, 8, 14)),
            ("Aug 14, 2026", date(2026, 8, 14)),
        ],
    )
    def test_dates_in_every_format_an_invoice_uses(self, text, expected):
        assert parse_value("date", text) == expected

    def test_an_ambiguous_date_is_read_day_first(self):
        # India-only deployment: 03/04/2026 is 3 April, never 4 March.
        assert parse_value("date", "03/04/2026") == date(2026, 4, 3)

    def test_a_wildly_out_of_range_year_is_rejected_as_a_misread(self):
        assert parse_value("date", "14/08/1907") is None

    def test_a_gstin_is_validated_structurally(self):
        assert parse_value("gstin", "GSTIN: 29ABCDE1234F1Z5") == "29ABCDE1234F1Z5"
        assert parse_value("gstin", "29ABCDE1234F1Z") is None

    def test_amount_in_words_never_parses_as_money(self):
        assert parse_value("money", "Amount in Words: Twenty Three Thousand Only") is None

    def test_non_values_return_none_so_the_candidate_is_dropped(self):
        assert parse_value("money", "Terms and Conditions") is None
        assert parse_value("date", "Payment Terms") is None


class TestCompetitionBetweenReadings:
    def test_grand_total_beats_the_line_item_amounts(self, resolved):
        """`Amount` is a table header and `15000.00` a line item; both look
        like a total to a naive matcher. Region priors settle it."""
        assert resolved["total_amount"].winner.parsed_value == Decimal("23363.60")

    def test_sub_total_and_grand_total_do_not_collide(self, resolved):
        assert resolved["subtotal"].winner.parsed_value == Decimal("19800.00")
        assert resolved["total_amount"].winner.parsed_value == Decimal("23363.60")

    def test_each_gst_component_gets_its_own_amount(self, resolved):
        assert resolved["cgst_amount"].winner.parsed_value == Decimal("1782.00")
        assert resolved["sgst_amount"].winner.parsed_value == Decimal("1782.00")

    def test_the_round_off_is_not_read_as_a_total(self, resolved):
        assert resolved["round_off"].winner.parsed_value == Decimal("-0.40")

    def test_invoice_and_due_dates_are_not_swapped(self, resolved):
        assert resolved["invoice_date"].winner.parsed_value == date(2026, 8, 14)
        assert resolved["due_date"].winner.parsed_value == date(2026, 9, 13)

    def test_the_invoice_number_is_found(self, resolved):
        assert resolved["invoice_number"].winner.parsed_value == "INV-2026-0042"

    def test_amount_in_words_never_becomes_a_money_field(self, resolved):
        for name, resolution in resolved.items():
            if name.endswith("_amount") or name in ("subtotal", "round_off"):
                assert "Twenty Three Thousand" not in str(resolution.winner.raw_value)


class TestScoringMechanics:
    def make(self, **overrides):
        defaults = dict(
            field="total_amount", raw_value="1000.00", parsed_value=Decimal("1000.00"),
            relation="right", box=BoundingBox(0, 0, 100, 30), label_score=1.0,
            ocr_confidence=1.0, region=None, distance=0.0,
        )
        defaults.update(overrides)
        return Candidate(**defaults)

    def test_an_inline_label_beats_a_value_merely_to_the_right(self):
        inline = field_scoring.score_candidate(self.make(relation="same-token"))
        right = field_scoring.score_candidate(self.make(relation="right"))
        below = field_scoring.score_candidate(self.make(relation="below"))

        assert inline > right > below

    def test_a_labelled_reading_beats_a_positional_guess(self):
        labelled = field_scoring.score_candidate(self.make(relation="right", label_score=1.0))
        positional = field_scoring.score_candidate(self.make(relation="positional", label_score=0.0))

        assert labelled > positional

    def test_a_nearer_value_outscores_a_distant_one(self):
        near = field_scoring.score_candidate(self.make(distance=20))
        far = field_scoring.score_candidate(self.make(distance=900))

        assert near > far

    def test_low_ocr_confidence_drags_the_score_down(self):
        """A value the recogniser doubted is a doubtful value, however
        perfect its label was."""
        confident = field_scoring.score_candidate(self.make(ocr_confidence=0.99))
        doubtful = field_scoring.score_candidate(self.make(ocr_confidence=0.35))

        assert doubtful < confident * 0.5

    def test_the_region_prior_separates_a_summary_total_from_a_table_one(self):
        from app.modules.ocr.layout_service import REGION_SUMMARY, REGION_TABLE

        in_summary = field_scoring.score_candidate(self.make(region=REGION_SUMMARY))
        in_table = field_scoring.score_candidate(self.make(region=REGION_TABLE))

        assert in_summary > in_table * 2

    def test_quality_ignores_proximity_and_region_but_score_does_not(self):
        """Ranking and believing are different questions.

        Proximity and region priors say how likely a reading was to *win*
        against its rivals, not how likely it is to be *right*. A summary
        amount right-aligned far from its label is not suspicious — that is
        what a summary block looks like — and charging it for the distance
        sends perfectly extracted invoices to manual review.
        """
        from app.modules.ocr.layout_service import REGION_SUMMARY

        near = self.make(distance=10)
        far = self.make(distance=900, region=REGION_SUMMARY)
        field_scoring.score_candidate(near)
        field_scoring.score_candidate(far)

        assert near.score != far.score, "ranking must still respond to position"
        assert near.quality == far.quality, "belief must not depend on position"

    def test_quality_still_responds_to_evidence_quality(self):
        strong = self.make(relation="same-token", label_score=1.0, ocr_confidence=1.0)
        weak = self.make(relation="positional", label_score=0.0, ocr_confidence=0.4)
        field_scoring.score_candidate(strong)
        field_scoring.score_candidate(weak)

        assert strong.quality > weak.quality

    def test_scores_stay_within_range(self):
        assert 0.0 <= field_scoring.score_candidate(self.make()) <= 1.0
        assert 0.0 <= field_scoring.score_candidate(
            self.make(relation="same-token", region="SUMMARY", ocr_confidence=1.0)
        ) <= 1.0


class TestAgreementAndMargin:
    def make(self, value, score_hint, **overrides):
        defaults = dict(
            field="total_amount", raw_value=value, parsed_value=Decimal(value),
            relation=score_hint, box=BoundingBox(0, 0, 100, 30), label_score=1.0,
            ocr_confidence=1.0,
        )
        defaults.update(overrides)
        return Candidate(**defaults)

    def test_two_routes_agreeing_on_a_value_reinforce_it(self):
        agreeing = resolve([
            self.make("1000.00", "right"),
            self.make("1000.00", "below"),
            self.make("2000.00", "right"),
        ])

        assert agreeing["total_amount"].winner.parsed_value == Decimal("1000.00")
        assert "agree" in " ".join(agreeing["total_amount"].winner.reasons)

    def test_agreement_cannot_let_repeated_weak_readings_beat_one_strong_one(self):
        result = resolve([
            self.make("1000.00", "positional", label_score=0.0),
            self.make("1000.00", "positional", label_score=0.0),
            self.make("1000.00", "positional", label_score=0.0),
            self.make("2000.00", "same-token", label_score=1.0),
        ])

        assert result["total_amount"].winner.parsed_value == Decimal("2000.00")

    def test_a_clear_winner_reports_a_wide_margin(self):
        result = resolve([
            self.make("1000.00", "same-token"),
            self.make("2000.00", "positional", label_score=0.0),
        ])

        assert result["total_amount"].margin > 0.3
        assert not result["total_amount"].contested

    def test_a_close_call_is_flagged_as_contested(self):
        """The thing first-match-wins can never tell you: that this field was
        nearly a different value, and a human should look."""
        result = resolve([
            self.make("1000.00", "right"),
            self.make("2000.00", "right", ocr_confidence=0.97),
        ])

        assert result["total_amount"].contested

    def test_an_uncontested_single_candidate_has_a_full_margin(self):
        result = resolve([self.make("1000.00", "same-token")])

        assert result["total_amount"].margin == 1.0
        assert result["total_amount"].runner_up is None


class TestExplainability:
    def test_every_decision_carries_its_reasoning(self, resolved):
        """A reviewer asking 'why that value' must get an answer, and a
        threshold must be retunable against real decisions."""
        explanation = resolved["total_amount"].winner.explain()

        assert explanation["value"]
        assert explanation["reasons"]
        assert explanation["score"] > 0
        assert explanation["box"] is not None, "evidence must be locatable on the page"

    def test_the_evidence_box_points_at_the_value_not_the_whole_page(self, resolved):
        box = resolved["total_amount"].winner.box

        assert box.width < 2000
        assert box.height < 100


class TestPositionalFallback:
    def test_a_total_is_still_found_when_its_label_was_destroyed(self):
        """The scan whose summary labels OCR mangled beyond matching. The
        bottom-most amount in the summary block is still the total."""
        page = page_of([
            line("TAX INVOICE", 1050, 200),
            line("Description   Qty   Rate    Amount", 150, 900),
            line("Steel Work     10  1500.00  15000.00", 150, 960),
            line("5ub T0tl                    19800.00", 1300, 1400),
            line("C65T 9%                      1782.00", 1300, 1450),
            line("6rnd T0tl                   23364.00", 1300, 1600),
        ])

        result = extract_fields(analyse_page(page))

        assert "total_amount" in result
        assert result["total_amount"].winner.parsed_value == Decimal("23364.00")

    def test_a_positional_candidate_records_why_it_was_proposed(self):
        page = page_of([
            line("Sub Total    19800.00", 1300, 1400),
            line("Grand Total  23364.00", 1300, 1600),
        ])

        result = extract_fields(analyse_page(page))
        winner = result["total_amount"].winner

        assert winner.parsed_value == Decimal("23364.00")
        assert winner.reasons
