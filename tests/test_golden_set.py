"""Golden-set regression: per-field accuracy across a corpus of layouts.

This is the test that stops the extractor getting quietly worse. Every other
test in the suite pins one behaviour; this one measures the thing anyone
actually cares about — how often the right value comes out — and fails when
that number drops.

**Accuracy is reported per field, not per document.** "78% of invoices were
perfect" is not actionable; "invoice_number 100%, total_amount 83%,
customer_gstin 50%" says exactly where to spend the next hour. A document
score also punishes a wholly correct extraction for one missing minor field,
which distorts the incentive to improve.

The thresholds below are floors, not targets. They are set at the level the
extractor currently clears, so that a change which drops a field's accuracy
fails the build rather than being noticed months later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from decimal import Decimal
from pathlib import Path

import pytest

from app.modules.invoices.services import structured_extraction_service as structured
from tests.golden.corpus import build_corpus, load_real_corpus

REAL_CORPUS_DIR = Path(__file__).parent / "golden" / "real"

# Money is compared to the rupee: an extractor that reads 24544.00 as 24544
# is correct, and treating that as a miss would measure the formatter.
MONEY_TOLERANCE = Decimal("0.05")

# Per-field floors. A field below its floor fails the build.
#
# All at 1.00 because the synthetic corpus is read from PDF text layers, so
# the result is deterministic — there is no OCR noise to leave headroom for,
# and a floor set below what the code achieves protects nothing.
#
# **Adding a real scanned invoice will breach these, and that is correct.**
# A new document the extractor gets wrong should fail the build until someone
# looks at it; the choice then is to fix the extractor or to re-baseline
# deliberately, and both are better than a floor quietly set low enough that
# nobody ever has to choose.
ACCURACY_FLOORS: dict[str, float] = {
    "invoice_number": 1.00,
    "invoice_date": 1.00,
    "due_date": 1.00,
    "total_amount": 1.00,
    "subtotal": 1.00,
    "taxable_amount": 1.00,
    "tax_amount": 1.00,
    "cgst_amount": 1.00,
    "sgst_amount": 1.00,
    "igst_amount": 1.00,
    "round_off": 1.00,
    "vendor_gstin": 1.00,
    "customer_gstin": 1.00,
    "vendor_name": 1.00,
    "customer_name": 1.00,
    "line_item_count": 1.00,
}

# Documents that must come out with no validation errors at all. An
# arithmetic error on a synthetic invoice that reconciles by construction
# means the extraction is wrong, not the invoice.
MUST_VALIDATE_CLEANLY = 1.00


@dataclass
class FieldOutcome:
    field: str
    document: str
    expected: object
    actual: object
    correct: bool


@dataclass
class AccuracyReport:
    outcomes: list[FieldOutcome] = dataclass_field(default_factory=list)
    clean_documents: int = 0
    total_documents: int = 0

    def by_field(self) -> dict[str, tuple[int, int]]:
        """{field: (correct, total)} over documents that carry the field."""
        tallies: dict[str, list[int]] = {}
        for outcome in self.outcomes:
            entry = tallies.setdefault(outcome.field, [0, 0])
            entry[1] += 1
            if outcome.correct:
                entry[0] += 1
        return {name: (values[0], values[1]) for name, values in tallies.items()}

    def accuracy(self, field_name: str) -> float | None:
        correct, total = self.by_field().get(field_name, (0, 0))
        return correct / total if total else None

    def failures(self) -> list[FieldOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.correct]

    def render(self) -> str:
        lines = [
            "",
            f"{'field':22} {'accuracy':>10}  {'n':>4}",
            "-" * 40,
        ]
        for name, (correct, total) in sorted(self.by_field().items()):
            lines.append(f"{name:22} {correct / total:>9.0%}  {total:>4}")
        lines.append("-" * 40)
        lines.append(
            f"{'documents validating':22} "
            f"{self.clean_documents / self.total_documents:>9.0%}  {self.total_documents:>4}"
        )
        if self.failures():
            lines.append("")
            lines.append("misses:")
            for outcome in self.failures():
                lines.append(
                    f"  {outcome.document}/{outcome.field}: "
                    f"expected {outcome.expected!r}, got {outcome.actual!r}"
                )
        return "\n".join(lines)


def _matches(field_name: str, expected: object, actual: object) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False

    if field_name == "line_item_count":
        return int(expected) == int(actual)

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(Decimal(str(actual)) - Decimal(str(expected))) <= MONEY_TOLERANCE
        except (ArithmeticError, ValueError, TypeError):
            return False

    if field_name.endswith("_date"):
        return str(actual) == str(expected)

    # Names and identifiers: compared case- and whitespace-insensitively.
    # A vendor name recovered as "ALPHA STEEL WORKS" is not a miss.
    return " ".join(str(actual).split()).casefold() == " ".join(str(expected).split()).casefold()


def _actual_for(field_name: str, result) -> object:
    parsed = result.parsed
    if field_name == "line_item_count":
        return len(parsed.line_items)
    if field_name.startswith("vendor_"):
        return getattr(parsed.vendor, field_name[len("vendor_"):], None)
    if field_name.startswith("customer_"):
        return getattr(parsed.customer, field_name[len("customer_"):], None)
    value = getattr(parsed, field_name, None)
    return value.isoformat() if hasattr(value, "isoformat") else value


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> AccuracyReport:
    """Run the whole corpus once and score it."""
    directory = tmp_path_factory.mktemp("golden")
    corpus = build_corpus(directory) + load_real_corpus(REAL_CORPUS_DIR)

    accuracy = AccuracyReport(total_documents=len(corpus))
    for path, expected in corpus:
        result = structured.extract(path)
        if result.validation.is_clean:
            accuracy.clean_documents += 1
        for field_name, want in expected.items():
            got = _actual_for(field_name, result)
            accuracy.outcomes.append(
                FieldOutcome(field_name, path.stem, want, got, _matches(field_name, want, got))
            )
    return accuracy


class TestAccuracy:
    @pytest.mark.parametrize("field_name", sorted(ACCURACY_FLOORS))
    def test_each_field_clears_its_floor(self, field_name, report):
        measured = report.accuracy(field_name)
        if measured is None:
            pytest.skip(f"no corpus document carries {field_name}")

        floor = ACCURACY_FLOORS[field_name]
        misses = [
            f"{outcome.document}: expected {outcome.expected!r}, got {outcome.actual!r}"
            for outcome in report.failures() if outcome.field == field_name
        ]
        assert measured >= floor, (
            f"{field_name} accuracy fell to {measured:.0%} (floor {floor:.0%})\n"
            + "\n".join(f"  {miss}" for miss in misses)
        )

    def test_most_documents_reconcile(self, report):
        """A synthetic invoice reconciles by construction, so a validation
        error on one means the extraction is wrong, not the invoice."""
        rate = report.clean_documents / report.total_documents
        assert rate >= MUST_VALIDATE_CLEANLY, (
            f"only {rate:.0%} of the corpus validated cleanly "
            f"(floor {MUST_VALIDATE_CLEANLY:.0%})"
        )

    def test_the_report_is_printed_for_the_record(self, report, capsys):
        """Not an assertion — the numbers themselves are the deliverable, and
        a run that only says "passed" tells nobody where the extractor is
        weak."""
        with capsys.disabled():
            print(report.render())


class TestCorpusHealth:
    def test_the_corpus_covers_distinct_layouts(self):
        """A corpus of one layout repeated cannot test the claim that the
        extractor needs no template."""
        from tests.golden.corpus import GOLDEN_INVOICES

        assert len(GOLDEN_INVOICES) >= 5
        assert len({invoice.description for invoice in GOLDEN_INVOICES}) == len(GOLDEN_INVOICES)

    def test_every_golden_invoice_states_its_expected_total(self):
        from tests.golden.corpus import GOLDEN_INVOICES

        for invoice in GOLDEN_INVOICES:
            assert "total_amount" in invoice.expected, invoice.name

    def test_real_invoices_are_picked_up_without_registration(self, tmp_path):
        """Adding a real invoice to the set must take a minute, or it will
        never happen."""
        import fitz

        document = fitz.open()
        document.new_page().insert_text((72, 72), "Tax Invoice")
        document.save(str(tmp_path / "vendor_a.pdf"))
        document.close()
        (tmp_path / "vendor_a.expected.json").write_text(
            json.dumps({"invoice_number": "X-1"}), encoding="utf-8"
        )

        found = load_real_corpus(tmp_path)

        assert len(found) == 1
        assert found[0][1]["invoice_number"] == "X-1"

    def test_a_document_without_expectations_is_ignored(self, tmp_path):
        (tmp_path / "stray.pdf").write_bytes(b"%PDF-1.4 not really")

        assert load_real_corpus(tmp_path) == []

    def test_a_missing_real_corpus_directory_is_not_an_error(self, tmp_path):
        assert load_real_corpus(tmp_path / "nope") == []
