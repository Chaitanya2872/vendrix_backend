"""Measure the model two ways, per format.

Line classification accuracy on its own is misleading: it counts a document
where every field was found but the total was misread as a near-perfect
result. So this reports both layers.

  line level    did the model point at the right line? (precision/recall/F1
                per label, macro-averaged)
  field level   did the pipeline end up with the right *value*? (exact match
                against the generator's ground truth)

And it splits the field-level miss into two causes, which have different
fixes:

  model miss    no line was proposed for the field, or the proposed line was
                the wrong one              -> the classifier needs work
  ocr miss      the right line was proposed, but the characters on it were
                already corrupt when the model saw them
                                           -> no classifier change can help;
                                              this is an OCR/preprocessing
                                              ceiling

Everything is reported per format because that is the question being asked:
the same invoice reaches the parser as crisp PDF text or as OCR of a creased
phone photo, and one number averaged over both describes neither.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Iterable

from sklearn.metrics import classification_report, f1_score

from .corpus import LabelledLine
from .extract import extract_fields
from .features import LineContext
from .labels import OCR_FORMATS, OTHER, SINGLE_VALUE_LABELS, SPEC_BY_LABEL
from .synth import generate_invoice

# A recovered party name rarely matches character-for-character after OCR
# ("Pvt Ltd" vs "Pvt Ltd."), and calling that a failure would understate the
# pipeline. Identifiers and amounts get no such latitude — a total that is
# nearly right is wrong.
NAME_MATCH_THRESHOLD = 0.90


@dataclass
class FieldOutcome:
    expected: int = 0
    extracted: int = 0
    correct: int = 0
    ocr_ceiling: int = 0  # expected fields whose value survived extraction intact

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.extracted if self.extracted else 0.0

    @property
    def attainable_recall(self) -> float:
        """Recall measured only over fields whose value OCR did not destroy —
        the best any classifier could have done on this input."""
        return self.correct / self.ocr_ceiling if self.ocr_ceiling else 0.0


def _document_seed(doc_id: str) -> int:
    return int(doc_id.rsplit("-", 1)[-1])


def to_contexts(rows: list[LabelledLine]) -> list[LineContext]:
    return [
        LineContext(
            text=row.text,
            previous_text=row.previous_text,
            next_text=row.next_text,
            line_index=row.line_index,
            line_count=row.line_count,
        )
        for row in rows
    ]


def group_by_rendering(rows: Iterable[LabelledLine]) -> dict[tuple[str, str], list[LabelledLine]]:
    """One group per (document, format) — the unit the field extractor runs
    on."""
    grouped: dict[tuple[str, str], list[LabelledLine]] = defaultdict(list)
    for row in rows:
        grouped[(row.doc_id, row.fmt)].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row.line_index)
    return dict(grouped)


def line_predictions(model, rows: list[LabelledLine]) -> tuple[list[str], list[str], list[str]]:
    """(truth, predicted, format) per line — the raw material both the single
    split and the cross-validated report are built from."""
    predicted = model.predict(to_contexts(rows))
    return [row.label for row in rows], list(predicted), [row.fmt for row in rows]


def line_level_report_from(truth: list[str], predicted: list[str], formats: list[str]) -> dict:
    """Per-label precision/recall/F1, plus the same split by format."""
    overall = classification_report(truth, predicted, output_dict=True, zero_division=0)
    # Macro-F1 over the *field* labels only. Including OTHER — the majority of
    # every document — would let a model that finds nothing still look
    # respectable.
    field_labels = [label for label in SINGLE_VALUE_LABELS if label in set(truth)]
    field_macro_f1 = f1_score(truth, predicted, labels=field_labels, average="macro", zero_division=0)

    by_format: dict[str, dict] = {}
    for fmt in sorted(set(formats)):
        indices = [index for index, value in enumerate(formats) if value == fmt]
        if not indices:
            continue
        format_truth = [truth[index] for index in indices]
        format_predicted = [predicted[index] for index in indices]
        present = [label for label in field_labels if label in set(format_truth)]
        by_format[fmt] = {
            "lines": len(indices),
            "field_macro_f1": round(
                f1_score(format_truth, format_predicted, labels=present, average="macro", zero_division=0), 4
            ),
            "accuracy": round(
                sum(a == b for a, b in zip(format_truth, format_predicted)) / len(indices), 4
            ),
        }

    return {
        "field_macro_f1": round(float(field_macro_f1), 4),
        "accuracy": round(overall["accuracy"], 4),
        "per_label": {
            label: {
                "precision": round(overall[label]["precision"], 4),
                "recall": round(overall[label]["recall"], 4),
                "f1": round(overall[label]["f1-score"], 4),
                "support": int(overall[label]["support"]),
            }
            for label in sorted(set(truth) | {OTHER})
            if label in overall
        },
        "per_format": by_format,
    }


def line_level_report(model, rows: list[LabelledLine]) -> dict:
    return line_level_report_from(*line_predictions(model, rows))


def _values_match(kind: str, extracted: object, expected: object) -> bool:
    if extracted is None or expected is None:
        return False
    if kind == "money":
        return isinstance(extracted, Decimal) and extracted == expected
    if kind == "date":
        return extracted == expected
    if kind == "gstin":
        return str(extracted).upper() == str(expected).upper()
    if kind == "identifier":
        clean = lambda value: "".join(character for character in str(value).upper() if character.isalnum())
        return clean(extracted) == clean(expected)
    normalise = lambda value: " ".join(str(value).lower().split())
    return SequenceMatcher(None, normalise(extracted), normalise(expected)).ratio() >= NAME_MATCH_THRESHOLD


def _expected_values(invoice) -> dict[str, object]:
    """Ground truth as typed values, keyed by label."""
    values: dict[str, object] = {
        "INVOICE_NUMBER": invoice.invoice_number,
        "VENDOR_NAME": invoice.vendor_name,
        "VENDOR_GSTIN": invoice.vendor_gstin,
        "CUSTOMER_NAME": invoice.customer_name,
        "CUSTOMER_GSTIN": invoice.customer_gstin,
    }
    values.update(invoice.numeric_truth())
    values.update(invoice.date_truth())
    return values


def accumulate_field_outcomes(
    model,
    rows: list[LabelledLine],
    per_format: dict[str, dict[str, FieldOutcome]],
    documents_per_format: dict[str, int],
) -> None:
    """Run the field extractor over every rendering in `rows` and fold the
    outcomes into the running totals.

    Kept separate from reporting so a cross-validated run can pour several
    folds' results into one set of counts. That matters here: OCR is expensive
    enough that each raster format only gets ~30 documents in the whole
    corpus, and a single holdout split would leave ~8 of them to be measured
    on — a number too small to say anything with.
    """
    for (doc_id, fmt), group in group_by_rendering(rows).items():
        invoice = generate_invoice(_document_seed(doc_id))
        expected = _expected_values(invoice)
        # Only fields this particular document actually carries can be
        # expected of the extractor.
        present_labels = set(invoice.ground_truth())
        # Whether the value was still intact in the extracted text at all.
        exact_by_label = {row.label: row.value_exact for row in group}

        found = extract_fields(model, to_contexts(group))
        documents_per_format[fmt] += 1

        for label in SINGLE_VALUE_LABELS:
            outcome = per_format[fmt][label]
            if label in present_labels:
                outcome.expected += 1
                if exact_by_label.get(label, False):
                    outcome.ocr_ceiling += 1
            if label in found:
                outcome.extracted += 1
                if label in present_labels and _values_match(
                    SPEC_BY_LABEL[label].kind, found[label].value, expected.get(label)
                ):
                    outcome.correct += 1


def new_field_accumulator() -> tuple[dict[str, dict[str, FieldOutcome]], dict[str, int]]:
    return defaultdict(lambda: defaultdict(FieldOutcome)), defaultdict(int)


def field_report_from(
    per_format: dict[str, dict[str, FieldOutcome]],
    documents_per_format: dict[str, int],
) -> dict:
    report: dict = {"per_format": {}, "per_label": {}}
    totals: dict[str, FieldOutcome] = defaultdict(FieldOutcome)

    for fmt in sorted(per_format):
        outcomes = per_format[fmt]
        expected_total = sum(outcome.expected for outcome in outcomes.values())
        correct_total = sum(outcome.correct for outcome in outcomes.values())
        extracted_total = sum(outcome.extracted for outcome in outcomes.values())
        ceiling_total = sum(outcome.ocr_ceiling for outcome in outcomes.values())
        report["per_format"][fmt] = {
            "documents": documents_per_format[fmt],
            "uses_ocr": fmt in OCR_FORMATS,
            "field_recall": round(correct_total / expected_total, 4) if expected_total else 0.0,
            "field_precision": round(correct_total / extracted_total, 4) if extracted_total else 0.0,
            "ocr_intact_rate": round(ceiling_total / expected_total, 4) if expected_total else 0.0,
            "recall_vs_ocr_ceiling": round(correct_total / ceiling_total, 4) if ceiling_total else 0.0,
            "per_label": {
                label: {
                    "expected": outcome.expected,
                    "correct": outcome.correct,
                    "recall": round(outcome.recall, 4),
                    "precision": round(outcome.precision, 4),
                }
                for label, outcome in sorted(outcomes.items())
                if outcome.expected
            },
        }
        for label, outcome in outcomes.items():
            total = totals[label]
            total.expected += outcome.expected
            total.extracted += outcome.extracted
            total.correct += outcome.correct
            total.ocr_ceiling += outcome.ocr_ceiling

    report["per_label"] = {
        label: {
            "expected": outcome.expected,
            "correct": outcome.correct,
            "recall": round(outcome.recall, 4),
            "precision": round(outcome.precision, 4),
            "recall_vs_ocr_ceiling": round(outcome.attainable_recall, 4),
        }
        for label, outcome in sorted(totals.items())
        if outcome.expected
    }
    grand_expected = sum(outcome.expected for outcome in totals.values())
    grand_correct = sum(outcome.correct for outcome in totals.values())
    grand_ceiling = sum(outcome.ocr_ceiling for outcome in totals.values())
    report["field_recall"] = round(grand_correct / grand_expected, 4) if grand_expected else 0.0
    report["recall_vs_ocr_ceiling"] = round(grand_correct / grand_ceiling, 4) if grand_ceiling else 0.0
    return report


def field_level_report(model, rows: list[LabelledLine]) -> dict:
    """End-to-end on a single test split: run the field extractor per
    rendering and compare values against what the generator wrote."""
    per_format, documents_per_format = new_field_accumulator()
    accumulate_field_outcomes(model, rows, per_format, documents_per_format)
    return field_report_from(per_format, documents_per_format)


def format_report(line_report: dict, field_report: dict) -> str:
    """A readable summary for the terminal — the numbers someone actually
    needs to decide whether this model is fit to deploy."""
    lines: list[str] = []
    lines.append("LINE CLASSIFICATION")
    lines.append(f"  field macro-F1 : {line_report['field_macro_f1']:.4f}")
    lines.append(f"  line accuracy  : {line_report['accuracy']:.4f}")
    lines.append("")
    lines.append(f"  {'label':<18}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for label, scores in line_report["per_label"].items():
        lines.append(
            f"  {label:<18}{scores['precision']:>8.3f}{scores['recall']:>8.3f}"
            f"{scores['f1']:>8.3f}{scores['support']:>9d}"
        )

    lines.append("")
    lines.append("FIELD EXTRACTION BY FORMAT  (recall = correct value / documents carrying the field)")
    lines.append(
        f"  {'format':<12}{'docs':>6}{'ocr':>5}{'recall':>9}{'prec':>8}{'ocr-intact':>12}{'vs ceiling':>12}"
    )
    for fmt, scores in field_report["per_format"].items():
        lines.append(
            f"  {fmt:<12}{scores['documents']:>6}{'yes' if scores['uses_ocr'] else 'no':>5}"
            f"{scores['field_recall']:>9.3f}{scores['field_precision']:>8.3f}"
            f"{scores['ocr_intact_rate']:>12.3f}{scores['recall_vs_ocr_ceiling']:>12.3f}"
        )

    lines.append("")
    lines.append("FIELD EXTRACTION BY LABEL  (all formats)")
    lines.append(f"  {'label':<18}{'expected':>10}{'correct':>9}{'recall':>9}{'prec':>8}{'vs ceiling':>12}")
    for label, scores in field_report["per_label"].items():
        lines.append(
            f"  {label:<18}{scores['expected']:>10}{scores['correct']:>9}"
            f"{scores['recall']:>9.3f}{scores['precision']:>8.3f}{scores['recall_vs_ocr_ceiling']:>12.3f}"
        )

    lines.append("")
    lines.append(f"OVERALL field recall        : {field_report['field_recall']:.4f}")
    lines.append(f"OVERALL vs OCR ceiling      : {field_report['recall_vs_ocr_ceiling']:.4f}")
    return "\n".join(lines)
