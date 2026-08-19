"""Build the corpus, train the field classifier, evaluate it, save it.

Run it as a module:

    python -m app.ml.ocr.train --documents 220
    python -m app.ml.ocr.train --corpus artifacts/corpus.jsonl --skip-build

The train/test split is by *document*, never by line — and every format of a
document goes to the same side. Splitting by line would put lines from the
same invoice on both sides; splitting by rendering would put the PDF of an
invoice in train and the photo of that same invoice in test. Either one turns
the test score into a measurement of memorisation.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from .corpus import LabelledLine, build_corpus, load_corpus
from .evaluate import (
    accumulate_field_outcomes,
    field_level_report,
    field_report_from,
    format_report,
    line_level_report,
    line_level_report_from,
    line_predictions,
    new_field_accumulator,
    to_contexts,
)
from .model import FieldLineModel

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CORPUS = ARTIFACTS_DIR / "corpus.jsonl"
DEFAULT_MODEL = ARTIFACTS_DIR / "field_model.joblib"
DEFAULT_REPORT = ARTIFACTS_DIR / "evaluation.json"


def split_by_document(
    rows: list[LabelledLine], test_fraction: float = 0.25, seed: int = 17
) -> tuple[list[LabelledLine], list[LabelledLine]]:
    """Group split on doc_id, so no invoice appears on both sides in any
    format."""
    document_ids = sorted({row.doc_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(document_ids)
    cut = max(1, int(len(document_ids) * test_fraction))
    test_ids = set(document_ids[:cut])
    train_rows = [row for row in rows if row.doc_id not in test_ids]
    test_rows = [row for row in rows if row.doc_id in test_ids]
    return train_rows, test_rows


def train_model(train_rows: list[LabelledLine], **hyperparameters) -> FieldLineModel:
    model = FieldLineModel(**hyperparameters)
    model.fit(to_contexts(train_rows), [row.label for row in train_rows])
    model.metadata.document_count = len({row.doc_id for row in train_rows})
    model.metadata.formats = sorted({row.fmt for row in train_rows})
    return model


def k_fold_by_document(
    rows: list[LabelledLine], folds: int = 5, seed: int = 17
) -> list[tuple[list[LabelledLine], list[LabelledLine]]]:
    """Grouped k-fold: documents are dealt into `folds` buckets, and every
    format of a document travels with it.

    Cross-validation rather than one holdout split because OCR is expensive
    enough that the corpus only holds ~30 documents per raster format. A single
    25% split would measure each of those formats on ~8 documents, which cannot
    distinguish a real difference between formats from noise. Under k-fold every
    document is predicted exactly once, by a model that never saw it, so the
    per-format numbers rest on the whole corpus.
    """
    document_ids = sorted({row.doc_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(document_ids)
    assignment = {doc_id: index % folds for index, doc_id in enumerate(document_ids)}

    splits = []
    for fold in range(folds):
        held_out = {doc_id for doc_id, bucket in assignment.items() if bucket == fold}
        splits.append((
            [row for row in rows if row.doc_id not in held_out],
            [row for row in rows if row.doc_id in held_out],
        ))
    return splits


def cross_validate(rows: list[LabelledLine], folds: int, **hyperparameters) -> tuple[dict, dict]:
    """Train and score one model per fold, pooling every fold's predictions
    into a single line-level and field-level report."""
    truth: list[str] = []
    predicted: list[str] = []
    formats: list[str] = []
    per_format, documents_per_format = new_field_accumulator()

    for index, (train_rows, test_rows) in enumerate(k_fold_by_document(rows, folds), start=1):
        model = train_model(train_rows, **hyperparameters)
        fold_truth, fold_predicted, fold_formats = line_predictions(model, test_rows)
        truth.extend(fold_truth)
        predicted.extend(fold_predicted)
        formats.extend(fold_formats)
        accumulate_field_outcomes(model, test_rows, per_format, documents_per_format)
        logger.info(
            "ocr_model.fold_complete fold=%d/%d train_docs=%d test_docs=%d",
            index, folds,
            len({row.doc_id for row in train_rows}), len({row.doc_id for row in test_rows}),
        )

    return line_level_report_from(truth, predicted, formats), field_report_from(per_format, documents_per_format)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the OCR invoice field-extraction model.")
    parser.add_argument("--documents", type=int, default=200, help="synthetic invoices to generate")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--work-dir", type=Path, default=None, help="scratch space for rendered files")
    parser.add_argument("--skip-build", action="store_true", help="reuse an existing corpus file")
    parser.add_argument("--keep-renders", action="store_true", help="do not delete rendered documents")
    parser.add_argument("--workers", type=int, default=4, help="parallel document builders")
    parser.add_argument(
        "--ocr-formats-per-document", type=int, default=1,
        help="raster formats to OCR per document; rotated so each format gets an equal share",
    )
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="holdout size when --folds is 0")
    parser.add_argument("--folds", type=int, default=5,
                        help="grouped k-fold cross-validation; 0 for a single holdout split")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--C", type=float, default=4.0, dest="regularisation")
    parser.add_argument("--min-df", type=int, default=2)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    work_dir = arguments.work_dir or (ARTIFACTS_DIR / "work")
    if not arguments.skip_build:
        logger.info("ocr_model.corpus_build_started documents=%d", arguments.documents)
        build_corpus(
            count=arguments.documents,
            output_path=arguments.corpus,
            work_dir=work_dir,
            seed_offset=arguments.seed_offset,
            keep_renders=arguments.keep_renders,
            ocr_formats_per_document=arguments.ocr_formats_per_document,
            workers=arguments.workers,
        )

    if not arguments.corpus.exists():
        logger.error("ocr_model.corpus_missing path=%s", arguments.corpus)
        return 1

    rows = load_corpus(arguments.corpus)
    logger.info(
        "ocr_model.corpus_loaded lines=%d documents=%d formats=%s",
        len(rows), len({row.doc_id for row in rows}), sorted({row.fmt for row in rows}),
    )

    hyperparameters = {"C": arguments.regularisation, "min_df": arguments.min_df}

    if arguments.folds and arguments.folds >= 2:
        logger.info("ocr_model.cross_validation folds=%d", arguments.folds)
        line_report, field_report = cross_validate(rows, arguments.folds, **hyperparameters)
        # The scores above come from fold models, each trained on a subset. The
        # artifact that ships is trained on everything: there is no reason to
        # serve a model deliberately starved of a fifth of the corpus.
        model = train_model(rows, **hyperparameters)
    else:
        train_rows, test_rows = split_by_document(rows, arguments.test_fraction)
        logger.info(
            "ocr_model.split train_lines=%d train_docs=%d test_lines=%d test_docs=%d",
            len(train_rows), len({row.doc_id for row in train_rows}),
            len(test_rows), len({row.doc_id for row in test_rows}),
        )
        model = train_model(train_rows, **hyperparameters)
        line_report = line_level_report(model, test_rows)
        field_report = field_level_report(model, test_rows)

    logger.info("ocr_model.training_complete labels=%d", len(model.classes))
    model.metadata.metrics = {
        "line_field_macro_f1": line_report["field_macro_f1"],
        "line_accuracy": line_report["accuracy"],
        "field_recall": field_report["field_recall"],
        "recall_vs_ocr_ceiling": field_report["recall_vs_ocr_ceiling"],
        "per_format": {fmt: scores["field_recall"] for fmt, scores in field_report["per_format"].items()},
        "evaluation": f"{arguments.folds}-fold cross-validation" if arguments.folds >= 2
                      else f"holdout {arguments.test_fraction:.0%}",
    }
    model.save(arguments.model)

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps({"line_level": line_report, "field_level": field_report}, indent=2),
        encoding="utf-8",
    )

    print()
    print(format_report(line_report, field_report))
    print()
    print(f"model  -> {arguments.model}")
    print(f"report -> {arguments.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
