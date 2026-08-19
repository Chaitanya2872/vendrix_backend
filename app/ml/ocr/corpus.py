"""Turn synthetic invoices into labelled training lines.

The pipeline per document is: generate content -> render to a format -> run
the *production* extractor (`text_extraction.extract_document`) -> align the
recovered lines against the known values -> emit one labelled row per line.

Two deliberate choices:

1. Extraction goes through the real production function, not a shortcut. A
   corpus built from a private text path would train the model on text the
   application never actually produces, and every measurement taken on it
   would be about the wrong distribution.

2. Alignment is evidence-based rather than exact-match. On a degraded scan,
   OCR routinely returns "Amount Payable" as "AmountPayabIe" and 1,92,407.04
   as 1,92,4O7.04. Requiring an exact match there would label those lines
   OTHER — teaching the model that a mangled total is not a total, which is
   precisely backwards. So a line is labelled when the *evidence* (nearby
   label wording plus a value of the right shape) is strong enough, and
   whether the value survived intact is recorded separately as
   `value_exact`. That flag is what evaluate.py uses to separate "the model
   found the right line" from "OCR read the line correctly" — two failures
   with completely different fixes.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path

from .labels import FORMATS, OTHER, SINGLE_VALUE_LABELS, SPEC_BY_LABEL
from .augment import severity_for
from .render import RenderedDocument, render_all
from .synth import SynthInvoice, generate_invoice

logger = logging.getLogger(__name__)

# A money token: optional currency marker, then grouped digits with a decimal
# part. Both Western (125,000.00) and Indian (1,25,000.00) grouping match, and
# so does an ungrouped number, which is what OCR often returns when it loses a
# thin comma.
MONEY_TOKEN = re.compile(r"(?:₹|Rs\.?|INR|USD|\$)?\s*-?\d[\d,]*\.\d{2}\b", re.IGNORECASE)

# Every date shape the generator emits, plus the common real-world ones.
DATE_TOKEN = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}"
)

GSTIN_TOKEN = re.compile(r"\b[0-9A-Z]{15}\b")
ALNUM_TOKEN = re.compile(r"\b[A-Z0-9][A-Z0-9/\-]{3,}\b")

# Assignment thresholds. Set from inspecting alignment on heavy-degradation
# samples: below these the "match" is usually a coincidence, and a wrong
# positive is more expensive than a missed one because it trains the model
# towards a line that carries nothing.
MIN_COMBINED_SCORE = 0.60
MIN_VALUE_EVIDENCE = 0.55
MIN_LINE_ITEM_SIMILARITY = 0.62

# How much a label found on the *previous* line counts relative to one on the
# line itself. OCR frequently puts "Total:" and its amount on separate lines,
# so previous-line evidence has to count — but not enough to outrank a line
# that carries both.
PREVIOUS_LINE_DISCOUNT = 0.90

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


@dataclass
class LabelledLine:
    """One text line from one rendering of one document, with its label."""
    doc_id: str
    fmt: str
    severity: str
    line_index: int
    line_count: int
    text: str
    previous_text: str
    next_text: str
    label: str
    value_exact: bool


def _normalise(text: str) -> str:
    return _NON_ALNUM.sub("", _WHITESPACE.sub(" ", text.lower())).strip()


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _contains_ratio(needle: str, haystack: str) -> float:
    """How strongly `needle` appears somewhere inside `haystack`.

    A plain full-string ratio would score a five-character label against a
    ninety-character table row as near zero even when the label is right
    there, so this slides a needle-sized window instead.
    """
    needle, haystack = _normalise(needle), _normalise(haystack)
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    if len(haystack) <= len(needle):
        return _ratio(needle, haystack)
    width = len(needle)
    best = 0.0
    for start in range(0, len(haystack) - width + 1):
        best = max(best, _ratio(needle, haystack[start:start + width]))
        if best >= 0.99:
            break
    return best


def _money_values(line: str) -> list[Decimal]:
    from ...modules.invoices.parsers.money_utils import parse_amount

    values = []
    for match in MONEY_TOKEN.finditer(line):
        parsed = parse_amount(match.group(0))
        if parsed is not None:
            values.append(parsed)
    return values


def _date_values(line: str) -> list[date]:
    from dateutil import parser as dateutil_parser

    values = []
    for match in DATE_TOKEN.finditer(line):
        for dayfirst in (True, False):
            try:
                values.append(dateutil_parser.parse(match.group(0), dayfirst=dayfirst).date())
            except (ValueError, OverflowError, TypeError):
                continue
    return values


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _value_evidence(kind: str, line: str, truth_text: str, invoice: SynthInvoice, label: str) -> tuple[float, bool]:
    """How strongly this line carries the expected *value*.

    Returns (score, exact). `exact` means the value survived extraction
    intact — the model's job is the score, OCR's job is the flag.
    """
    if kind == "money":
        expected = invoice.numeric_truth().get(label)
        found = _money_values(line)
        if expected is not None and any(value == expected for value in found):
            return 1.0, True
        if not found:
            return 0.0, False
        # A near-miss on digits is still a money line — it is the line we want
        # the model to find, even though OCR corrupted a character in it.
        target = _digits(str(expected)) if expected is not None else ""
        return max(_ratio(target, _digits(str(value))) for value in found), False

    if kind == "date":
        expected_date = invoice.date_truth().get(label)
        found_dates = _date_values(line)
        if expected_date is not None and expected_date in found_dates:
            return 1.0, True
        tokens = DATE_TOKEN.findall(line)
        if not tokens:
            return 0.0, False
        return max(_ratio(_normalise(truth_text), _normalise(token)) for token in tokens), False

    if kind == "gstin":
        tokens = GSTIN_TOKEN.findall(line.upper())
        if truth_text.upper() in tokens:
            return 1.0, True
        if not tokens:
            # OCR sometimes splits a GSTIN with a space; fall back to the line.
            return _contains_ratio(truth_text, line) * 0.8, False
        return max(_ratio(truth_text.upper(), token) for token in tokens), False

    if kind == "identifier":
        if truth_text.upper() in line.upper():
            return 1.0, True
        tokens = ALNUM_TOKEN.findall(line.upper())
        if not tokens:
            return _contains_ratio(truth_text, line), False
        return max(_ratio(truth_text.upper(), token) for token in tokens), False

    # Free text (party names): the line is expected to *be* the value.
    score = _contains_ratio(truth_text, line)
    return score, _normalise(truth_text) == _normalise(line)


_WORDING_KEY_FOR_LABEL: dict[str, str] = {
    "INVOICE_NUMBER": "invoice_number",
    "INVOICE_DATE": "invoice_date",
    "DUE_DATE": "due_date",
    "VENDOR_NAME": "vendor_block",
    "VENDOR_GSTIN": "gstin",
    "CUSTOMER_NAME": "customer_block",
    "CUSTOMER_GSTIN": "gstin",
    "SUBTOTAL": "subtotal",
    "CGST_AMOUNT": "cgst",
    "SGST_AMOUNT": "sgst",
    "IGST_AMOUNT": "igst",
    "TAX_AMOUNT": "tax",
    "TOTAL_AMOUNT": "total",
}

# How much the surrounding label wording counts versus the value itself.
# For an amount, the wording is the only thing separating CGST from SGST —
# they carry identical values on every intra-state invoice — so it has to
# weigh heavily. For a party name or a GSTIN the value is distinctive on its
# own and the wording is just a block heading some layouts omit.
_WORDING_WEIGHT: dict[str, float] = {
    "money": 0.50, "date": 0.45, "gstin": 0.20, "identifier": 0.30, "text": 0.25,
}


def label_lines(invoice: SynthInvoice, lines: list[str]) -> tuple[list[str], dict[str, bool]]:
    """Assign one label per line. Returns (labels, exactness per label)."""
    truth = invoice.ground_truth()
    scores: list[tuple[float, str, int, bool]] = []

    for label in SINGLE_VALUE_LABELS:
        truth_text = truth.get(label)
        if truth_text is None:
            continue  # this document does not carry the field at all
        kind = SPEC_BY_LABEL[label].kind
        wording = invoice.wording[_WORDING_KEY_FOR_LABEL[label]]
        weight = _WORDING_WEIGHT[kind]

        for index, line in enumerate(lines):
            value_score, exact = _value_evidence(kind, line, truth_text, invoice, label)
            if value_score < MIN_VALUE_EVIDENCE:
                continue
            wording_score = _contains_ratio(wording, line)
            if index > 0:
                wording_score = max(wording_score, PREVIOUS_LINE_DISCOUNT * _contains_ratio(wording, lines[index - 1]))
            combined = weight * wording_score + (1 - weight) * value_score
            if combined >= MIN_COMBINED_SCORE:
                scores.append((combined, label, index, exact))

    # Greedy assignment, best evidence first. One label per document and one
    # label per line: a line holding both a subtotal and a total is not a
    # thing, and assigning both would put contradictory targets on one row.
    labels = [OTHER] * len(lines)
    exactness: dict[str, bool] = {}
    used_lines: set[int] = set()
    for combined, label, index, exact in sorted(scores, key=lambda item: -item[0]):
        if label in exactness or index in used_lines:
            continue
        labels[index] = label
        exactness[label] = exact
        used_lines.add(index)

    # Line items are recognised, not reduced: several lines carry them, so
    # they are labelled after the single-value fields have claimed theirs.
    for index, line in enumerate(lines):
        if index in used_lines:
            continue
        normalised = _normalise(line)
        if not normalised:
            continue
        if any(item.hsn_sac in line for item in invoice.line_items):
            labels[index] = "LINE_ITEM"
            continue
        if any(_contains_ratio(item.description, line) >= MIN_LINE_ITEM_SIMILARITY for item in invoice.line_items):
            labels[index] = "LINE_ITEM"

    return labels, exactness


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def rows_for_document(invoice: SynthInvoice, rendered: RenderedDocument) -> list[LabelledLine]:
    """Extract and label one rendering. Returns [] when extraction fails —
    a document format that cannot be read is a corpus gap, not a crash."""
    from ...modules.invoices.parsers.text_extraction import (
        UnsupportedDocumentError,
        extract_document,
    )

    try:
        extracted = extract_document(str(rendered.path))
    except UnsupportedDocumentError as exc:
        logger.warning("ocr_model.extraction_failed doc=%s fmt=%s error=%s", rendered.doc_id, rendered.fmt, exc)
        return []
    except Exception:
        logger.exception("ocr_model.extraction_error doc=%s fmt=%s", rendered.doc_id, rendered.fmt)
        return []

    lines = _clean_lines(extracted.text)
    if not lines:
        logger.warning("ocr_model.no_text doc=%s fmt=%s", rendered.doc_id, rendered.fmt)
        return []

    labels, exact_by_label = label_lines(invoice, lines)

    return [
        LabelledLine(
            doc_id=rendered.doc_id,
            fmt=rendered.fmt,
            severity=rendered.severity,
            line_index=index,
            line_count=len(lines),
            text=line,
            previous_text=lines[index - 1] if index > 0 else "",
            next_text=lines[index + 1] if index + 1 < len(lines) else "",
            label=labels[index],
            value_exact=exact_by_label.get(labels[index], False),
        )
        for index, line in enumerate(lines)
    ]


# Formats that need no OCR. Every document gets all of these: they cost
# milliseconds and they are what the model needs in order to learn the clean
# case as well as the degraded one.
NATIVE_FORMATS: tuple[str, ...] = ("pdf_native", "docx", "xlsx")
OCR_FORMAT_POOL: tuple[str, ...] = ("jpg", "png", "webp", "pdf_scan")


def formats_for_document(seed: int, ocr_formats_per_document: int = 1) -> tuple[str, ...]:
    """Which formats to render for one document.

    OCR of a single page costs ~2 minutes of CPU here, so rendering all four
    raster formats of every document would put the corpus out of reach. They
    are rotated instead: each document contributes one (or a few) OCR
    renderings, and across the corpus every raster format gets an equal share
    of documents. Rotating rather than sampling keeps that share exact, which
    matters when the per-format numbers are the point of the exercise.
    """
    if ocr_formats_per_document <= 0:
        return NATIVE_FORMATS
    count = min(ocr_formats_per_document, len(OCR_FORMAT_POOL))
    offset = seed % len(OCR_FORMAT_POOL)
    chosen = tuple(OCR_FORMAT_POOL[(offset + step) % len(OCR_FORMAT_POOL)] for step in range(count))
    return NATIVE_FORMATS + chosen


def build_document(
    seed: int,
    work_dir: Path,
    keep_renders: bool = False,
    formats: tuple[str, ...] | None = None,
    ocr_formats_per_document: int = 1,
    skip: frozenset[str] = frozenset(),
) -> list[dict]:
    """Render and extract one document, returning JSON-ready labelled rows.

    Module-level and self-contained so it can run in a worker process: the
    corpus build is CPU-bound in OCR, and the only way to use more than one
    core is to hand whole documents to separate processes.
    """
    invoice = generate_invoice(seed)
    severity = severity_for(seed)
    wanted = formats if formats is not None else formats_for_document(seed, ocr_formats_per_document)
    wanted = tuple(fmt for fmt in wanted if fmt not in skip)
    if not wanted:
        return []

    document_dir = Path(work_dir) / invoice.doc_id
    rows: list[dict] = []
    try:
        for rendered in render_all(invoice, document_dir, severity, formats=wanted):
            rows.extend(asdict(row) for row in rows_for_document(invoice, rendered))
    except Exception:
        # Keep whatever was already extracted. A raster-stage failure should
        # not also throw away the native formats that came out fine, and the
        # resumable build will retry the missing ones on the next run.
        logger.exception("ocr_model.document_partially_failed doc=%s recovered_rows=%d", invoice.doc_id, len(rows))
    finally:
        if not keep_renders:
            # A degraded page as PNG plus an image-only PDF runs to ~16 MB per
            # document; a few hundred documents would fill the disk for no
            # benefit once the text has been extracted.
            shutil.rmtree(document_dir, ignore_errors=True)
    return rows


def completed_formats(output_path: Path) -> dict[str, set[str]]:
    """Which (document, format) pairs the corpus file already holds. The file
    on disk is the source of truth for resuming — not an in-memory tally,
    which a crashed build would have taken with it."""
    done: dict[str, set[str]] = defaultdict(set)
    if not Path(output_path).exists():
        return done
    with Path(output_path).open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a partial last line from an interrupted write
            done[row["doc_id"]].add(row["fmt"])
    return done


def _pending_documents(
    count: int,
    seed_offset: int,
    done: dict[str, set[str]],
    formats: tuple[str, ...] | None,
    ocr_formats_per_document: int,
) -> list[tuple[int, frozenset[str]]]:
    pending: list[tuple[int, frozenset[str]]] = []
    for index in range(count):
        seed = seed_offset + index
        doc_id = generate_invoice(seed).doc_id
        wanted = formats if formats is not None else formats_for_document(seed, ocr_formats_per_document)
        already = done.get(doc_id, set())
        if all(fmt in already for fmt in wanted):
            continue
        pending.append((seed, frozenset(already)))
    return pending


# A crashed worker takes the whole pool with it. Restart it this many times,
# shrinking the worker count each round, before giving up: the crash is
# usually memory pressure from several PaddleOCR processes at once, so fewer
# workers is the fix rather than a reason to abandon the build.
MAX_POOL_RESTARTS = 3


def build_corpus(
    count: int,
    output_path: Path,
    work_dir: Path,
    seed_offset: int = 0,
    keep_renders: bool = False,
    formats: tuple[str, ...] | None = None,
    ocr_formats_per_document: int = 1,
    workers: int = 3,
) -> Path:
    """Generate `count` invoices, render and extract each, append labelled
    lines to `output_path` as JSONL.

    Rows are appended per document rather than written once at the end, so an
    interrupted build resumes instead of starting over — OCR of a few hundred
    pages takes long enough that losing it matters.

    Raises RuntimeError if documents remain unbuilt after the pool has been
    restarted its allowance of times. A partial corpus is a legitimate thing
    to keep and resume from, but it must never be mistaken for a complete one:
    every per-format number downstream would silently be computed over fewer
    documents than it claims.

    Call this from under an `if __name__ == "__main__"` guard when
    `workers > 1`: worker processes are spawned on Windows and re-import the
    calling module.
    """
    output_path, work_dir = Path(output_path), Path(work_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    done = completed_formats(output_path)
    if done:
        logger.info("ocr_model.corpus_resuming existing_documents=%d", len(done))

    pending = _pending_documents(count, seed_offset, done, formats, ocr_formats_per_document)
    if not pending:
        logger.info("ocr_model.corpus_already_complete documents=%d", count)
        return output_path

    total_outstanding = len(pending)
    logger.info("ocr_model.corpus_build documents=%d workers=%d", total_outstanding, workers)

    completed = 0

    def emit(handle, rows: list[dict]) -> None:
        nonlocal completed
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        completed += 1
        if completed % 5 == 0 or completed == total_outstanding:
            logger.info("ocr_model.corpus_progress documents=%d/%d", completed, total_outstanding)

    if workers <= 1:
        with output_path.open("a", encoding="utf-8") as handle:
            for seed, already in pending:
                emit(handle, build_document(seed, work_dir, keep_renders, formats, ocr_formats_per_document, already))
        return output_path

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    for attempt in range(MAX_POOL_RESTARTS + 1):
        if not pending:
            break
        if attempt:
            # Recompute from disk: the crashed run may have written documents
            # whose futures never reported back.
            pending = _pending_documents(
                count, seed_offset, completed_formats(output_path), formats, ocr_formats_per_document
            )
            if not pending:
                break
            workers = max(1, workers - 1)
            logger.warning(
                "ocr_model.pool_restart attempt=%d remaining=%d workers=%d",
                attempt, len(pending), workers,
            )

        try:
            with output_path.open("a", encoding="utf-8") as handle:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            build_document, seed, work_dir, keep_renders,
                            formats, ocr_formats_per_document, already,
                        )
                        for seed, already in pending
                    ]
                    for future in as_completed(futures):
                        emit(handle, future.result())
            pending = []
        except BrokenProcessPool:
            # One worker died natively — usually PaddleOCR under memory
            # pressure. Everything still queued is lost with it, so the pool
            # has to come back up rather than each future being logged as its
            # own failure.
            logger.exception("ocr_model.pool_broken attempt=%d", attempt)

    remaining = _pending_documents(
        count, seed_offset, completed_formats(output_path), formats, ocr_formats_per_document
    )
    if remaining:
        raise RuntimeError(
            f"Corpus build incomplete: {len(remaining)} of {count} documents still missing after "
            f"{MAX_POOL_RESTARTS} pool restarts. The corpus written so far is valid and the build "
            f"resumes where it stopped — rerun, optionally with fewer --workers."
        )

    return output_path


def load_corpus(path: Path) -> list[LabelledLine]:
    rows: list[LabelledLine] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(LabelledLine(**json.loads(raw)))
    return rows
