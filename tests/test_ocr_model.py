"""The OCR field-extraction model: generation, alignment, training, inference.

None of these tests run OCR. Every assertion here is about logic that has to
be right *before* OCR quality becomes the limiting factor — the generator's
ground truth, the alignment that turns it into labels, the value parsers, the
model round-trip and the fallback behaviour when no artifact is present. OCR
itself is slow enough (minutes per page on CPU) that exercising it in the unit
suite would make the suite unrunnable; it is covered by the evaluation report
that `python -m app.ml.ocr.train` produces.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.ml.ocr import predict
from app.ml.ocr.augment import SEVERITIES, degrade, severity_for
from app.ml.ocr.corpus import (
    NATIVE_FORMATS,
    OCR_FORMAT_POOL,
    build_document,
    formats_for_document,
    label_lines,
    rows_for_document,
)
from app.ml.ocr.extract import _parse_money, extract_fields, repair_gstin
from app.ml.ocr.features import contexts_from_lines
from app.ml.ocr.model import FieldLineModel
from app.ml.ocr.render import rasterise, render_all
from app.ml.ocr.synth import generate_invoice
from app.modules.invoices.parsers.gst_utils import GSTIN_PATTERN
from app.modules.invoices.parsers.text_extraction import extract_document


# ── generator ───────────────────────────────────────────────────────────────

def test_generated_invoice_is_internally_consistent():
    """Ground truth is only useful if it is actually true: the totals the
    generator claims must be the totals its own line items add up to."""
    for seed in range(25):
        invoice = generate_invoice(seed)

        assert invoice.subtotal == sum(item.taxable_value for item in invoice.line_items)
        assert invoice.total_amount == invoice.subtotal + invoice.tax_amount
        if invoice.interstate:
            assert invoice.igst_amount == invoice.tax_amount
            assert invoice.cgst_amount is None and invoice.sgst_amount is None
        else:
            assert invoice.cgst_amount + invoice.sgst_amount == invoice.tax_amount


def test_generated_gstins_are_structurally_valid():
    """A GSTIN the project's own validator rejects would train the model on
    documents it will never see."""
    for seed in range(25):
        invoice = generate_invoice(seed)
        assert GSTIN_PATTERN.fullmatch(invoice.vendor_gstin)
        assert GSTIN_PATTERN.fullmatch(invoice.customer_gstin)


def test_generation_is_deterministic():
    """Reproducibility is the whole reason documents are seeded — a corpus
    that cannot be regenerated cannot be debugged."""
    assert generate_invoice(11) == generate_invoice(11)
    assert generate_invoice(11) != generate_invoice(12)


def test_corpus_covers_both_tax_treatments():
    """CGST/SGST and IGST are different labels; a corpus that is 95% one of
    them cannot teach the other."""
    interstate = sum(generate_invoice(seed).interstate for seed in range(200))
    assert 60 <= interstate <= 140, f"tax treatment badly skewed: {interstate}/200 inter-state"


def test_indian_digit_grouping():
    """1,25,000.00 is not 125,000.00 — getting this wrong would put a wrong
    string in the ground truth and silently poison every money label."""
    invoice = generate_invoice(1)
    invoice.currency, invoice.indian_grouping = "", True
    assert invoice.money(Decimal("125000.00")) == "1,25,000.00"
    invoice.indian_grouping = False
    assert invoice.money(Decimal("125000.00")) == "125,000.00"


# ── augmentation ────────────────────────────────────────────────────────────

def test_degradation_is_seeded_and_shape_preserving():
    page = rasterise_sample()
    first = degrade(page, "medium", seed=5)
    assert first.shape == page.shape
    assert (first == degrade(page, "medium", seed=5)).all(), "same seed must reproduce the page"
    assert not (first == degrade(page, "medium", seed=6)).all()


def test_degradation_severity_increases_damage():
    """If 'heavy' were not measurably worse than 'clean', the severity tiers
    would be decorative and the per-severity numbers meaningless."""
    page = rasterise_sample()
    import numpy as np

    def difference(severity: str) -> float:
        return float(np.abs(degrade(page, severity, seed=3).astype(float) - page.astype(float)).mean())

    assert difference("clean") < difference("light") < difference("heavy")


def test_severity_sampler_uses_every_tier():
    sampled = {severity_for(seed) for seed in range(200)}
    assert sampled == set(SEVERITIES)


# ── rendering and format routing ────────────────────────────────────────────

def test_every_native_format_renders_and_extracts(tmp_path):
    """The point of the exercise: one invoice, several formats, all readable
    through the production extractor."""
    invoice = generate_invoice(4)
    rendered = render_all(invoice, tmp_path, severity="clean", formats=NATIVE_FORMATS)

    assert {document.fmt for document in rendered} == set(NATIVE_FORMATS)
    for document in rendered:
        assert document.path.exists()
        extracted = extract_document(str(document.path))
        assert extracted.used_ocr is False
        assert invoice.invoice_number in extracted.text
        assert invoice.vendor_gstin in extracted.text


def test_ocr_formats_are_rotated_evenly():
    """Each raster format must get the same share of documents, or the
    per-format comparison is measuring sample size."""
    from collections import Counter

    counts = Counter(formats_for_document(seed)[-1] for seed in range(200))
    assert set(counts) == set(OCR_FORMAT_POOL)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_render_skips_unrequested_formats(tmp_path):
    """Rasterising for a format nobody asked for is the single most expensive
    thing this package can do by accident."""
    rendered = render_all(generate_invoice(6), tmp_path, "clean", formats=("docx",))
    assert [document.fmt for document in rendered] == ["docx"]
    assert not list(tmp_path.glob("*.png"))


# ── alignment (ground truth -> labels) ──────────────────────────────────────

def test_alignment_finds_core_fields_on_a_clean_render(tmp_path):
    invoice = generate_invoice(9)
    rendered = render_all(invoice, tmp_path, "clean", formats=("docx",))
    rows = rows_for_document(invoice, rendered[0])

    found = {row.label for row in rows if row.label != "OTHER"}
    for label in ("INVOICE_NUMBER", "INVOICE_DATE", "VENDOR_NAME", "VENDOR_GSTIN",
                  "SUBTOTAL", "TOTAL_AMOUNT", "LINE_ITEM"):
        assert label in found, f"{label} not aligned on a clean DOCX render"


def test_alignment_assigns_each_label_at_most_once():
    invoice = generate_invoice(2)
    lines = [
        "TAX INVOICE",
        f"{invoice.wording['invoice_number']}: {invoice.invoice_number}",
        f"{invoice.wording['total']}: {invoice.money(invoice.total_amount)}",
        f"{invoice.wording['total']}: {invoice.money(invoice.total_amount)}",
    ]
    labels, _ = label_lines(invoice, lines)
    assert labels.count("TOTAL_AMOUNT") == 1


def test_alignment_survives_label_and_value_on_separate_lines():
    """The OCR case: a right-aligned totals block comes back as two lines.
    If alignment cannot label that, no scanned invoice ever gets a total."""
    invoice = generate_invoice(2)
    lines = ["TAX INVOICE", f"{invoice.wording['total']}:", invoice.money(invoice.total_amount)]
    labels, _ = label_lines(invoice, lines)
    assert labels[2] == "TOTAL_AMOUNT", labels


def test_alignment_labels_a_corrupted_value_but_flags_it_inexact():
    """A total OCR mangled is still the total line. Labelling it OTHER would
    teach the model to ignore exactly the lines it most needs to find."""
    invoice = generate_invoice(2)
    intact = invoice.money(invoice.total_amount)
    # Change the last paisa digit: still a well-formed amount, but the wrong
    # one — which is exactly what a misread scan produces.
    mangled = intact[:-1] + ("3" if intact[-1] != "3" else "4")
    labels, exact = label_lines(invoice, ["TAX INVOICE", f"{invoice.wording['total']}: {mangled}"])

    assert labels[1] == "TOTAL_AMOUNT"
    assert exact["TOTAL_AMOUNT"] is False


# ── value parsing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "corrupted, expected",
    [
        ("29ABCDE1234F1Z5", "29ABCDE1234F1Z5"),          # already valid
        ("Z9ABCDE1234F1Z5", "29ABCDE1234F1Z5"),          # Z read for 2 in a digit slot
        ("29ABCDE1Z34F1Z5", "29ABCDE1234F1Z5"),          # Z read for 2 inside the numeric run
        ("29ABCDE1O34F1Z5", "29ABCDE1034F1Z5"),          # O read for 0
        ("Z9ABCDE1Z34F1Z5", "29ABCDE1234F1Z5"),          # two digit slots at once
    ],
)
def test_gstin_ocr_confusions_are_repaired(corrupted, expected):
    assert repair_gstin(corrupted) == expected


def test_gstin_repair_leaves_ambiguous_but_valid_positions_alone():
    """Position 13 (the entity code) legitimately accepts both letters and
    digits, so an 'I' there is a real value, not a misread '1'. "Repairing"
    it would corrupt a GSTIN that was already correct."""
    assert repair_gstin("29ABCDE1234FIZ5") == "29ABCDE1234FIZ5"


def test_gstin_repair_refuses_unrecoverable_input():
    assert repair_gstin("NOTAGSTINATALL") is None
    assert repair_gstin("29ABCDE1234F1Z") is None  # 14 characters


def test_party_name_keeps_an_internal_hyphen():
    """A block heading is stripped off a name line, but "Agro-Exports" is a
    name, not a heading followed by a value."""
    from app.ml.ocr.extract import _parse_text

    assert _parse_text("Coromandel Agro-Exports") == "Coromandel Agro-Exports"
    assert _parse_text("Bill To: Meridian Technologies Pvt Ltd") == "Meridian Technologies Pvt Ltd"
    assert _parse_text("Seller - Kaveri Steel Works") == "Kaveri Steel Works"
    assert _parse_text("Seller:") is None


def test_money_parser_takes_the_value_not_the_rate():
    """"CGST @ 9%: 1,234.00" holds two numbers and only one of them is the
    amount. Taking the first would silently return the tax rate."""
    assert _parse_money("CGST @ 9%: INR 1,234.00") == Decimal("1234.00")
    assert _parse_money("Taxable Value 1,25,000.00") == Decimal("125000.00")
    assert _parse_money("no amount here") is None


# ── features ────────────────────────────────────────────────────────────────

def test_contexts_match_the_neighbourhood_recorded_at_training_time(tmp_path):
    """Training rows and inference contexts must describe the same
    neighbourhood, or the model is served a different view than it learnt."""
    invoice = generate_invoice(13)
    rendered = render_all(invoice, tmp_path, "clean", formats=("docx",))
    rows = rows_for_document(invoice, rendered[0])
    contexts = contexts_from_lines([row.text for row in rows])

    assert len(contexts) == len(rows)
    for context, row in zip(contexts, rows):
        assert context.text == row.text
        assert context.previous_text == row.previous_text
        assert context.next_text == row.next_text


# ── model ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A real model trained on a handful of native-format documents. Small
    enough for the unit suite, real enough that predict/save/load are being
    exercised on an actual fitted pipeline."""
    from app.ml.ocr.corpus import LabelledLine
    from app.ml.ocr.evaluate import to_contexts

    work = tmp_path_factory.mktemp("tiny-corpus")
    rows = [
        LabelledLine(**row)
        for seed in range(14)
        for row in build_document(seed, work, formats=NATIVE_FORMATS)
    ]
    model = FieldLineModel(min_df=1).fit(to_contexts(rows), [row.label for row in rows])
    return model, rows


def test_model_learns_the_field_labels(tiny_model):
    model, _ = tiny_model
    for label in ("INVOICE_NUMBER", "TOTAL_AMOUNT", "VENDOR_GSTIN", "OTHER"):
        assert label in set(model.classes)


def test_model_extracts_values_from_an_unseen_document(tmp_path, tiny_model):
    """End to end on a document the model was not trained on."""
    model, _ = tiny_model
    invoice = generate_invoice(500)
    rendered = render_all(invoice, tmp_path, "clean", formats=("docx",))
    extracted = extract_document(str(rendered[0].path))

    found = extract_fields(model, contexts_from_lines(extracted.text.splitlines()))

    assert found["TOTAL_AMOUNT"].value == invoice.total_amount
    assert found["INVOICE_DATE"].value == invoice.invoice_date
    assert found["VENDOR_GSTIN"].value == invoice.vendor_gstin


def test_model_round_trips_through_disk(tmp_path, tiny_model):
    model, rows = tiny_model
    path = model.save(tmp_path / "field_model.joblib")
    reloaded = FieldLineModel.load(path)

    from app.ml.ocr.evaluate import to_contexts

    contexts = to_contexts(rows[:60])
    assert reloaded.predict(contexts) == model.predict(contexts)


def test_loading_a_stale_artifact_is_refused(tmp_path, tiny_model):
    """A pipeline whose feature layout no longer matches the code would not
    raise on load — it would quietly predict nonsense."""
    import joblib

    model, _ = tiny_model
    path = tmp_path / "stale.joblib"
    joblib.dump({"pipeline": model.pipeline, "metadata": {"format_version": 0}}, path)

    with pytest.raises(ValueError, match="format version"):
        FieldLineModel.load(path)


# ── inference wiring and fallback ───────────────────────────────────────────

def test_missing_artifact_degrades_instead_of_failing(tmp_path, monkeypatch):
    """An untrained checkout must still parse invoices."""
    monkeypatch.setattr(predict, "DEFAULT_MODEL_PATH", tmp_path / "absent.joblib")
    predict.reset_cache()

    assert predict.available() is False
    assert predict.predict_fields("Invoice No: INV-1\nTotal: 100.00") == {}


def test_unreadable_artifact_degrades_instead_of_failing(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib file")
    monkeypatch.setattr(predict, "DEFAULT_MODEL_PATH", corrupt)
    predict.reset_cache()

    assert predict.available() is False


def test_parser_selection_falls_back_without_a_model(tmp_path, monkeypatch):
    from app.modules.invoices.parsers import select_parser

    monkeypatch.setattr(predict, "DEFAULT_MODEL_PATH", tmp_path / "absent.joblib")
    predict.reset_cache()

    assert select_parser("Tax Invoice No: INV-1\nGrand Total: 100.00").name == "generic"


def test_ml_parser_fills_a_field_the_regex_parser_misses(tmp_path, monkeypatch, tiny_model):
    """The reason the model exists: a wording the alias list does not contain.
    "Total Payable" is absent from invoice_field_parser's label table, so the
    deterministic parser returns None for the grand total."""
    from app.modules.invoices.parsers.generic_invoice_parser import GenericInvoiceParser
    from app.modules.invoices.parsers.ml_invoice_parser import MlAssistedInvoiceParser
    from app.modules.invoices.parsers.text_extraction import ExtractedDocument

    model, _ = tiny_model
    monkeypatch.setattr(predict, "_model", model)
    monkeypatch.setattr(predict, "_load_attempted", True)

    invoice = generate_invoice(501)
    text = "\n".join([
        "TAX INVOICE",
        f"Tax Invoice No.: {invoice.invoice_number}",
        f"Invoice Date: {invoice.formatted_date(invoice.invoice_date)}",
        "Seller:",
        invoice.vendor_name,
        f"GSTIN: {invoice.vendor_gstin}",
        f"Taxable Value: {invoice.money(invoice.subtotal)}",
        f"Total Payable: {invoice.money(invoice.total_amount)}",
    ])
    extracted = ExtractedDocument(text=text, tables_per_page=[[]], used_ocr=True, page_count=1)

    baseline = GenericInvoiceParser().parse(extracted)
    assert baseline.total_amount is None, "precondition: the regex parser should miss this wording"

    assisted = MlAssistedInvoiceParser().parse(extracted)
    assert assisted.total_amount == invoice.total_amount
    assert assisted.parser_name == "ml_assisted"
    assert any("OCR field model" in warning for warning in assisted.warnings)


def test_ml_parser_never_overwrites_a_deterministic_value(tmp_path, monkeypatch, tiny_model):
    """Where the regex parser found a value, it wins. It matched an exact
    label; the model only ever offers a probability."""
    from app.modules.invoices.parsers.ml_invoice_parser import MlAssistedInvoiceParser
    from app.modules.invoices.parsers.text_extraction import ExtractedDocument

    model, _ = tiny_model
    monkeypatch.setattr(predict, "_model", model)
    monkeypatch.setattr(predict, "_load_attempted", True)

    invoice = generate_invoice(502)
    text = "\n".join([
        "TAX INVOICE",
        f"Invoice No: {invoice.invoice_number}",
        f"Invoice Date: {invoice.formatted_date(invoice.invoice_date)}",
        f"Grand Total: {invoice.money(invoice.total_amount)}",
    ])
    extracted = ExtractedDocument(text=text, tables_per_page=[[]], used_ocr=False, page_count=1)

    assisted = MlAssistedInvoiceParser().parse(extracted)
    assert assisted.total_amount == invoice.total_amount


def rasterise_sample():
    """A rendered page as an array, for the augmentation tests."""
    import tempfile
    from pathlib import Path

    from app.ml.ocr.render import render_pdf

    directory = Path(tempfile.mkdtemp(prefix="ocr-augment-"))
    return rasterise(render_pdf(generate_invoice(3), directory / "page.pdf"), dpi=72)

