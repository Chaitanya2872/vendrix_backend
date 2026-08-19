# OCR invoice field-extraction model

A trainable model that reads an uploaded invoice — in any format the upload
endpoint accepts — and returns typed header fields.

It does **not** replace the OCR engine. PaddleOCR (`app/workers/vision.py`)
still turns pixels into text. This package solves the layer above: given the
text that comes back, which line carries the invoice number, which carries the
grand total, and what are the values. That layer was previously a fixed list of
label aliases and regexes (`parsers/invoice_field_parser.py`), which is exact
when it matches and blind when it does not — a wording nobody listed, or an OCR
pass that turned `Amount Payable` into `AmountPayabIe`, and the field comes back
empty.

## How it works

```
synth.py     generate an invoice's *content* — ground truth is exact by construction
render.py    render that same invoice to pdf / docx / xlsx / jpg / png / webp / scanned-pdf
augment.py   degrade the raster formats: skew, blur, noise, shadow, JPEG artefacts
   |
   v
text_extraction.extract_document()      <- the real production extractor, not a shortcut
   |
   v
corpus.py    align recovered lines against known values -> one labelled row per line
features.py  char n-grams + contextual word n-grams + shape/position features
model.py     class-weighted multinomial logistic regression -> per-line label + probability
extract.py   label + line -> typed value (Decimal / date / GSTIN / identifier / text)
```

Three decisions carry most of the weight:

**Extraction runs through the production code path.** The corpus is built by
calling the same `extract_document()` the application calls. A corpus built
from a private text path would train the model on text the application never
produces, and every number measured against it would describe the wrong
distribution.

**Labels come from evidence, not exact matching.** On a degraded scan, OCR
returns `1,92,4O7.04` for `1,92,407.04`. Requiring an exact match would label
that line `OTHER` — teaching the model that a mangled total is not a total,
which is backwards. A line is labelled when the nearby label wording plus a
value of the right shape are jointly convincing; whether the value survived
intact is recorded separately as `value_exact`. That flag is what lets the
evaluation separate *the model pointed at the wrong line* from *OCR destroyed
the right line* — two failures with completely different fixes.

**Splits are by document, never by line.** Every format of one invoice goes to
the same side of the train/test split. Splitting by line puts lines of one
invoice on both sides; splitting by rendering puts the PDF of an invoice in
train and the photo of it in test. Either turns the test score into a
measurement of memorisation.

Scoring is 5-fold grouped cross-validation, not a single holdout. OCR is
expensive enough that the corpus holds only ~30 documents per raster format; a
25% holdout would measure each of those on ~8 documents, which cannot tell a
real difference between formats from noise. Under k-fold every document is
predicted exactly once by a model that never saw it, so the per-format numbers
rest on the whole corpus. The artifact that ships is then retrained on
everything — there is no reason to serve a model starved of a fifth of the
data.

## Training

```bash
python -m app.ml.ocr.train --documents 120 --workers 4
```

Writes `artifacts/corpus.jsonl`, `artifacts/field_model.joblib` and
`artifacts/evaluation.json`, and prints the per-format report.

Useful flags:

| flag | why |
|---|---|
| `--skip-build` | retrain on an existing corpus without re-running OCR |
| `--workers N` | parallel document builders (OCR is the whole cost) |
| `--ocr-formats-per-document N` | raster formats to OCR per document, rotated evenly |
| `--keep-renders` | keep the rendered files for inspection |
| `--folds N` | grouped k-fold cross-validation (default 5); `0` for a single holdout |

**OCR is the bottleneck, by two orders of magnitude.** On this CPU build one
page takes ~2 minutes (oneDNN is disabled in `vision.py` to work around a crash
in text detection). Four parallel workers give ~2.1x, not 4x, because Paddle is
already multi-threaded. That is why each document contributes one OCR rendering
by default, rotated across `jpg / png / webp / pdf_scan` so every raster format
gets an equal share of documents. Building the corpus is hours; retraining on an
existing corpus is seconds.

The corpus file is appended per document and the build is resumable — rerunning
after an interruption picks up the documents that are missing.

## Serving

`predict.py` loads the artifact lazily and caches it. If no artifact is present
the model reports itself unavailable and **nothing breaks**:
`MlAssistedInvoiceParser.can_parse()` returns `False`, dispatch falls through to
`GenericInvoiceParser`, and invoice parsing behaves exactly as it did before
this package existed. An untrained checkout is a supported state.

When the model *is* available it does not replace the deterministic parser — it
runs after it and fills only the fields that came back `None`. A model value
never overwrites a regex match: where both are confident they agree, and where
they disagree the exact label match is the better bet. Fields the model supplied
are named in a warning on the result, so a reviewer can see which values were
inferred rather than read.

Settings (`app/core/config.py`):

- `ocr_field_model_path` — override the artifact location
- `ocr_field_model_min_confidence` — ignore model fields below this probability

## Reading the evaluation

`format_report()` prints three blocks. The one that matters is the per-format
table:

- `recall` — fields whose **value** came out correct, over documents carrying
  the field
- `ocr-intact` — fraction of those fields whose value survived extraction at
  all. This is the ceiling; no classifier change can beat it.
- `vs ceiling` — recall measured only over fields OCR did not destroy. This is
  the model's actual score. A low `recall` with a high `vs ceiling` means the
  OCR stage is the problem, not the model.

## Known limitations

- **Two-column layouts on native PDFs.** `pdfplumber.extract_text()` collapses
  the gap between columns to a single space, so a side-by-side seller/buyer
  block arrives as one line holding both parties' details. A line-level
  classifier can assign that line to only one label, so the customer fields are
  lost on those documents. Fixing it needs word positions
  (`page.extract_words()`) to split columns before classification, or a
  token-level model. This affects the deterministic parser identically.
- **Synthetic training data.** The corpus covers wording, date-format, currency,
  grouping, layout and degradation variation, but it is generated from one
  renderer. Real vendor templates will contain shapes it has never seen; the
  first real correction data from the review screen is worth more than another
  thousand synthetic documents.
- **Header fields only.** Line items are recognised as a class but are not
  reduced to structured rows — the existing table parser owns that, and it works
  from real table geometry that OCR does not preserve.
