# OCR performance

Where the time in document extraction actually goes, measured rather than
guessed, and which knobs move it. Numbers below are from one machine — an
8-core Windows CPU box — on the sample invoices in `storage/`. Re-measure
before trusting them anywhere else; the point of this file is the *shape* of
the cost, which is stable, not the absolute seconds, which are not.

## Where it landed

| Document | Path | End to end |
|---|---|---|
| PDF with a text layer | `pdfplumber` native extraction | **~0.5s** |
| DOCX / XLSX | direct read | **<0.5s** |
| Scanned PDF or photographed page | OCR (ONNX Runtime) | **~7s per page** |

Plus a one-off ~7s at startup, on a background thread, to build the OCR
session and load the ML field model — see *Warm-up* below. Measure with
`docs`-adjacent scripts against a *warmed* process; a benchmark that skips
warm-up simply moves that cost into whichever document it reads first, which
is what made an early version of this table report a 6s "parse" for a text
PDF that actually parses in 0.08s.

`ocr_max_pages` bounds the OCR case: the default of 3 puts the worst case at
roughly 20s for a three-page scan. Lower it to 1 if single-page invoices are
the only scanned input you expect.

Almost every invoice a supplier emails is the first row, and `text_extraction.py`
keeps it that way: native text is tried first and OCR runs only when a PDF
yields fewer than `MIN_NATIVE_TEXT_CHARACTERS`.

## The backend switch, and why

The OCR path used to take ~170s for the first page of a process and now takes
~7s. Two changes did almost all of it, and only the second required a new
dependency.

**Warm-up (see below) was worth ~35s** on the first document of every process.

**The inference runtime was worth ~7x on every page after that.** Both
backends run the same PP-OCR detection and recognition networks; they differ
only in what executes the graph:

| | Build (once per process) | One 71-line page | Lines found | Mean confidence |
|---|---|---|---|---|
| PaddlePaddle | 15–35s | ~38s | 71 | 0.83 |
| ONNX Runtime | ~1.2s | **~5s** | 71 | 0.99 |

Under paddle the cost was ~450ms **per text line**, and nothing reachable
from `PaddleOCR(...)` moved it:

| Change | Result |
|---|---|
| Medium models → mobile models | 130s → 40s ✅ kept |
| Image 1600px → 960px | 41.7s → 36.5s — barely matters |
| Recognition batch 1 → 8 | 34s → **68s**, i.e. worse |
| oneDNN enabled | crashes the *detector* on this build |
| Uniform input shape (letterboxed 48×320) | 37.8s → 30.3s — a fifth, not an order of magnitude |

Batching loses because text-line crops have very different widths (57
distinct widths among 71 crops here) and a batch is padded to its widest
member. Uniform shapes help only a little, which rules out shape
re-specialisation as the dominant cost. What remained was per-inference
overhead in the paddle CPU build — which is exactly what changing the
executor fixes.

`engine.py` is the only module in the project that imports an OCR engine, so
the swap was contained to that one file. Both backends produce the same
`RawDetection` shape and `service.py` was not touched.

## Warm-up

`ocr_warm_up_on_startup` builds the OCR session **and** loads the ML field
model on a daemon thread at application startup.

This was the single biggest contributor to "extraction never finishes". Both
are fixed per-process costs that used to be paid inside the first upload, so
the first document of every deploy looked like a minute-long extraction even
when the document itself parsed in under a second. Both loaders are cached
and thread-safe, so a request that arrives first simply wins the race.

## What is configured, and why

All of these live in `app/core/config.py` and are overridable by environment
variable.

| Setting | Default | Why |
|---|---|---|
| `ocr_backend` | `onnx` | See the table above. Falls back to paddle automatically if `rapidocr` is not installed, so a deployment that has not picked up the dependency still works. |
| `ocr_warm_up_on_startup` | `true` | Moves ~7s (previously ~40s) off the first upload. |
| `ocr_cpu_threads` | `8` | Match to the deployment's core allocation. Both backends read it. |
| `ocr_max_image_side` | `1600` | A 300-DPI A4 page is ~3500px and the detector downsamples it anyway. Capping changed neither the line count nor the recognised text. |
| `ocr_max_pages` | `3` | An invoice's number, dates, parties and totals are on the first page or two. A bound annexure would multiply the wait without changing a parsed field. |
| `ocr_detection_model` / `ocr_recognition_model` | `PP-OCRv5_mobile_*` | **Paddle backend only.** ~3x faster than the medium pair paddle picks from `lang`, no measurable accuracy cost. Unused under ONNX, which uses the models packaged with `rapidocr`. |
| `ocr_recognition_batch_size` | `1` | **Paddle backend only.** Measured: batching is *slower* there. See above. |

Two changes are not settings because there is no case for the old behaviour:

- **The invoice and vendor-document paths no longer route OCR through
  `workers.vision.read_text`.** That helper wraps the same engine in a fixed
  `fastNlMeansDenoising` + CLAHE pass tuned for number-plate crops. On a full
  page it costs ~2s and rises with resolution, and it does not help a clean
  scan. Pages that genuinely need conditioning get it from
  `image_preprocessing_service`, which measures the page first. ANPR still
  uses `read_text`, which is what that pass was built for.
- **`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`** is set alongside the existing
  oneDNN workaround in `engine.py`. Without it, paddle contacts its model
  hosts on first construction — seconds on a warm cache, a connection
  timeout on a host with no outbound access.

## Reading the logs

`invoice_parsing.completed` carries the split that matters:

```
invoice_parsing.completed document_id=… ocr=True pages=1
  extraction_seconds=5.78 parse_seconds=0.03 total_seconds=5.83
```

`extraction_seconds` is dominated by OCR whenever `ocr=True` and is near zero
when it is `False`. `parse_seconds` is regex over a string and should stay
well under a second once the field model is loaded. A slow document with
`ocr=False` is a parser problem, not an OCR problem — look at
`field_scoring.py`, not here.

`ocr.engine_ready backend=…` says which backend actually got built, which is
worth checking before trusting any timing: it logs `backend=onnx` only when
ONNX Runtime was really used, and the fallback to paddle logs
`ocr.onnx_unavailable` first.

## A note on measuring this on Windows

The first run after installing `onnxruntime` showed a 28s parse step that
never reproduced. That was the OS scanning newly-written DLLs, not the code.
Discard the first run after any install before drawing conclusions.
