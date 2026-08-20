# Invoice OCR and data extraction

Reads invoices from any supplier without a per-vendor template, entirely on
this machine. No cloud OCR service is contacted at any point, and after the
image is built no network is needed at all.

## What it does

```
upload → validate → per-page routing → preprocess → OCR → layout → tables
       → scored field extraction → validation → confidence → review
```

The whole design rests on one idea: **work from where text sits, not just
what it says.** A parser reading a flat string can only ask "what characters
follow the word `Total`". This one can ask "what sits immediately right of
this label across 400px of whitespace", "which column is this token in", and
"is this line in the summary block or the item table" — and those questions
generalise across vendors where a template does not.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/invoices/upload` | Accept a file, queue it, return a document number |
| `GET` | `/api/v1/invoices/{id}/status` | Progress: lifecycle status, pipeline stage, percentage |
| `GET` | `/api/v1/invoices/{id}/result` | Fields, evidence boxes and confidence, in one call |
| `GET` | `/api/v1/invoices/{id}/pages/{n}` | Page rendered as PNG, for the review overlay |
| `POST` | `/api/v1/invoices/{id}/corrections` | Save a reviewer's corrections |
| `POST` | `/api/v1/invoices/{id}/reprocess` | Run the pipeline again over the stored file |

`{id}` accepts either the human-facing `DOC-2026-000001` or the internal UUID.

Accepted formats: PDF, JPG/JPEG, PNG, TIFF (including multi-page), WEBP.

### Status is two fields, not one

```json
{
  "document_id": "DOC-2026-000001",
  "status": "PROCESSING",
  "progress": 65,
  "current_stage": "TABLE_EXTRACTION",
  "stage_label": "Extracting tables"
}
```

`status` is the lifecycle — `UPLOADED`, `PROCESSING`, `COMPLETED`,
`REVIEW_REQUIRED`, `FAILED` — and is what a list view filters on.
`current_stage` is the position inside the pipeline and is what a progress
caption names. Collapsing them into one field would mean every caller had to
know that `OCR_PROCESSING` and `TABLE_EXTRACTION` both mean "not finished",
and adding a stage would break all of them.

`progress` is derived from the stage using measured wall-clock weights. OCR
holds about 60% of the bar because it holds about 60% of the time; equal
slices would show a bar that leaps to 60% and then sits still for the whole
actual wait.

## Which OCR engine runs

Two backends, selected by `OCR_BACKEND`, running the same PP-OCR networks:

- **`onnx`** (default) — ONNX Runtime, via `rapidocr`. About seven times
  faster per page on CPU than paddle, and about twenty times cheaper to
  construct. Its models ship inside the wheel.
- **`paddle`** — PaddleOCR. The fallback, selected automatically if
  `rapidocr` is not importable. Still what the ANPR path uses.

`app/modules/ocr/engine.py` is the only module that imports either, and both
normalise to the same `RawDetection`, so nothing downstream knows which ran.
Check `ocr.engine_ready backend=…` in the logs to see which one did.
See [ocr-performance.md](ocr-performance.md) for the measurements behind the
default.

## Air-gapped deployment

Neither backend may reach the network at run time.

The ONNX models are packaged inside the `rapidocr` wheel, so `pip install`
during the image build is all they need. The paddle weights are downloaded
**at image build time** and baked in:

```dockerfile
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', ...)"
```

Without this the image passes CI (which has a network) and fails on the
customer's machine at the first upload. The build asserts that both sets of
weights are present, so a broken build fails loudly rather than shipping.

Two environment variables carry the paddle half of the air-gapped claim, and
both have names that cannot be guessed:

- **`PADDLE_PDX_CACHE_HOME`** is where PaddleX caches weights
  (`paddlex/utils/cache.py`). Any other name — `PADDLEX_HOME`, say — is
  silently ignored, and the runtime falls back to `~/.paddlex`, which is
  empty.
- **`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`** stops PaddleX probing
  huggingface.co, modelscope, aistudio and bos on startup to see which are
  reachable. The weights are already local, so the probe finds nothing out —
  but on an isolated network it is a series of TCP timeouts standing between
  the worker and the first invoice.

Build once where there is a network, then move the image:

```bash
docker compose build
docker save iotiq-api:latest | gzip > iotiq-api.tar.gz
# on the target host
gunzip -c iotiq-api.tar.gz | docker load
docker compose up -d
```

### What `compose.yml` gets right that the obvious version does not

- **A shared storage volume.** The API writes the uploaded file and the
  worker reads it back. With per-container filesystems the worker looks for a
  file that only ever existed in the API container.
- **A shared database.** `sqlite:///./iotiq.db` gives the API and the worker
  one file *each*, so progress the worker writes is invisible to the status
  endpoint the client is polling, and the document sits at 0% forever.
- **`CELERY_ENABLED=true`.** Left unset it defaults to false: the API
  processes everything in-process and the worker container sits idle.
- **Redis is not published to the host.** An exposed Redis with no auth is
  the classic on-premise foothold.

## Configuration

| Setting | Default | Notes |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | 25 | Holds a 20-page 300-DPI colour scan |
| `OCR_RENDER_DPI` | 300 | Floor for reliable recognition of 8pt text; cost grows ~quadratically |
| `OCR_MIN_CONFIDENCE` | 0.30 | Detections below this are dropped as noise. Low by design — a dropped line cannot be recovered downstream, while retained noise merely competes and loses |
| `OCR_LANGUAGE` | `en` | |
| `OCR_DEBUG_ARTIFACTS` | false | Writes intermediate images; multiplies storage per document |
| `CELERY_ENABLED` | false | Without a broker the API processes in-process, which works but does not survive a restart |

## Performance

OCR dominates everything else by two orders of magnitude — roughly **two
minutes per page** on a CPU build. Everything downstream of it is
milliseconds. Two consequences worth planning around:

- A page whose text layer is intact is never sent to OCR. A page carrying no
  raster images is read natively *however sparse its text*, because OCR can
  only recover what is in pixels and there are none it does not already have.
- Worker concurrency is set to 2, not the CPU count: PaddleOCR is already
  multi-threaded, and four parallel document builders measured ~2.1× rather
  than 4× (see `app/ml/ocr/README.md`).

## Confidence and review

Confidence decides whether an invoice posts automatically or goes to a human,
so it is built from evidence that is genuinely independent: how well the
label matched, how far clear the winning value was of the runner-up, what the
recogniser's own confidence was, and whether the value survives cross-checks.

Two rules carry most of the weight:

- **A validation error caps the document.** Arithmetic that does not
  reconcile proves at least one field is wrong, which no amount of per-field
  confidence can see.
- **The margin over the runner-up counts.** A value that won by a hair was
  very nearly a different number, and that is exactly what a human should
  look at — no first-match-wins scheme can even tell you such a field exists.

Note that *proximity and region priors deliberately do not lower confidence*.
They rank candidates against each other; a summary amount right-aligned far
from its label is not suspicious, it is what a summary block looks like.

## Corrections are training data

`POST /corrections` records what was read alongside what was right, in
`invoice_field_corrections`. The difference between the two is the only real
training signal this system will ever get, and it is worth more than any
amount of synthetic data.

## Regression testing

`tests/test_golden_set.py` measures per-field accuracy across a corpus of
distinct layouts and fails the build when a field's accuracy drops.

Per field, not per document: "78% of invoices were perfect" is not
actionable, while "invoice_number 100%, total_amount 83%" says where to spend
the next hour.

To add a real vendor invoice, drop it in `tests/golden/real/` next to a
`<name>.expected.json`. Nothing needs registering. It will probably breach
the floors, and that is the point — a document the extractor gets wrong
should fail the build until someone decides whether to fix the extractor or
re-baseline.

```bash
pytest tests/test_golden_set.py -s     # -s prints the accuracy table
```
