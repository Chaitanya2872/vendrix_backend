# IoTIQ Vendor Management Backend

FastAPI backend for vendor onboarding, fleet compliance, invoice approvals, payments, document review and internal ANPR lookup.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`. The initial administrator is `admin@iotiq.example.com` / `Admin@123`; change it before deployment.

Use Docker Compose to run the API, Redis and worker: `docker compose up --build`.

Document uploads are stored under `storage/` in development. Configure MinIO/S3 in production.

OCR runs on ONNX Runtime by default (`OCR_BACKEND=onnx`), with PaddleOCR as
the fallback; both run the same PP-OCR models and the weights for both are
baked into the Docker image at build time, so an air-gapped deployment needs
no network at run time. See [docs/invoice-ocr.md](docs/invoice-ocr.md) for the
invoice extraction pipeline, its endpoints, configuration and the golden-set
regression suite, and [docs/ocr-performance.md](docs/ocr-performance.md) for
where the time goes and which settings move it.

Gate security records inward and outward vehicle movements through
`/api/v1/vehicle-entries`: one record per visit, signed out with
`POST /api/v1/vehicle-entries/{id}/exit`, with weighbridge readings and a
live "who is on the premises" view. See [docs/api.md](docs/api.md).

For number-plate photos, mount an ANPR-trained YOLO weights file and set `PLATE_DETECTOR_MODEL_PATH` as shown in `.env.example`. The API exposes `POST /api/v1/anpr/recognize` for image recognition and `POST /api/v1/anpr/lookup` for text-only lookup. OCR extraction is queued at document upload and persisted as `REVIEW_REQUIRED` for user confirmation.

See [docs/api.md](docs/api.md) for the endpoint reference.
