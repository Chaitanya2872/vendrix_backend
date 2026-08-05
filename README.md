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

Document uploads are stored under `storage/` in development. Configure MinIO/S3 in production and ensure that image OCR workers have PaddleOCR and the required model assets installed.

For number-plate photos, mount an ANPR-trained YOLO weights file and set `PLATE_DETECTOR_MODEL_PATH` as shown in `.env.example`. The API exposes `POST /api/v1/anpr/recognize` for image recognition and `POST /api/v1/anpr/lookup` for text-only lookup. OCR extraction is queued at document upload and persisted as `REVIEW_REQUIRED` for user confirmation.
