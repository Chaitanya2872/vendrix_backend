from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IoTIQ Vendor Management API"
    environment: str = "development"
    database_url: str = "sqlite:///./iotiq.db"
    jwt_secret: str = "change-this-development-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    storage_path: str = "./storage"
    # Upload ceiling. 25MB comfortably holds a 20-page 300-DPI colour scan;
    # anything larger is usually a photo burst or a mis-selected file, and
    # accepting it would tie up a worker for an hour.
    max_upload_size_mb: int = 25
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_enabled: bool = False
    plate_detector_model_path: str | None = None
    # Trained OCR field-extraction model (app/ml/ocr). Left unset, the packaged
    # artifact is used if present and the parser falls back to deterministic
    # extraction if it is not — so an untrained deployment still works.
    ocr_field_model_path: str | None = None
    # Fields the model proposes below this probability are ignored, so a weak
    # guess never displaces a value the deterministic parser was sure of.
    ocr_field_model_min_confidence: float = 0.35
    # --- OCR stage ---------------------------------------------------------
    ocr_language: str = "en"
    # Resolution PDF pages are rasterised at before OCR. 300 is the floor for
    # reliable recognition of 8pt invoice text; going higher costs runtime
    # roughly quadratically for little accuracy gain.
    ocr_render_dpi: int = 300
    # Detections scoring below this are discarded as noise (stamps, logos,
    # scan artefacts). Low by design: a dropped line cannot be recovered
    # downstream, while a little noise merely competes and loses on score.
    ocr_min_confidence: float = 0.30
    # Write per-stage intermediate images to storage for debugging. Off by
    # default — it multiplies storage per document by the number of stages.
    ocr_debug_artifacts: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
