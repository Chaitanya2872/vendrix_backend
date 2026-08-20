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
    # --- stuck-document recovery -------------------------------------------
    # A queued task is not a performed one. If Celery is enabled but nothing
    # is consuming the queue, uploads sit at UPLOADED indefinitely and the UI
    # waits on an extraction nobody is running. These bound the sweep that
    # picks those up — see modules/documents/dispatch.py.
    documents_recover_stuck_on_startup: bool = True
    # How old a document must be before "still processing" means "abandoned"
    # rather than "in flight". Comfortably longer than the slowest document.
    documents_stuck_after_minutes: int = 5
    # Ceiling on one sweep. A queue broken for a week must not turn the next
    # restart into an hour of OCR before the service becomes usable.
    documents_recovery_limit: int = 25
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
    # Longest side, in pixels, an image is scaled to before OCR. A 300-DPI A4
    # page is ~3500px, and the detector resizes internally to a fraction of
    # that anyway — so the extra pixels are rendered, copied and discarded.
    # Measured: cropping to 1600 changed neither the line count nor the text
    # on the sample invoices. 0 disables the cap.
    ocr_max_image_side: int = 1600
    # Which runtime executes the OCR models: "onnx" (ONNX Runtime, via the
    # rapidocr package) or "paddle" (PaddlePaddle). Same PP-OCR networks
    # either way — only the executor differs, and it differs by a factor of
    # seven on CPU: ~5s versus ~38s for the same invoice page, with identical
    # line counts. Falls back to paddle automatically if rapidocr is missing.
    # See docs/ocr-performance.md.
    ocr_backend: str = "onnx"
    # Paddle-backend model names. The "mobile" pair is ~3x faster per page
    # than the medium pair paddle picks by default from `lang`, at no
    # measurable accuracy cost on invoice text. Blank falls back to paddle's
    # own choice for `ocr_language`. Unused by the ONNX backend, which uses
    # the models packaged with rapidocr.
    ocr_detection_model: str = "PP-OCRv5_mobile_det"
    ocr_recognition_model: str = "PP-OCRv5_mobile_rec"
    # Paddle backend only: how many detected text lines are recognised per
    # forward pass.
    #
    # Left at 1 on purpose, against the usual intuition. Measured on a
    # 71-line invoice with the mobile recogniser: batch 1 took 34s, batch 8
    # took 68s. Text-line crops have wildly different widths, so a batch is
    # padded to its widest member and paddle re-specialises for each new
    # input shape — batching buys padding and recompilation, not throughput.
    # Raise it only with a measurement on the same build.
    ocr_recognition_batch_size: int = 1
    # Threads each runtime may use for one page. Match to the deployment's
    # core allocation; both backends read it.
    ocr_cpu_threads: int = 8
    # Build the OCR engine on a background thread at startup. Construction is
    # a fixed per-process cost of tens of seconds; paying it here rather than
    # inside the first upload is the difference between "extraction is slow"
    # and "the first extraction of the day is slow".
    ocr_warm_up_on_startup: bool = True
    # Pages OCR'd per document. Invoice headers and totals are on the first
    # page or two; a 40-page annexure scanned into the same PDF is not worth
    # the wall-clock, and reading it would not change a single parsed field.
    ocr_max_pages: int = 3
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
