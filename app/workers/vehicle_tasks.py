from app.workers.celery_app import celery_app


@celery_app.task
def detect_number_plate(document_id: str) -> dict:
    """Worker boundary for YOLO/OpenCV/PaddleOCR deployment."""
    return {"document_id": document_id, "status": "detection_required"}


@celery_app.task
def send_document_expiry_reminders() -> dict:
    return {"status": "completed"}
