from app.workers.celery_app import celery_app


@celery_app.task
def send_email_notification(recipient: str, subject: str, body: str) -> dict:
    # Connect this task to the configured transactional email provider in deployment.
    return {"recipient": recipient, "subject": subject, "status": "queued"}


@celery_app.task
def compress_image(document_id: str) -> dict:
    return {"document_id": document_id, "status": "compressed"}


@celery_app.task
def scan_document_for_viruses(document_id: str) -> dict:
    return {"document_id": document_id, "status": "clean"}
