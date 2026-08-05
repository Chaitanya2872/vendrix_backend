from app.workers.celery_app import celery_app


@celery_app.task
def process_invoice(invoice_id: str) -> dict:
    return {"invoice_id": invoice_id, "status": "processed"}


@celery_app.task
def generate_report(report_type: str, filters: dict | None = None) -> dict:
    return {"report_type": report_type, "filters": filters or {}, "status": "generated"}


@celery_app.task
def schedule_recurring_bill(bill_id: str) -> dict:
    return {"bill_id": bill_id, "status": "scheduled"}
