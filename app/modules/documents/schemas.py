from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentListItem(BaseModel):
    """Lean projection used by the list endpoint — omits extracted_fields, which can hold up to ~10k characters of OCR text per document and was the main cause of list latency."""
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    content_type: str
    document_type: str
    status: str
    owner_id: str
    review_confirmed_at: datetime | None
    created_at: datetime
