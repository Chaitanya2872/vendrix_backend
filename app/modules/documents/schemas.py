from datetime import date, datetime
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
    # Cheap scalars the Documents list filters and sorts on. They are a few
    # bytes each, so including them costs nothing against the reason
    # extracted_fields is excluded, and leaving them out would force the page
    # to fetch every document individually just to draw its filter chips.
    vendor_id: str | None = None
    expires_on: date | None = None
    size_bytes: int | None = None
    review_confirmed_at: datetime | None
    created_at: datetime
