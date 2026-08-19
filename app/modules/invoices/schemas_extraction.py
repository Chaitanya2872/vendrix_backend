"""API contracts for the invoice extraction endpoints.

Separate from schemas.py, which holds the manual-CRUD contracts the generic
`attach_crud` helper is wired to. Mixing pipeline payloads into that file
would make it unclear which models are part of the CRUD surface.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class UploadAccepted(BaseModel):
    """202-style acknowledgement: the file is stored and queued, nothing has
    been extracted yet. Deliberately does not carry extracted fields — OCR
    takes minutes, and a client that expected them in this response would
    have to be rewritten the first time someone uploads a ten-page scan."""

    document_id: str = Field(examples=["DOC-2026-000001"])
    filename: str
    status: str = Field(examples=["PROCESSING"])
    file_format: str = Field(examples=["pdf"])
    size_bytes: int
    page_count: int | None = None
    duplicate_of: str | None = Field(
        default=None,
        description="Set when an identical file was already uploaded; that "
                    "document's number, so the client can jump to it rather "
                    "than wait for a second identical extraction.",
    )


class ProcessingError(BaseModel):
    code: str
    message: str


class ProcessingStatus(BaseModel):
    """Progress of one document.

    `status` is the lifecycle (is it still working?) and `current_stage` is
    the position within the pipeline. Both are present because they answer
    different questions and a client needs both: one drives a spinner, the
    other drives its caption.
    """

    document_id: str
    filename: str
    status: str = Field(examples=["PROCESSING"])
    progress: int = Field(ge=0, le=100, examples=[65])
    current_stage: str = Field(examples=["TABLE_EXTRACTION"])
    stage_label: str = Field(examples=["Extracting tables"])
    attempt: int | None = None
    page_count: int | None = None
    used_ocr: bool | None = None
    ocr_confidence: float | None = None
    extraction_confidence: float | None = None
    duration_seconds: float | None = None
    error: ProcessingError | None = None
