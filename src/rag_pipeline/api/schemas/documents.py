"""Document management API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentItem(BaseModel):
    """Single document in listing."""

    id: str
    filename: str
    source_url: str | None = None
    mime_type: str | None = None
    file_hash: str | None = None
    size_bytes: int | None = None
    chunk_count: int = 0
    created_at: str | None = None


class DocumentListResponse(BaseModel):
    """Document listing response."""

    total: int
    count: int
    offset: int
    limit: int
    documents: list[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    """Document deletion response."""

    deleted: bool
    document_id: str
    message: str
