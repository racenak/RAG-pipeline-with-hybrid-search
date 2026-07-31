"""Pydantic schemas for ingestion endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestURLRequest(BaseModel):
    """Request body for URL ingestion."""

    url: str = Field(..., description="URL to ingest via Firecrawl", examples=["https://docs.example.com"])


class IngestDirRequest(BaseModel):
    """Request body for directory ingestion."""

    directory: str = Field(..., description="Directory path to scan", examples=["./docs"])
    recursive: bool = Field(True, description="Scan subdirectories recursively")


class IngestResponse(BaseModel):
    """Response for a single ingestion result."""

    document_id: str
    source: str
    source_type: str
    success: bool
    error: str | None = None
    seaweedfs_fid: str | None = None


class IngestBatchResponse(BaseModel):
    """Response for batch ingestion."""

    total: int
    successful: int
    failed: int
    results: list[IngestResponse]
