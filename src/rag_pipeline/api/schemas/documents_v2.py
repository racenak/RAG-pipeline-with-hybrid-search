"""Schemas for incremental ingestion API (documents v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IncrementalIngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to file")
    index_name: str = Field(default="rag", description="OpenSearch index name")


class BatchIngestRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
    index_name: str = Field(default="rag")


class ReindexRequest(BaseModel):
    index_name: str = Field(default="rag")


class IngestionJobResponse(BaseModel):
    job_id: str
    document_path: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    chunks_indexed: int = 0
    total_chunks: int = 0


class BatchIngestResponse(BaseModel):
    jobs: list[IngestionJobResponse]
    total: int


class ReindexResponse(BaseModel):
    jobs: list[IngestionJobResponse]
    total: int
