"""Generation API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request body for generation endpoint."""

    query: str = Field(..., min_length=1, description="User query")
    mode: str = Field(
        default="hybrid",
        pattern="^(hybrid|bm25|vector)$",
        description="Search mode",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to use")
    model: str | None = Field(default=None, description="Override LLM model")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class CitationResponse(BaseModel):
    """Single citation in the response."""

    marker: str
    source_document: str
    chunk_id: str
    chunk_index: int
    page: int | None = None
    score: float = 0.0
    text_snippet: str = ""


class CitationBundleResponse(BaseModel):
    """Citation bundle in the response."""

    citations: list[CitationResponse] = Field(default_factory=list)
    formatted_sources: str = ""
    validation_warnings: list[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    """Response body for generation endpoint."""

    answer: str
    context_used: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    sources: list[dict[str, str]] = Field(default_factory=list)
    citations: CitationBundleResponse | None = None
