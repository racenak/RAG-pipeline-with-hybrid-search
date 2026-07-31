"""Search API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., min_length=1, description="Search query text")
    mode: str = Field(
        default="hybrid",
        pattern="^(hybrid|bm25|vector)$",
        description="Search mode: hybrid, bm25, or vector",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    metadata_filter: dict[str, str] | None = Field(
        default=None, description="Metadata filters (e.g. {crawl_group: '...'})"
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class SearchResultItem(BaseModel):
    """Single search result."""

    id: str
    score: float
    content: str
    metadata: dict[str, str] | None = None


class SearchResponse(BaseModel):
    """Search response body."""

    query: str
    mode: str
    count: int
    elapsed_ms: float
    results: list[SearchResultItem]
