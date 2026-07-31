"""Schemas for evaluation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    mode: str = Field(
        default="retrieval", description="Evaluation mode: retrieval|generation|both"
    )
    dataset_path: str | None = Field(default=None, description="Path to golden dataset")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to retrieve")
    category: str | None = Field(default=None, description="Filter by category")
    difficulty: str | None = Field(default=None, description="Filter by difficulty")


class EvalResponse(BaseModel):
    status: str
    total_queries: int
    metrics: dict
    report_path: str | None = None
