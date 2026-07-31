"""Pydantic models for RAG evaluation cases."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QueryCategory(StrEnum):
    """Categories of evaluation queries."""

    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    EDGE_CASE = "edge_case"


class DifficultyLevel(StrEnum):
    """Difficulty levels for evaluation cases."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalCase(BaseModel):
    """A single evaluation case with query, expected answer, and metadata."""

    id: str
    query: str
    expected_answer: str
    expected_documents: list[str] = Field(default_factory=list)
    category: QueryCategory
    difficulty: DifficultyLevel
    metadata: dict = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A collection of evaluation cases with versioning."""

    version: str
    description: str
    cases: list[EvalCase]
