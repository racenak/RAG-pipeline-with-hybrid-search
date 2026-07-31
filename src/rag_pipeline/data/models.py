"""Data models for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Section:
    """A structural section extracted from a document."""

    level: int
    title: str
    offset: int = 0


@dataclass
class ParsedDocument:
    """Output of a parser — raw text + structural metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of file validation."""

    valid: bool
    error: str | None = None
    file_hash: str | None = None


@dataclass
class Document:
    """A validated, parsed document ready for chunking."""

    id: str  # SHA-256 of content
    source: str  # file path or URL
    source_type: str  # "file" | "url"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class IngestionResult:
    """Result of ingesting a single document."""

    document_id: str
    source: str
    source_type: str
    success: bool
    error: str | None = None
    chunks_count: int = 0
    seaweedfs_fid: str | None = None
