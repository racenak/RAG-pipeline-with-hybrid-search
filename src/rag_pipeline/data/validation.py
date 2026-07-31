"""File validation — MIME type, size limits, SHA-256 dedup."""

from __future__ import annotations

import hashlib
import mimetypes
from typing import TYPE_CHECKING

from rag_pipeline.data.models import ValidationResult

if TYPE_CHECKING:
    from pathlib import Path

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
}

SUFFIX_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
}


def compute_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_mime(path: Path) -> str | None:
    """Detect MIME type from extension (preferred) or content."""
    suffix = path.suffix.lower()
    if suffix in SUFFIX_TO_MIME:
        return SUFFIX_TO_MIME[suffix]
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type


def validate_file(path: Path) -> ValidationResult:
    """Validate a file for ingestion.

    Checks: existence, readability, non-empty, size limit, supported format.
    Returns ValidationResult with file_hash on success.
    """
    if not path.exists():
        return ValidationResult(valid=False, error=f"File not found: {path}")

    if not path.is_file():
        return ValidationResult(valid=False, error=f"Not a file: {path}")

    if path.stat().st_size == 0:
        return ValidationResult(valid=False, error="File is empty")

    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return ValidationResult(valid=False, error="File exceeds 100MB limit")

    mime_type = detect_mime(path)
    if mime_type not in SUPPORTED_MIME_TYPES:
        return ValidationResult(valid=False, error=f"Unsupported format: {mime_type}")

    file_hash = compute_hash(path)
    return ValidationResult(valid=True, file_hash=file_hash)
