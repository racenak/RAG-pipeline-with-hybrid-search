"""Ingestion orchestrator — single file, batch, and URL ingestion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from rag_pipeline.data.fetchers import URLFetcher
    from rag_pipeline.data.storage import S3Storage

from rag_pipeline.data.cleaning import CleaningConfig, TextCleaner
from rag_pipeline.data.models import IngestionResult
from rag_pipeline.data.parsers import get_parser
from rag_pipeline.data.validation import validate_file

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv"}


def _get_cleaner() -> TextCleaner:
    """Create a TextCleaner from config."""
    try:
        from rag_pipeline.config import get_settings

        settings = get_settings()
        cfg = settings.cleaning
        config = CleaningConfig(
            fix_encoding=cfg.fix_encoding,
            normalize_unicode=cfg.normalize_unicode,
            decode_html_entities=cfg.decode_html_entities,
            remove_control_chars=cfg.remove_control_chars,
            normalize_whitespace=cfg.normalize_whitespace,
            collapse_blank_lines=cfg.collapse_blank_lines,
            max_blank_lines=cfg.max_blank_lines,
            clean_pdf_artifacts=cfg.clean_pdf_artifacts,
            strip_residual_html=cfg.strip_residual_html,
        )
        return TextCleaner(config)
    except Exception:
        return TextCleaner()


def ingest_file(
    path: Path,
    storage: S3Storage | None = None,
) -> IngestionResult:
    """Ingest a single file through validation → parse → clean → store in S3."""
    validation = validate_file(path)
    if not validation.valid:
        return IngestionResult(
            document_id="",
            source=str(path),
            source_type="file",
            success=False,
            error=validation.error,
        )

    # Parse
    parser = get_parser(path)
    parsed = parser.parse(path)

    # Clean
    cleaner = _get_cleaner()
    format_type = parsed.metadata.get("format", "generic")
    _, stats = cleaner.clean(parsed.content, text_format=format_type)
    logger.debug(
        "Cleaned %s: %d → %d chars (%d removed)",
        path.name, stats.chars_before, stats.chars_after, stats.chars_removed,
    )

    # Upload raw file to S3 (may raise on connection failure)
    fid: str | None = None
    if storage is not None:
        fid = storage.upload_file(path)
        logger.info("Stored %s in S3: %s", path.name, fid)

    return IngestionResult(
        document_id=validation.file_hash or "",
        source=str(path),
        source_type="file",
        success=True,
        seaweedfs_fid=fid,
    )


def ingest_url(
    url: str,
    fetcher: URLFetcher,
    storage: S3Storage | None = None,
) -> IngestionResult:
    """Ingest a single URL through Firecrawl → clean → store in S3."""
    fetched = fetcher.fetch_url(url)
    if not fetched.success:
        return IngestionResult(
            document_id="",
            source=url,
            source_type="url",
            success=False,
            error=fetched.error,
        )

    # Clean fetched content
    cleaner = _get_cleaner()
    cleaned_text, stats = cleaner.clean(fetched.content, text_format="markdown")
    logger.debug(
        "Cleaned URL content: %d → %d chars (%d removed)",
        stats.chars_before, stats.chars_after, stats.chars_removed,
    )

    # Upload cleaned content to S3 (may raise on connection failure)
    fid: str | None = None
    if storage is not None:
        fid = storage.upload_bytes(
            data=cleaned_text.encode("utf-8"),
            key=f"urls/{fetched.content_hash}.md",
        )
        logger.info("Stored URL content in S3: %s", fid)

    return IngestionResult(
        document_id=fetched.content_hash,
        source=url,
        source_type="url",
        success=True,
        seaweedfs_fid=fid,
    )


def ingest_directory(
    directory: Path,
    storage: S3Storage | None = None,
    recursive: bool = True,
) -> Iterator[IngestionResult]:
    """Ingest all supported files in a directory."""
    for path in discover_files(directory, recursive):
        yield ingest_file(path, storage=storage)


def discover_files(directory: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield all ingestible files under a directory."""
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
