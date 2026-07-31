"""Pipeline orchestrator — full document ingestion end-to-end.

File → Validate → Parse → Clean → S3 → Chunk → Embed → OpenSearch → PostgreSQL
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003

from rag_pipeline.data.chunking import ChunkingConfig, TextChunker
from rag_pipeline.data.cleaning import CleaningConfig, TextCleaner
from rag_pipeline.data.parsers import get_parser
from rag_pipeline.data.validation import validate_file

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    document_id: str
    source: str
    source_type: str
    success: bool
    chunks_count: int = 0
    indexed_in_opensearch: bool = False
    registered_in_postgres: bool = False
    s3_key: str | None = None
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


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


def _get_chunker() -> TextChunker:
    """Create a TextChunker from config."""
    try:
        from rag_pipeline.config import get_settings
        settings = get_settings()
        cfg = settings.chunking
        config = ChunkingConfig(
            max_tokens=cfg.max_tokens,
            min_tokens=cfg.min_tokens,
            overlap_tokens=cfg.overlap_tokens,
        )
        return TextChunker(config)
    except Exception:
        return TextChunker()


def _get_embedder():
    """Create an EmbeddingGenerator from config."""
    from rag_pipeline.embeddings.generator import EmbeddingGenerator
    try:
        from rag_pipeline.config import get_settings
        settings = get_settings()
        cfg = settings.embedding
        return EmbeddingGenerator(
            model_name=cfg.model,
            device=cfg.device,
            normalize=cfg.normalize,
            batch_size=cfg.batch_size,
            cache_enabled=cfg.cache_enabled,
            cache_dir=cfg.cache_dir,
        )
    except Exception:
        return EmbeddingGenerator()


def _get_opensearch_client():
    """Create an OpenSearchClient from config."""
    from rag_pipeline.storage.opensearch import OpenSearchClient
    try:
        from rag_pipeline.config import get_settings
        settings = get_settings()
        s = settings.storage
        return OpenSearchClient(
            host=s.opensearch_host,
            port=s.opensearch_port,
            scheme=s.opensearch_scheme,
            username=s.opensearch_username,
            password=s.opensearch_password,
            timeout=s.opensearch_timeout,
        )
    except Exception:
        return OpenSearchClient()


def _get_postgres_client():
    """Create a PostgresClient from config."""
    from rag_pipeline.storage.postgres import PostgresClient
    try:
        from rag_pipeline.config import get_settings
        settings = get_settings()
        s = settings.storage
        return PostgresClient(
            host=s.postgres_host,
            port=s.postgres_port,
            database=s.postgres_database,
            user=s.postgres_user,
            password=s.postgres_password,
        )
    except Exception:
        return PostgresClient()


def _get_index_name() -> str:
    """Get the OpenSearch index name from config."""
    try:
        from rag_pipeline.config import get_settings
        settings = get_settings()
        return f"{settings.storage.opensearch_index_prefix}-chunks-v1"
    except Exception:
        return "rag-chunks-v1"


# ------------------------------------------------------------------ #
#  Main pipeline
# ------------------------------------------------------------------ #


def ingest_file_full(
    path: Path,
    storage=None,
    embedder=None,
    opensearch_client=None,
    postgres_client=None,
    index_name: str | None = None,
) -> PipelineResult:
    """Full pipeline: file → validate → parse → clean → S3 → chunk → embed → OpenSearch → PostgreSQL.

    Args:
        path: Path to the file to ingest.
        storage: S3Storage instance (optional, skips S3 if None).
        embedder: EmbeddingGenerator instance (optional, creates from config if None).
        opensearch_client: OpenSearchClient instance (optional, creates from config if None).
        postgres_client: PostgresClient instance (optional, creates from config if None).
        index_name: OpenSearch index name (optional, auto-generated if None).

    Returns:
        PipelineResult with success/error status and metadata.
    """
    import time

    timings: dict[str, float] = {}
    doc_id = ""

    # 1. Validate
    t0 = time.monotonic()
    validation = validate_file(path)
    timings["validate"] = time.monotonic() - t0
    if not validation.valid:
        return PipelineResult(
            document_id="",
            source=str(path),
            source_type="file",
            success=False,
            error=validation.error,
            timings=timings,
        )
    doc_id = validation.file_hash or str(uuid.uuid4())

    # 1b. Dedup check — skip if file_hash already exists in PostgreSQL
    if validation.file_hash:
        try:
            pg = _get_postgres_client()
            existing = pg.find_by_file_hash(validation.file_hash)
            if existing:
                logger.info("Duplicate file detected (hash=%s), skipping: %s", validation.file_hash[:12], path.name)
                return PipelineResult(
                    document_id=existing.id,
                    source=str(path),
                    source_type="file",
                    success=True,
                    chunks_count=existing.chunk_count,
                    indexed_in_opensearch=True,
                    registered_in_postgres=True,
                    s3_key=None,
                    timings=timings,
                )
        except Exception as e:
            logger.warning("Dedup check failed (continuing): %s", e)

    # 2. Parse
    t0 = time.monotonic()
    parser = get_parser(path)
    parsed = parser.parse(path)
    timings["parse"] = time.monotonic() - t0

    # 3. Clean
    t0 = time.monotonic()
    cleaner = _get_cleaner()
    format_type = parsed.metadata.get("format", "generic")
    cleaned_text, stats = cleaner.clean(parsed.content, text_format=format_type)
    timings["clean"] = time.monotonic() - t0
    logger.debug(
        "Cleaned %s: %d → %d chars",
        path.name, stats.chars_before, stats.chars_after,
    )

    # 4. Upload to S3
    t0 = time.monotonic()
    s3_key: str | None = None
    if storage is not None:
        try:
            s3_key = storage.upload_file(path)
        except Exception as e:
            logger.warning("S3 upload failed: %s", e)
    timings["s3_upload"] = time.monotonic() - t0

    # 5. Chunk
    t0 = time.monotonic()
    chunker = _get_chunker()
    chunks = chunker.chunk(
        text=cleaned_text,
        document_id=doc_id,
        metadata={"source": str(path), "format": format_type, **parsed.metadata},
    )
    timings["chunk"] = time.monotonic() - t0
    logger.info("Chunked %s into %d chunks", path.name, len(chunks))

    if not chunks:
        return PipelineResult(
            document_id=doc_id,
            source=str(path),
            source_type="file",
            success=True,
            chunks_count=0,
            s3_key=s3_key,
            timings=timings,
        )

    # 6. Embed
    t0 = time.monotonic()
    if embedder is None:
        embedder = _get_embedder()
    embedder.embed_chunks(chunks)
    timings["embed"] = time.monotonic() - t0
    logger.info("Embedded %d chunks", len(chunks))

    # 7. Index in OpenSearch
    t0 = time.monotonic()
    if opensearch_client is None:
        opensearch_client = _get_opensearch_client()
    if postgres_client is None:
        postgres_client = _get_postgres_client()
    if index_name is None:
        index_name = _get_index_name()
    try:
        opensearch_client.index_chunks(index_name, chunks)
        indexed = True
    except Exception as e:
        logger.error("OpenSearch indexing failed: %s", e)
        indexed = False
    timings["opensearch_index"] = time.monotonic() - t0

    # 8. Register in PostgreSQL
    t0 = time.monotonic()
    if postgres_client is None:
        postgres_client = _get_postgres_client()
    try:
        from rag_pipeline.storage.postgres import ChunkRecord, DocumentRecord

        postgres_client.insert_document(DocumentRecord(
            id=doc_id,
            filename=path.name,
            source_url=str(path),
            mime_type=parsed.metadata.get("mime_type"),
            file_hash=validation.file_hash,
            size_bytes=path.stat().st_size if path.exists() else None,
            chunk_count=len(chunks),
        ))

        postgres_client.insert_chunks([
            ChunkRecord(
                id=c.id,
                document_id=doc_id,
                chunk_index=c.index,
                content=c.content,
                token_count=c.token_count,
                indexed=indexed,
            )
            for c in chunks
        ])
        registered = True
    except Exception as e:
        logger.error("PostgreSQL registration failed: %s", e)
        registered = False
    timings["postgres_register"] = time.monotonic() - t0

    return PipelineResult(
        document_id=doc_id,
        source=str(path),
        source_type="file",
        success=True,
        chunks_count=len(chunks),
        indexed_in_opensearch=indexed,
        registered_in_postgres=registered,
        s3_key=s3_key,
        timings=timings,
    )


# ------------------------------------------------------------------ #
#  Crawl pipeline
# ------------------------------------------------------------------ #


@dataclass
class CrawlResult:
    """Result of a crawl + ingest run."""

    source_url: str
    crawl_group: str
    pages_crawled: int = 0
    pages_succeeded: int = 0
    total_chunks: int = 0
    indexed_in_opensearch: bool = False
    registered_in_postgres: bool = False
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def _slug_from_url(url: str) -> str:
    """Extract a filename slug from a URL path."""
    from urllib.parse import urlparse

    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "index"
    # Sanitize
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in slug)
    return slug.lower()


def _crawl_group_from_url(url: str) -> str:
    """Derive a crawl_group name from the seed URL.

    e.g. https://docs.prefect.io/v3/concepts → prefect-v3-concepts
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    path_parts = [p for p in parsed.path.split("/") if p]

    # Remove trailing "index" or common page names
    while path_parts and path_parts[-1] in ("index", "index.html"):
        path_parts.pop()

    return f"{host.replace('.', '-')}-{'-'.join(path_parts)}" if path_parts else host.replace(".", "-")


def crawl_and_ingest(
    url: str,
    limit: int = 100,
    storage=None,
    embedder=None,
    opensearch_client=None,
    postgres_client=None,
    index_name: str | None = None,
    api_key: str | None = None,
) -> CrawlResult:
    """Crawl a website and ingest all pages into the RAG pipeline.

    Flow: crawl_site → clean → S3 (prefect/v3-concepts/{slug}.md) → chunk → embed → OpenSearch → PostgreSQL

    Args:
        url: Seed URL to crawl from.
        limit: Max pages to crawl.
        storage: S3Storage instance (optional).
        embedder: EmbeddingGenerator instance (optional).
        opensearch_client: OpenSearchClient instance (optional).
        postgres_client: PostgresClient instance (optional).
        index_name: OpenSearch index name (optional).
        api_key: Firecrawl API key (optional, reads from env if None).

    Returns:
        CrawlResult with summary stats.
    """
    import time

    timings: dict[str, float] = {}
    crawl_group = _crawl_group_from_url(url)
    errors: list[str] = []

    # 1. Crawl
    t0 = time.monotonic()
    from rag_pipeline.data.fetchers import URLFetcher

    if api_key is None:
        try:
            from rag_pipeline.config import get_settings
            api_key = get_settings().firecrawl.api_key
        except Exception:
            pass

    fetcher = URLFetcher(api_key=api_key or "")
    pages = fetcher.crawl_site(url, limit=limit)
    timings["crawl"] = time.monotonic() - t0
    logger.info("Crawled %d pages from %s", len(pages), url)

    # 2. Prepare shared resources
    cleaner = _get_cleaner()
    chunker = _get_chunker()
    if embedder is None:
        embedder = _get_embedder()
    if opensearch_client is None:
        opensearch_client = _get_opensearch_client()
    if postgres_client is None:
        postgres_client = _get_postgres_client()
    if index_name is None:
        index_name = _get_index_name()

    all_chunks = []
    pages_succeeded = 0
    t_process = time.monotonic()

    # 3. Process each page
    for page in pages:
        if not page.success:
            errors.append(f"{page.source_url}: {page.error}")
            continue

        # 3a. Clean
        cleaned_text, _ = cleaner.clean(page.content, text_format="markdown")

        # 3b. Upload to S3
        s3_key: str | None = None
        if storage is not None:
            try:
                slug = _slug_from_url(page.source_url)
                s3_key = f"crawl/{crawl_group}/{slug}.md"
                storage.upload_bytes(
                    data=cleaned_text.encode("utf-8"),
                    key=s3_key,
                )
            except Exception as e:
                logger.warning("S3 upload failed for %s: %s", page.source_url, e)

        # 3c. Chunk
        doc_id = f"{crawl_group}-{_slug_from_url(page.source_url)}"
        chunks = chunker.chunk(
            text=cleaned_text,
            document_id=doc_id,
            metadata={
                "source": "crawl",
                "crawl_group": crawl_group,
                "source_url": page.source_url,
                "title": page.metadata.get("title", ""),
                "s3_key": s3_key or "",
            },
        )

        if not chunks:
            continue

        # 3d. Embed
        embedder.embed_chunks(chunks)

        all_chunks.extend(chunks)
        pages_succeeded += 1

    timings["process_pages"] = time.monotonic() - t_process

    # 4. Bulk index in OpenSearch
    t0 = time.monotonic()
    try:
        opensearch_client.index_chunks(index_name, all_chunks)
        indexed = True
    except Exception as e:
        logger.error("OpenSearch bulk indexing failed: %s", e)
        indexed = False
    timings["opensearch_index"] = time.monotonic() - t0

    # 5. Register in PostgreSQL
    t0 = time.monotonic()
    try:
        from rag_pipeline.storage.postgres import ChunkRecord, DocumentRecord

        postgres_client.insert_document(DocumentRecord(
            id=crawl_group,
            filename=f"{crawl_group}/",
            source_url=url,
            mime_type="text/markdown",
            file_hash=None,
            size_bytes=None,
            chunk_count=len(all_chunks),
        ))

        postgres_client.insert_chunks([
            ChunkRecord(
                id=c.id,
                document_id=crawl_group,
                chunk_index=c.index,
                content=c.content,
                token_count=c.token_count,
                indexed=indexed,
            )
            for c in all_chunks
        ])
        registered = True
    except Exception as e:
        logger.error("PostgreSQL registration failed: %s", e)
        registered = False
    timings["postgres_register"] = time.monotonic() - t0

    return CrawlResult(
        source_url=url,
        crawl_group=crawl_group,
        pages_crawled=len(pages),
        pages_succeeded=pages_succeeded,
        total_chunks=len(all_chunks),
        indexed_in_opensearch=indexed,
        registered_in_postgres=registered,
        errors=errors,
        timings=timings,
    )
