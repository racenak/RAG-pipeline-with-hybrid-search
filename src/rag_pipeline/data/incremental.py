"""Incremental ingestion — hash-based dedup, change detection, reindex."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from rag_pipeline.data.chunking import Chunk
from rag_pipeline.data.cleaning import CleaningConfig, TextCleaner
from rag_pipeline.data.chunking import ChunkingConfig, TextChunker
from rag_pipeline.data.parsers import get_parser
from rag_pipeline.data.validation import validate_file
from rag_pipeline.embeddings.generator import EmbeddingGenerator
from rag_pipeline.storage.opensearch import OpenSearchClient
from rag_pipeline.storage.postgres import ChunkRecord, DocumentRecord, PostgresClient
from rag_pipeline.data.storage import S3Storage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Status tracking
# ------------------------------------------------------------------ #


class IngestionStatus(Enum):
    """Lifecycle state of an ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"


@dataclass
class IngestionJob:
    """Tracks the state of a single ingestion operation."""

    job_id: str
    document_path: str
    status: IngestionStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    chunks_indexed: int = 0
    total_chunks: int = 0


class IngestionStatusTracker:
    """In-memory tracker for ingestion job statuses."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}

    def create(self, document_path: str) -> IngestionJob:
        """Register a new PENDING job."""
        job = IngestionJob(
            job_id=str(uuid4()),
            document_path=document_path,
            status=IngestionStatus.PENDING,
        )
        self._jobs[job.job_id] = job
        return job

    def start(self, job_id: str) -> None:
        """Transition job to PROCESSING."""
        job = self._jobs[job_id]
        job.status = IngestionStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)

    def complete(self, job_id: str, chunks_indexed: int, total_chunks: int) -> None:
        """Transition job to INDEXED."""
        job = self._jobs[job_id]
        job.status = IngestionStatus.INDEXED
        job.completed_at = datetime.now(timezone.utc)
        job.chunks_indexed = chunks_indexed
        job.total_chunks = total_chunks

    def fail(self, job_id: str, error_message: str) -> None:
        """Transition job to ERROR."""
        job = self._jobs[job_id]
        job.status = IngestionStatus.ERROR
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = error_message

    def get(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)


# ------------------------------------------------------------------ #
#  Content hashing
# ------------------------------------------------------------------ #


class ContentHashTracker:
    """SHA-256 hashing for files and text content."""

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of a file's content."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of a text string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def check_changed(file_path: str, known_hash: str) -> bool:
        """Return True if the file's current hash differs from *known_hash*."""
        current = ContentHashTracker.compute_file_hash(file_path)
        return current != known_hash


# ------------------------------------------------------------------ #
#  Incremental ingestor
# ------------------------------------------------------------------ #


def _get_cleaner() -> TextCleaner:
    try:
        from rag_pipeline.config import get_settings
        cfg = get_settings().cleaning
        return TextCleaner(CleaningConfig(
            fix_encoding=cfg.fix_encoding,
            normalize_unicode=cfg.normalize_unicode,
            decode_html_entities=cfg.decode_html_entities,
            remove_control_chars=cfg.remove_control_chars,
            normalize_whitespace=cfg.normalize_whitespace,
            collapse_blank_lines=cfg.collapse_blank_lines,
            max_blank_lines=cfg.max_blank_lines,
            clean_pdf_artifacts=cfg.clean_pdf_artifacts,
            strip_residual_html=cfg.strip_residual_html,
        ))
    except Exception:
        return TextCleaner()


def _get_chunker() -> TextChunker:
    try:
        from rag_pipeline.config import get_settings
        cfg = get_settings().chunking
        return TextChunker(ChunkingConfig(
            max_tokens=cfg.max_tokens,
            min_tokens=cfg.min_tokens,
            overlap_tokens=cfg.overlap_tokens,
        ))
    except Exception:
        return TextChunker()


class IncrementalIngestor:
    """Hash-aware incremental ingestion orchestrator."""

    def __init__(
        self,
        postgres: PostgresClient,
        opensearch: OpenSearchClient,
        embedder: EmbeddingGenerator,
        s3: S3Storage,
    ) -> None:
        self._postgres = postgres
        self._opensearch = opensearch
        self._embedder = embedder
        self._s3 = s3
        self._tracker = IngestionStatusTracker()
        self._hasher = ContentHashTracker()

    # ---- public API ---------------------------------------------------

    def ingest_incremental(
        self,
        file_path: str,
        index_name: str,
    ) -> IngestionJob:
        """Ingest a file, skipping if content hasn't changed.

        1. Compute file hash.
        2. Look up PostgreSQL for a document with that hash.
        3. If match → skip (return PENDING job with note in error_message).
        4. If changed → remove old data from OpenSearch, re-run pipeline.
        """
        job = self._tracker.create(file_path)

        try:
            file_hash = self._hasher.compute_file_hash(file_path)
        except OSError as exc:
            self._tracker.fail(job.job_id, f"Cannot read file: {exc}")
            return job

        # Check for existing document with the same hash
        existing = self._postgres.find_by_file_hash(file_hash)
        if existing is not None:
            logger.info(
                "File unchanged (hash=%s), skipping: %s",
                file_hash[:12],
                file_path,
            )
            job.status = IngestionStatus.PENDING
            job.error_message = "unchanged"
            return job

        # Content changed or new — run full pipeline
        self._tracker.start(job.job_id)
        try:
            chunks = self._run_pipeline(Path(file_path), job, file_hash, index_name)
            self._tracker.complete(job.job_id, chunks_indexed=len(chunks), total_chunks=len(chunks))
        except Exception as exc:
            logger.error("Incremental ingest failed for %s: %s", file_path, exc)
            self._tracker.fail(job.job_id, str(exc))

        return job

    def batch_incremental(
        self,
        file_paths: list[str],
        index_name: str,
    ) -> list[IngestionJob]:
        """Process multiple files incrementally."""
        return [self.ingest_incremental(fp, index_name) for fp in file_paths]

    def delete_document(self, document_id: str) -> bool:
        """Remove a document from OpenSearch, PostgreSQL, and S3.

        Returns True if at least one store reported a deletion.
        """
        deleted_any = False

        # 1. Fetch document metadata for S3 key
        doc = self._postgres.get_document(document_id)

        # 2. Delete chunks from OpenSearch
        try:
            chunks = self._postgres.get_chunks_by_document(document_id)
            if chunks:
                chunk_ids = [c.id for c in chunks]
                self._opensearch.delete_chunks(
                    _get_default_index_name(),
                    chunk_ids,
                )
                deleted_any = True
        except Exception as exc:
            logger.warning("OpenSearch chunk delete failed: %s", exc)

        # 3. Delete from PostgreSQL (cascades to chunks table)
        try:
            if self._postgres.delete_document(document_id):
                deleted_any = True
        except Exception as exc:
            logger.warning("PostgreSQL delete failed: %s", exc)

        # 4. Delete from S3
        if doc and doc.source_url:
            try:
                self._s3.delete_file(doc.source_url)
                deleted_any = True
            except Exception as exc:
                logger.warning("S3 delete failed: %s", exc)

        return deleted_any

    def get_status(self, job_id: str) -> IngestionJob | None:
        return self._tracker.get(job_id)

    # ---- internals ----------------------------------------------------

    def _run_pipeline(
        self,
        path: Path,
        job: IngestionJob,
        file_hash: str,
        index_name: str,
    ) -> list[Chunk]:
        """Execute the full parse→clean→chunk→embed→index pipeline.

        If a document already exists under the same path (different hash),
        its old chunks are removed from OpenSearch first.
        """
        doc_id = file_hash

        # Find any existing document with same path (to clean up old chunks)
        existing = self._postgres.find_by_file_hash(file_hash)
        if existing is None:
            # Try to find by source_url to remove stale chunks
            existing = self._find_by_source(str(path))

        if existing is not None:
            self._remove_old_chunks(existing.id, index_name)

        # Parse
        parser = get_parser(path)
        parsed = parser.parse(path)

        # Clean
        cleaner = _get_cleaner()
        format_type = parsed.metadata.get("format", "generic")
        cleaned_text, _stats = cleaner.clean(parsed.content, text_format=format_type)

        # S3 upload
        s3_key: str | None = None
        try:
            s3_key = self._s3.upload_file(path)
        except Exception as exc:
            logger.warning("S3 upload failed: %s", exc)

        # Chunk
        chunker = _get_chunker()
        chunks = chunker.chunk(
            text=cleaned_text,
            document_id=doc_id,
            metadata={"source": str(path), "format": format_type, **parsed.metadata},
        )

        if not chunks:
            self._register_empty_document(path, doc_id, file_hash, s3_key)
            return []

        # Embed
        self._embedder.embed_chunks(chunks)

        # Index in OpenSearch
        try:
            self._opensearch.index_chunks(index_name, chunks)
        except Exception as exc:
            logger.error("OpenSearch indexing failed: %s", exc)

        # Register in PostgreSQL
        self._postgres.insert_document(DocumentRecord(
            id=doc_id,
            filename=path.name,
            source_url=str(path),
            mime_type=parsed.metadata.get("mime_type"),
            file_hash=file_hash,
            size_bytes=path.stat().st_size if path.exists() else None,
            chunk_count=len(chunks),
        ))
        self._postgres.insert_chunks([
            ChunkRecord(
                id=c.id,
                document_id=doc_id,
                chunk_index=c.index,
                content=c.content,
                token_count=c.token_count,
                indexed=True,
            )
            for c in chunks
        ])

        return chunks

    def _find_by_source(self, source_url: str) -> DocumentRecord | None:
        """Find an existing document by source_url."""
        docs = self._postgres.find_documents_by_metadata(source_url=source_url)
        return docs[0] if docs else None

    def _remove_old_chunks(self, document_id: str, index_name: str) -> None:
        """Delete old chunks from OpenSearch for a document being re-ingested."""
        try:
            old_chunks = self._postgres.get_chunks_by_document(document_id)
            if old_chunks:
                self._opensearch.delete_chunks(index_name, [c.id for c in old_chunks])
                self._postgres.delete_chunks_by_document(document_id)
                logger.info("Removed %d old chunks for document %s", len(old_chunks), document_id)
        except Exception as exc:
            logger.warning("Failed to remove old chunks: %s", exc)

    def _register_empty_document(
        self,
        path: Path,
        doc_id: str,
        file_hash: str,
        s3_key: str | None,
    ) -> None:
        """Register a document that produced zero chunks."""
        self._postgres.insert_document(DocumentRecord(
            id=doc_id,
            filename=path.name,
            source_url=str(path),
            file_hash=file_hash,
            size_bytes=path.stat().st_size if path.exists() else None,
            chunk_count=0,
        ))


def _get_default_index_name() -> str:
    try:
        from rag_pipeline.config import get_settings
        return f"{get_settings().storage.opensearch_index_prefix}-chunks-v1"
    except Exception:
        return "rag-chunks-v1"


# ------------------------------------------------------------------ #
#  Reindex manager
# ------------------------------------------------------------------ #


class ReindexManager:
    """Zero-downtime reindex using OpenSearch alias swap."""

    def __init__(
        self,
        postgres: PostgresClient,
        opensearch: OpenSearchClient,
        embedder: EmbeddingGenerator,
    ) -> None:
        self._postgres = postgres
        self._opensearch = opensearch
        self._embedder = embedder
        self._tracker = IngestionStatusTracker()

    def reindex_all(self, index_name: str) -> list[IngestionJob]:
        """Reindex every document. Returns one job per document."""
        docs = self._postgres.list_documents(limit=10_000)
        return [self.reindex_document(doc.id, index_name) for doc in docs]

    def reindex_document(
        self,
        document_id: str,
        index_name: str,
    ) -> IngestionJob:
        """Reprocess a single document: re-chunk, re-embed, re-index.

        Uses alias swap for zero-downtime reindex:
        1. Create new index with ``_reindex`` suffix.
        2. Index all documents.
        3. Swap alias to new index.
        4. Delete old index.
        """
        job = self._tracker.create(document_id)
        self._tracker.start(job.job_id)

        try:
            doc = self._postgres.get_document(document_id)
            if doc is None:
                self._tracker.fail(job.job_id, f"Document not found: {document_id}")
                return job

            # Fetch existing chunks for content
            old_chunks = self._postgres.get_chunks_by_document(document_id)
            if not old_chunks:
                self._tracker.complete(job.job_id, chunks_indexed=0, total_chunks=0)
                return job

            new_index = f"{index_name}_reindex"

            # 1. Create new index
            self._opensearch.create_index(new_index)

            # 2. Rebuild chunks from stored content
            new_chunks = self._rebuild_chunks(old_chunks, document_id)

            # 3. Embed
            if new_chunks:
                self._embedder.embed_chunks(new_chunks)

            # 4. Index into new index
            if new_chunks:
                self._opensearch.index_chunks(new_index, new_chunks)

            # 5. Alias swap
            self._opensearch.alias_swap(
                alias=index_name,
                new_index=new_index,
                old_index=index_name if self._opensearch.index_exists(index_name) else None,
            )

            # 6. Delete old index (the one the alias pointed to before)
            if self._opensearch.index_exists(index_name) and index_name != new_index:
                self._opensearch.delete_index(index_name)

            # 7. Rename new index back to original name via alias
            self._opensearch.alias_swap(
                alias=index_name,
                new_index=new_index,
                old_index=None,
            )

            # 8. Update PostgreSQL chunk count
            self._postgres.update_chunk_count(document_id, len(new_chunks))

            self._tracker.complete(job.job_id, chunks_indexed=len(new_chunks), total_chunks=len(new_chunks))
        except Exception as exc:
            logger.error("Reindex failed for document %s: %s", document_id, exc)
            self._tracker.fail(job.job_id, str(exc))

        return job

    def _rebuild_chunks(
        self,
        old_chunks: list[ChunkRecord],
        document_id: str,
    ) -> list[Chunk]:
        """Reconstruct Chunk objects from stored content for re-embedding."""
        return [
            Chunk(
                id=c.id,
                document_id=document_id,
                content=c.content,
                index=c.chunk_index,
                token_count=c.token_count or 0,
            )
            for c in old_chunks
        ]

    def get_status(self, job_id: str) -> IngestionJob | None:
        return self._tracker.get(job_id)
