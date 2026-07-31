"""Document management API routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from rag_pipeline.api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
)
from rag_pipeline.api.schemas.documents_v2 import (
    BatchIngestRequest,
    BatchIngestResponse,
    IncrementalIngestRequest,
    IngestionJobResponse,
    ReindexRequest,
    ReindexResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _get_postgres_client():
    """Create a PostgresClient from config."""
    from rag_pipeline.config import get_settings
    from rag_pipeline.storage.postgres import PostgresClient

    settings = get_settings()
    s = settings.storage
    return PostgresClient(
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_database,
        user=s.postgres_user,
        password=s.postgres_password,
    )


def _build_clients():
    """Create storage clients from config."""
    from rag_pipeline.config import get_settings
    from rag_pipeline.data.storage import S3Storage
    from rag_pipeline.embeddings.generator import EmbeddingGenerator
    from rag_pipeline.storage.opensearch import OpenSearchClient
    from rag_pipeline.storage.postgres import PostgresClient

    settings = get_settings()
    s = settings.storage

    postgres = PostgresClient(
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_database,
        user=s.postgres_user,
        password=s.postgres_password,
    )
    opensearch = OpenSearchClient(
        host=s.opensearch_host,
        port=s.opensearch_port,
        scheme=s.opensearch_scheme,
        username=s.opensearch_username,
        password=s.opensearch_password,
        timeout=s.opensearch_timeout,
    )
    embedder = EmbeddingGenerator(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        normalize=settings.embedding.normalize,
        batch_size=settings.embedding.batch_size,
        cache_enabled=settings.embedding.cache_enabled,
        cache_dir=settings.embedding.cache_dir,
    )
    s3 = S3Storage(
        endpoint_url=settings.s3.endpoint_url,
        access_key=settings.s3.access_key,
        secret_key=settings.s3.secret_key,
        bucket=settings.s3.bucket,
    )
    return postgres, opensearch, embedder, s3


def _job_to_response(job) -> IngestionJobResponse:
    return IngestionJobResponse(
        job_id=job.job_id,
        document_path=job.document_path,
        status=job.status.value,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
        chunks_indexed=job.chunks_indexed,
        total_chunks=job.total_chunks,
    )


# ------------------------------------------------------------------ #
#  CRUD routes (v1)
# ------------------------------------------------------------------ #


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> DocumentListResponse:
    """List all ingested documents with metadata."""
    try:
        pg = _get_postgres_client()
        total = pg.count_documents()
        docs = pg.list_documents(limit=limit, offset=offset)
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=503, detail=f"PostgreSQL unavailable: {e}") from None

    return DocumentListResponse(
        total=total,
        count=len(docs),
        offset=offset,
        limit=limit,
        documents=[
            DocumentItem(
                id=d.id,
                filename=d.filename,
                source_url=d.source_url,
                mime_type=d.mime_type,
                file_hash=d.file_hash,
                size_bytes=d.size_bytes,
                chunk_count=d.chunk_count,
                created_at=d.created_at.isoformat() if d.created_at else None,
            )
            for d in docs
        ],
    )


@router.get("/{document_id}", response_model=DocumentItem)
async def get_document(document_id: str) -> DocumentItem:
    """Get a single document by ID."""
    try:
        pg = _get_postgres_client()
        doc = pg.get_document(document_id)
    except Exception as e:
        logger.error("Failed to get document: %s", e)
        raise HTTPException(status_code=503, detail=f"PostgreSQL unavailable: {e}") from None

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    return DocumentItem(
        id=doc.id,
        filename=doc.filename,
        source_url=doc.source_url,
        mime_type=doc.mime_type,
        file_hash=doc.file_hash,
        size_bytes=doc.size_bytes,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Delete a document and its chunks."""
    try:
        pg = _get_postgres_client()
        deleted = pg.delete_document(document_id)
    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        raise HTTPException(status_code=503, detail=f"PostgreSQL unavailable: {e}") from None

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    return DocumentDeleteResponse(
        deleted=True,
        document_id=document_id,
        message=f"Document {document_id} and its chunks deleted",
    )


# ------------------------------------------------------------------ #
#  Incremental ingestion routes (v2)
# ------------------------------------------------------------------ #


@router.post("/incremental", response_model=IngestionJobResponse)
async def incremental_ingest(req: IncrementalIngestRequest) -> IngestionJobResponse:
    """Incremental ingest a single file (skip if unchanged)."""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    try:
        postgres, opensearch, embedder, s3 = _build_clients()
        from rag_pipeline.data.incremental import IncrementalIngestor

        ingestor = IncrementalIngestor(postgres, opensearch, embedder, s3)
        job = ingestor.ingest_incremental(req.file_path, req.index_name)
        return _job_to_response(job)
    except Exception as exc:
        logger.error("Incremental ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/batch", response_model=BatchIngestResponse)
async def batch_incremental_ingest(req: BatchIngestRequest) -> BatchIngestResponse:
    """Batch incremental ingest multiple files."""
    missing = [fp for fp in req.file_paths if not Path(fp).exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Files not found: {missing}")

    try:
        postgres, opensearch, embedder, s3 = _build_clients()
        from rag_pipeline.data.incremental import IncrementalIngestor

        ingestor = IncrementalIngestor(postgres, opensearch, embedder, s3)
        jobs = ingestor.batch_incremental(req.file_paths, req.index_name)
        return BatchIngestResponse(
            jobs=[_job_to_response(j) for j in jobs],
            total=len(jobs),
        )
    except Exception as exc:
        logger.error("Batch ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_status(job_id: str) -> IngestionJobResponse:
    """Check the status of an ingestion job."""
    raise HTTPException(
        status_code=404,
        detail=f"Job {job_id} not found (in-memory tracker resets on restart)",
    )


@router.post("/reindex", response_model=ReindexResponse)
async def trigger_reindex(req: ReindexRequest) -> ReindexResponse:
    """Trigger a full or single-document reindex."""
    try:
        postgres, opensearch, embedder, _s3 = _build_clients()
        from rag_pipeline.data.incremental import ReindexManager

        manager = ReindexManager(postgres, opensearch, embedder)
        jobs = manager.reindex_all(req.index_name)
        return ReindexResponse(
            jobs=[_job_to_response(j) for j in jobs],
            total=len(jobs),
        )
    except Exception as exc:
        logger.error("Reindex failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
