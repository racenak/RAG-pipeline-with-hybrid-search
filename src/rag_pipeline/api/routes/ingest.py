"""Ingestion API routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from rag_pipeline.api.schemas.ingest import (
    IngestBatchResponse,
    IngestDirRequest,
    IngestResponse,
    IngestURLRequest,
)
from rag_pipeline.data.fetchers import MockURLFetcher
from rag_pipeline.data.ingestion import ingest_directory, ingest_file, ingest_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


def _get_storage():
    """Create S3Storage client if configured."""
    try:
        from rag_pipeline.config import get_settings
        from rag_pipeline.data.storage import S3Storage

        settings = get_settings()
        return S3Storage(
            endpoint_url=settings.s3.endpoint_url,
            access_key=settings.s3.access_key,
            secret_key=settings.s3.secret_key,
            bucket=settings.s3.bucket,
        )
    except Exception:
        logger.warning("S3 unavailable — files won't be stored")
        return None


@router.post("/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(..., description="File to ingest"),
) -> IngestResponse:
    """Ingest a single uploaded file."""
    suffix = Path(file.filename or "upload").suffix or ".txt"
    tmp_path = Path(f"/tmp/rag_upload_{hash(file.filename)}{suffix}")

    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        storage = _get_storage()
        try:
            result = ingest_file(tmp_path, storage=storage)
        except Exception as e:
            if storage:
                logger.warning("S3 failed (%s), retrying without storage", e)
                result = ingest_file(tmp_path, storage=None)
            else:
                raise

        return IngestResponse(
            document_id=result.document_id,
            source=result.source,
            source_type=result.source_type,
            success=result.success,
            error=result.error,
            seaweedfs_fid=result.seaweedfs_fid,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/url", response_model=IngestResponse)
async def ingest_url_endpoint(
    request: IngestURLRequest,
) -> IngestResponse:
    """Ingest a URL via Firecrawl."""
    storage = _get_storage()
    fetcher = MockURLFetcher()
    try:
        result = ingest_url(request.url, fetcher, storage=storage)
    except Exception as e:
        if storage:
            logger.warning("S3 failed (%s), retrying without storage", e)
            result = ingest_url(request.url, fetcher, storage=None)
        else:
            raise

    return IngestResponse(
        document_id=result.document_id,
        source=result.source,
        source_type=result.source_type,
        success=result.success,
        error=result.error,
        seaweedfs_fid=result.seaweedfs_fid,
    )


@router.post("/directory", response_model=IngestBatchResponse)
async def ingest_directory_endpoint(
    request: IngestDirRequest,
) -> IngestBatchResponse:
    """Ingest all supported files in a directory."""
    directory = Path(request.directory)
    if not directory.is_dir():
        return IngestBatchResponse(
            total=0,
            successful=0,
            failed=1,
            results=[
                IngestResponse(
                    document_id="",
                    source=str(directory),
                    source_type="directory",
                    success=False,
                    error=f"Directory not found: {directory}",
                )
            ],
        )

    storage = _get_storage()
    results = []
    for result in ingest_directory(
        directory,
        storage=storage,
        recursive=request.recursive,
    ):
        results.append(
            IngestResponse(
                document_id=result.document_id,
                source=result.source,
                source_type=result.source_type,
                success=result.success,
                error=result.error,
                seaweedfs_fid=result.seaweedfs_fid,
            )
        )

    successful = sum(1 for r in results if r.success)
    return IngestBatchResponse(
        total=len(results),
        successful=successful,
        failed=len(results) - successful,
        results=results,
    )
