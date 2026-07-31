"""Pipeline management — health, metrics, status."""

from __future__ import annotations

import logging

import psycopg2
import redis
from fastapi import APIRouter
from opensearchpy import OpenSearch

from rag_pipeline.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.get("/status")
async def pipeline_status() -> dict:
    """Component health status — check all services."""
    status: dict[str, dict] = {}

    # OpenSearch
    try:
        settings = get_settings()
        client = OpenSearch(
            hosts=[{"host": settings.storage.opensearch_host, "port": settings.storage.opensearch_port}],
            http_compress=True,
            use_ssl=False,
        )
        info = client.info()
        status["opensearch"] = {"status": "healthy", "version": info.get("version", {}).get("number", "unknown")}
    except Exception as e:
        status["opensearch"] = {"status": "unhealthy", "error": str(e)[:100]}

    # PostgreSQL
    try:
        settings = get_settings()
        conn = psycopg2.connect(settings.storage.postgres_dsn)
        conn.close()
        status["postgresql"] = {"status": "healthy"}
    except Exception as e:
        status["postgresql"] = {"status": "unhealthy", "error": str(e)[:100]}

    # Redis
    try:
        settings = get_settings()
        r = redis.Redis(host=settings.storage.redis_host, port=settings.storage.redis_port)
        r.ping()
        status["redis"] = {"status": "healthy"}
    except Exception as e:
        status["redis"] = {"status": "unhealthy", "error": str(e)[:100]}

    overall = "healthy" if all(s.get("status") == "healthy" for s in status.values()) else "degraded"
    return {"overall": overall, "components": status}


@router.get("/metrics")
async def pipeline_metrics() -> dict:
    """Pipeline metrics — document count, chunk count, index stats."""
    metrics: dict[str, int] = {}

    try:
        settings = get_settings()
        client = OpenSearch(
            hosts=[{"host": settings.storage.opensearch_host, "port": settings.storage.opensearch_port}],
            http_compress=True,
            use_ssl=False,
        )
        indices = client.indices.get(index="rag*")
        total_docs = sum(
            client.indices.stats(index=idx)["indices"][idx]["primaries"]["docs"]["count"]
            for idx in indices
        )
        metrics["total_documents_indexed"] = total_docs
        metrics["index_count"] = len(indices)
    except Exception:
        metrics["total_documents_indexed"] = 0
        metrics["index_count"] = 0

    try:
        settings = get_settings()
        conn = psycopg2.connect(settings.storage.postgres_dsn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        metrics["documents_registered"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        metrics["chunks_registered"] = cur.fetchone()[0]
        conn.close()
    except Exception:
        metrics["documents_registered"] = 0
        metrics["chunks_registered"] = 0

    return metrics
