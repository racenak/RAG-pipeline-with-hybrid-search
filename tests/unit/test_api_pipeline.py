"""Tests for pipeline management endpoints and API key auth."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_pipeline.api.auth import validate_api_key
from rag_pipeline.api.routes.pipeline import router as pipeline_router


@pytest.fixture
def pipeline_app():
    """Create a minimal app with only the pipeline router to avoid unrelated import errors."""
    app = FastAPI()
    app.include_router(pipeline_router)
    return app


@pytest.fixture
def client(pipeline_app):
    return TestClient(pipeline_app, raise_server_exceptions=False)


def _make_settings():
    settings = MagicMock()
    settings.storage.opensearch_host = "localhost"
    settings.storage.opensearch_port = 9200
    settings.storage.postgres_dsn = "postgresql://localhost/test"
    settings.storage.redis_host = "localhost"
    settings.storage.redis_port = 6379
    return settings


# ---- pipeline_status ---------------------------------------------------------


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_status_all_healthy(mock_settings, client):
    mock_settings.return_value = _make_settings()

    mock_os_client = MagicMock()
    mock_os_client.info.return_value = {"version": {"number": "2.11.0"}}
    mock_pg_conn = MagicMock()
    mock_redis_instance = MagicMock()

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", return_value=mock_os_client),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
        patch("rag_pipeline.api.routes.pipeline.redis") as mock_redis_mod,
    ):
        mock_pg.connect.return_value = mock_pg_conn
        mock_redis_mod.Redis.return_value = mock_redis_instance

        r = client.get("/api/v1/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert data["overall"] == "healthy"
        assert data["components"]["opensearch"]["status"] == "healthy"
        assert data["components"]["postgresql"]["status"] == "healthy"
        assert data["components"]["redis"]["status"] == "healthy"


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_status_degraded(mock_settings, client):
    mock_settings.return_value = _make_settings()

    mock_os_client = MagicMock()
    mock_os_client.info.return_value = {"version": {"number": "2.11.0"}}

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", return_value=mock_os_client),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
        patch("rag_pipeline.api.routes.pipeline.redis") as mock_redis_mod,
    ):
        mock_pg.connect.side_effect = Exception("connection refused")
        mock_redis_instance = MagicMock()
        mock_redis_mod.Redis.return_value = mock_redis_instance

        r = client.get("/api/v1/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert data["overall"] == "degraded"
        assert data["components"]["opensearch"]["status"] == "healthy"
        assert data["components"]["postgresql"]["status"] == "unhealthy"


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_status_all_unhealthy(mock_settings, client):
    mock_settings.return_value = _make_settings()

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", side_effect=Exception("os down")),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
        patch("rag_pipeline.api.routes.pipeline.redis") as mock_redis_mod,
    ):
        mock_pg.connect.side_effect = Exception("pg down")
        mock_redis_inst = MagicMock()
        mock_redis_inst.ping.side_effect = Exception("redis down")
        mock_redis_mod.Redis.return_value = mock_redis_inst

        r = client.get("/api/v1/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert data["overall"] == "degraded"
        assert data["components"]["opensearch"]["status"] == "unhealthy"
        assert data["components"]["postgresql"]["status"] == "unhealthy"
        assert data["components"]["redis"]["status"] == "unhealthy"


# ---- pipeline_metrics --------------------------------------------------------


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_metrics_returns_counts(mock_settings, client):
    mock_settings.return_value = _make_settings()

    mock_os_client = MagicMock()
    mock_os_client.indices.get.return_value = {"rag-chunks-v1": {}, "rag-docs-v1": {}}
    mock_os_client.indices.stats.side_effect = lambda index: {
        "indices": {index: {"primaries": {"docs": {"count": 100}}}}
    }

    mock_cur = MagicMock()
    mock_cur.fetchone.side_effect = [(50,), (200,)]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", return_value=mock_os_client),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
    ):
        mock_pg.connect.return_value = mock_conn

        r = client.get("/api/v1/pipeline/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["total_documents_indexed"] == 200
        assert data["index_count"] == 2
        assert data["documents_registered"] == 50
        assert data["chunks_registered"] == 200


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_metrics_returns_zeros_on_error(mock_settings, client):
    mock_settings.return_value = _make_settings()

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", side_effect=Exception("conn refused")),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
    ):
        mock_pg.connect.side_effect = Exception("pg down")

        r = client.get("/api/v1/pipeline/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["total_documents_indexed"] == 0
        assert data["index_count"] == 0
        assert data["documents_registered"] == 0
        assert data["chunks_registered"] == 0


@patch("rag_pipeline.api.routes.pipeline.get_settings")
def test_pipeline_metrics_opensearch_only(mock_settings, client):
    mock_settings.return_value = _make_settings()

    mock_os_client = MagicMock()
    mock_os_client.indices.get.return_value = {"rag-chunks-v1": {}}
    mock_os_client.indices.stats.side_effect = lambda index: {
        "indices": {index: {"primaries": {"docs": {"count": 42}}}}
    }

    with (
        patch("rag_pipeline.api.routes.pipeline.OpenSearch", return_value=mock_os_client),
        patch("rag_pipeline.api.routes.pipeline.psycopg2") as mock_pg,
    ):
        mock_pg.connect.side_effect = Exception("pg down")

        r = client.get("/api/v1/pipeline/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["total_documents_indexed"] == 42
        assert data["index_count"] == 1
        assert data["documents_registered"] == 0
        assert data["chunks_registered"] == 0


# ---- validate_api_key --------------------------------------------------------


@pytest.mark.asyncio
@patch("rag_pipeline.api.auth.get_settings")
async def test_validate_api_key_auth_disabled(mock_settings):
    settings = MagicMock()
    settings.api.auth_enabled = False
    mock_settings.return_value = settings

    result = await validate_api_key(api_key=None)
    assert result is None


@pytest.mark.asyncio
@patch("rag_pipeline.api.auth.get_settings")
async def test_validate_api_key_valid(mock_settings):
    settings = MagicMock()
    settings.api.auth_enabled = True
    mock_settings.return_value = settings

    result = await validate_api_key(api_key="dev-key-12345")
    assert result == "dev-key-12345"


@pytest.mark.asyncio
@patch("rag_pipeline.api.auth.get_settings")
async def test_validate_api_key_invalid_key(mock_settings):
    settings = MagicMock()
    settings.api.auth_enabled = True
    mock_settings.return_value = settings

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await validate_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("rag_pipeline.api.auth.get_settings")
async def test_validate_api_key_missing_key(mock_settings):
    settings = MagicMock()
    settings.api.auth_enabled = True
    mock_settings.return_value = settings

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await validate_api_key(api_key=None)
    assert exc_info.value.status_code == 401
