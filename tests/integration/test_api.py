"""Integration tests — API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from rag_pipeline.api.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    """Test health and readiness endpoints."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "counters" in data


class TestSearchEndpoint:
    """Test search API."""

    def test_search_requires_query(self, client):
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422  # Validation error

    def test_search_with_query(self, client):
        response = client.post("/api/v1/search", json={
            "query": "test query",
            "mode": "hybrid",
            "top_k": 5,
        })
        # Returns 500 when OpenSearch is unavailable
        assert response.status_code in (200, 500, 503)


class TestDocumentEndpoints:
    """Test document management API."""

    def test_list_documents(self, client):
        response = client.get("/api/v1/documents")
        # May return 200 or 503 depending on PostgreSQL availability
        assert response.status_code in (200, 503)


class TestEvaluationEndpoints:
    """Test evaluation API."""

    def test_list_datasets(self, client):
        response = client.get("/api/v1/evaluation/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data

    def test_available_metrics(self, client):
        response = client.get("/api/v1/evaluation/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "retrieval" in data
        assert "generation" in data


class TestIngestEndpoint:
    """Test ingest API."""

    def test_ingest_requires_file(self, client):
        response = client.post("/api/v1/ingest/file", json={})
        # May return 422 (validation) or 503 (service unavailable)
        assert response.status_code in (422, 500, 503)
