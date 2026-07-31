"""Tests for API health and readiness endpoints."""

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "rag-pipeline"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-pipeline"


def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
