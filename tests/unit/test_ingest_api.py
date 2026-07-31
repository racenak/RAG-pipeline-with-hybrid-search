"""Tests for ingestion API endpoints."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_pipeline.api.app import app

client = TestClient(app)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _patch_storage():
    return patch("rag_pipeline.api.routes.ingest._get_storage", return_value=None)


def test_ingest_file_upload():
    with _patch_storage():
        pdf_path = FIXTURES / "sample.pdf"
        with pdf_path.open("rb") as f:
            response = client.post(
                "/api/v1/ingest/file",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document_id"]
    assert data["source_type"] == "file"


def test_ingest_file_upload_txt():
    with _patch_storage():
        txt_path = FIXTURES / "sample.txt"
        with txt_path.open("rb") as f:
            response = client.post(
                "/api/v1/ingest/file",
                files={"file": ("sample.txt", f, "text/plain")},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_ingest_url():
    with _patch_storage():
        response = client.post(
            "/api/v1/ingest/url",
            json={"url": "https://example.com/docs"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["source_type"] == "url"


def test_ingest_directory():
    with _patch_storage():
        response = client.post(
            "/api/v1/ingest/directory",
            json={"directory": str(FIXTURES), "recursive": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 6
    assert data["successful"] >= 6
    assert data["failed"] == 0


def test_ingest_directory_not_found():
    response = client.post(
        "/api/v1/ingest/directory",
        json={"directory": "/nonexistent/path", "recursive": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["failed"] == 1
