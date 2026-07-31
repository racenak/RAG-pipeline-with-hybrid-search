"""Tests for ingestion orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

from rag_pipeline.data.fetchers import MockURLFetcher
from rag_pipeline.data.ingestion import (
    discover_files,
    ingest_directory,
    ingest_file,
    ingest_url,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_ingest_file_txt():
    result = ingest_file(FIXTURES / "sample.txt")
    assert result.success
    assert result.document_id
    assert result.source_type == "file"


def test_ingest_file_pdf():
    result = ingest_file(FIXTURES / "sample.pdf")
    assert result.success


def test_ingest_file_docx():
    result = ingest_file(FIXTURES / "sample.docx")
    assert result.success


def test_ingest_file_md():
    result = ingest_file(FIXTURES / "sample.md")
    assert result.success


def test_ingest_file_html():
    result = ingest_file(FIXTURES / "sample.html")
    assert result.success


def test_ingest_file_csv():
    result = ingest_file(FIXTURES / "sample.csv")
    assert result.success


def test_ingest_file_not_found():
    result = ingest_file(Path("/nonexistent/file.txt"))
    assert not result.success
    assert result.error


def test_ingest_file_with_storage():
    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "files/sample.txt"

    result = ingest_file(FIXTURES / "sample.txt", storage=mock_storage)

    assert result.success
    assert result.seaweedfs_fid == "files/sample.txt"
    mock_storage.upload_file.assert_called_once_with(FIXTURES / "sample.txt")


def test_ingest_file_without_storage():
    result = ingest_file(FIXTURES / "sample.txt")
    assert result.success
    assert result.seaweedfs_fid is None


def test_ingest_url_mock():
    fetcher = MockURLFetcher()
    result = ingest_url("https://example.com/docs", fetcher)
    assert result.success
    assert result.source_type == "url"
    assert result.document_id


def test_ingest_url_with_storage():
    fetcher = MockURLFetcher()
    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "urls/abc123.md"

    result = ingest_url("https://example.com/docs", fetcher, storage=mock_storage)

    assert result.success
    assert result.seaweedfs_fid == "urls/abc123.md"
    mock_storage.upload_bytes.assert_called_once()


def test_ingest_url_without_storage():
    fetcher = MockURLFetcher()
    result = ingest_url("https://example.com/docs", fetcher)
    assert result.success
    assert result.seaweedfs_fid is None


def test_discover_files():
    files = list(discover_files(FIXTURES))
    extensions = {f.suffix for f in files}
    assert ".txt" in extensions
    assert ".pdf" in extensions
    assert ".docx" in extensions
    assert ".md" in extensions
    assert ".html" in extensions
    assert ".csv" in extensions


def test_ingest_directory():
    results = list(ingest_directory(FIXTURES))
    assert len(results) >= 6
    assert all(r.success for r in results)
