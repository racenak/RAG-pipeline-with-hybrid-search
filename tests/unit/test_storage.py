"""Tests for S3 storage client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_pipeline.data.storage import S3Storage

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestS3Storage:
    """Unit tests for S3Storage with mocked boto3."""

    def _make_client(self) -> S3Storage:
        with patch("rag_pipeline.data.storage.boto3"):
            return S3Storage(endpoint_url="http://localhost:8333")

    def test_upload_file(self):
        client = self._make_client()
        client._client = MagicMock()

        key = client.upload_file(FIXTURES / "sample.txt")

        assert key.startswith("files/")
        client._client.upload_file.assert_called_once()

    def test_upload_bytes(self):
        client = self._make_client()
        client._client = MagicMock()

        key = client.upload_bytes(b"hello", "test/doc.md")

        assert key == "test/doc.md"
        client._client.put_object.assert_called_once()

    def test_download_file(self):
        client = self._make_client()
        mock_body = MagicMock()
        mock_body.read.return_value = b"content"
        client._client = MagicMock()
        client._client.get_object.return_value = {"Body": mock_body}

        content = client.download_file("test/doc.md")

        assert content == b"content"

    def test_delete_file(self):
        client = self._make_client()
        client._client = MagicMock()

        client.delete_file("test/doc.md")

        client._client.delete_object.assert_called_once()

    def test_ensure_bucket_creates_if_missing(self):
        client = self._make_client()
        client._client = MagicMock()
        client._client.head_bucket.side_effect = Exception("not found")

        client._ensure_bucket()

        client._client.create_bucket.assert_called_once()

    def test_close(self):
        client = self._make_client()
        client.close()  # Should not raise
