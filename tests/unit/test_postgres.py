"""Tests for PostgreSQL document/chunk registry (mocked)."""

from unittest.mock import MagicMock

from rag_pipeline.storage.postgres import (
    ChunkRecord,
    DocumentRecord,
    PostgresClient,
)


def _make_client() -> tuple[PostgresClient, MagicMock]:
    """Create a PostgresClient with a pre-set mocked connection.

    We bypass _get_conn entirely by setting _conn directly, so
    no live PostgreSQL is needed.
    """
    client = PostgresClient.__new__(PostgresClient)
    client._dsn = "postgresql://rag:pass@localhost/test"
    client._pool_size = 5
    mock_conn = MagicMock()
    mock_conn.closed = False
    client._conn = mock_conn
    return client, mock_conn


class TestDocumentRecord:
    def test_fields(self):
        doc = DocumentRecord(id="d1", filename="test.pdf")
        assert doc.id == "d1"
        assert doc.filename == "test.pdf"
        assert doc.source_url is None
        assert doc.chunk_count == 0

    def test_optional_fields(self):
        doc = DocumentRecord(
            id="d1",
            filename="test.pdf",
            source_url="http://example.com",
            mime_type="application/pdf",
            file_hash="abc123",
            size_bytes=1024,
            chunk_count=5,
        )
        assert doc.source_url == "http://example.com"
        assert doc.chunk_count == 5


class TestChunkRecord:
    def test_fields(self):
        chunk = ChunkRecord(id="c1", document_id="d1", chunk_index=0, content="hello")
        assert chunk.id == "c1"
        assert chunk.document_id == "d1"
        assert chunk.indexed is False


class TestPostgresClient:
    def test_health_check_ok(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        assert client.health_check() is True

    def test_health_check_failure(self):
        client, mock_conn = _make_client()
        mock_conn.cursor.side_effect = ConnectionError("refused")
        assert client.health_check() is False

    def test_insert_document(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        doc = DocumentRecord(id="d1", filename="test.pdf")
        client.insert_document(doc)
        mock_cursor.execute.assert_called_once()

    def test_get_document_not_found(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        result = client.get_document("nonexistent")
        assert result is None

    def test_list_documents(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": "d1", "filename": "a.pdf", "source_url": None, "mime_type": None,
             "file_hash": None, "size_bytes": None, "chunk_count": 3, "created_at": None}
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        docs = client.list_documents()
        assert len(docs) == 1
        assert docs[0].id == "d1"

    def test_count_documents(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        assert client.count_documents() == 42

    def test_insert_chunks_empty(self):
        client, _ = _make_client()
        client.insert_chunks([])
        # Empty list — no cursor call
        client._conn.cursor.assert_not_called()

    def test_count_chunks(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (10,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        assert client.count_chunks() == 10

    def test_delete_document(self):
        client, mock_conn = _make_client()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        assert client.delete_document("d1") is True

    def test_close(self):
        client, mock_conn = _make_client()
        mock_conn.closed = False
        client.close()
        mock_conn.close.assert_called_once()
