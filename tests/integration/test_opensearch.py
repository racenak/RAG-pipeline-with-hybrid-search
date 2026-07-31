"""Integration tests — OpenSearch operations."""

from unittest.mock import MagicMock, patch


class TestOpenSearchOperations:
    """Test OpenSearch client operations with mocks."""

    @patch("rag_pipeline.storage.opensearch.OpenSearch")
    def test_create_index(self, mock_os_cls):
        """Test index creation."""
        from rag_pipeline.storage.opensearch import OpenSearchClient

        mock_client = MagicMock()
        mock_os_cls.return_value = mock_client
        mock_client.indices.create.return_value = {"acknowledged": True}

        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = mock_client
        client._host = "localhost"
        client._port = 9200

        # Should not crash
        client.create_index("test_index")
        mock_client.indices.create.assert_called_once()

    @patch("rag_pipeline.storage.opensearch.OpenSearch")
    def test_index_exists(self, mock_os_cls):
        """Test checking if an index exists."""
        from rag_pipeline.storage.opensearch import OpenSearchClient

        mock_client = MagicMock()
        mock_os_cls.return_value = mock_client
        mock_client.indices.exists.return_value = True

        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = mock_client
        client._host = "localhost"
        client._port = 9200

        assert client.index_exists("test_index") is True

    @patch("rag_pipeline.storage.opensearch.OpenSearch")
    def test_index_chunks(self, mock_os_cls, sample_chunks):
        """Test bulk chunk indexing."""
        from rag_pipeline.storage.opensearch import OpenSearchClient

        mock_client = MagicMock()
        mock_os_cls.return_value = mock_client
        mock_client.bulk.return_value = {"errors": False, "items": []}

        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = mock_client
        client._host = "localhost"
        client._port = 9200

        count = client.index_chunks("test_index", sample_chunks)
        assert count == len(sample_chunks)
        mock_client.bulk.assert_called_once()

    @patch("rag_pipeline.storage.opensearch.OpenSearch")
    def test_health_check(self, mock_os_cls):
        """Test health check returns status."""
        from rag_pipeline.storage.opensearch import OpenSearchClient

        mock_client = MagicMock()
        mock_os_cls.return_value = mock_client
        mock_client.cluster.health.return_value = {"status": "green"}

        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = mock_client
        client._host = "localhost"
        client._port = 9200

        assert client.health_check() is True

    @patch("rag_pipeline.storage.opensearch.OpenSearch")
    def test_delete_index(self, mock_os_cls):
        """Test index deletion."""
        from rag_pipeline.storage.opensearch import OpenSearchClient

        mock_client = MagicMock()
        mock_os_cls.return_value = mock_client

        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = mock_client
        client._host = "localhost"
        client._port = 9200

        client.delete_index("test_index")
        mock_client.indices.delete.assert_called_once()
