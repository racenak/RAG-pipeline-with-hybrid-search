"""Tests for OpenSearch client (mocked — no live OpenSearch needed)."""

from unittest.mock import MagicMock, patch

from rag_pipeline.data.chunking import Chunk
from rag_pipeline.storage.opensearch import OpenSearchClient


def _make_client() -> OpenSearchClient:
    """Create an OpenSearchClient with a mocked underlying client."""
    with patch("rag_pipeline.storage.opensearch.OpenSearch"):
        client = OpenSearchClient.__new__(OpenSearchClient)
        client._client = MagicMock()
        client._host = "localhost"
        client._port = 9200
        return client


def _sample_chunks(n: int = 3) -> list[Chunk]:
    """Create sample chunks with embeddings."""
    return [
        Chunk(
            id=f"chunk-{i}",
            document_id="doc-1",
            content=f"Content of chunk {i}",
            index=i,
            token_count=10,
            embedding=[0.1] * 1024,
        )
        for i in range(n)
    ]


class TestHealthCheck:
    def test_healthy(self):
        client = _make_client()
        client._client.cluster.health.return_value = {"status": "green"}
        assert client.health_check() is True

    def test_unhealthy(self):
        client = _make_client()
        client._client.cluster.health.return_value = {"status": "red"}
        assert client.health_check() is False

    def test_connection_error(self):
        client = _make_client()
        client._client.cluster.health.side_effect = ConnectionError("refused")
        assert client.health_check() is False


class TestIndexManagement:
    def test_create_index(self):
        client = _make_client()
        client.create_index("test-index", dimension=1024)
        client._client.indices.create.assert_called_once()
        call_kwargs = client._client.indices.create.call_args
        assert call_kwargs[1]["index"] == "test-index"
        mapping = call_kwargs[1]["body"]["mappings"]["properties"]
        assert "embedding" in mapping
        assert mapping["embedding"]["type"] == "knn_vector"
        assert mapping["embedding"]["dimension"] == 1024

    def test_create_index_already_exists(self):
        client = _make_client()
        from opensearchpy.exceptions import RequestError

        client._client.indices.create.side_effect = RequestError(
            400, "resource_already_exists_exception"
        )
        # Should not raise
        client.create_index("test-index")

    def test_delete_index(self):
        client = _make_client()
        client.delete_index("test-index")
        client._client.indices.delete.assert_called_once_with(index="test-index")

    def test_delete_index_not_found(self):
        client = _make_client()
        from opensearchpy.exceptions import NotFoundError

        client._client.indices.delete.side_effect = NotFoundError(404)
        # Should not raise
        client.delete_index("nonexistent")

    def test_index_exists(self):
        client = _make_client()
        client._client.indices.exists.return_value = True
        assert client.index_exists("test-index") is True

    def test_alias_swap(self):
        client = _make_client()
        client.alias_swap("my-alias", "index-v2", "index-v1")
        call_kwargs = client._client.indices.update_aliases.call_args
        actions = call_kwargs[1]["body"]["actions"]
        assert len(actions) == 2
        assert actions[0]["add"]["index"] == "index-v2"
        assert actions[1]["remove"]["index"] == "index-v1"

    def test_alias_swap_no_old(self):
        client = _make_client()
        client.alias_swap("my-alias", "index-v1")
        call_kwargs = client._client.indices.update_aliases.call_args
        actions = call_kwargs[1]["body"]["actions"]
        assert len(actions) == 1
        assert actions[0]["add"]["index"] == "index-v1"


class TestDocumentOperations:
    def test_index_chunks(self):
        client = _make_client()
        client._client.bulk.return_value = {"errors": False}
        chunks = _sample_chunks(3)
        count = client.index_chunks("test-index", chunks)
        assert count == 3
        client._client.bulk.assert_called_once()
        bulk_body = client._client.bulk.call_args[1]["body"]
        # 3 chunks x 2 lines each (action + doc)
        assert len(bulk_body) == 6

    def test_index_chunks_empty(self):
        client = _make_client()
        count = client.index_chunks("test-index", [])
        assert count == 0
        client._client.bulk.assert_not_called()

    def test_delete_chunks(self):
        client = _make_client()
        client._client.bulk.return_value = {"errors": False}
        client.delete_chunks("test-index", ["c1", "c2"])
        bulk_body = client._client.bulk.call_args[1]["body"]
        assert len(bulk_body) == 2  # delete is action-only, no doc lines

    def test_delete_chunks_empty(self):
        client = _make_client()
        client.delete_chunks("test-index", [])
        client._client.bulk.assert_not_called()


class TestSearch:
    def test_knn_search(self):
        client = _make_client()
        client._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "c1", "_score": 0.95, "_source": {"content": "hello"}},
                    {"_id": "c2", "_score": 0.80, "_source": {"content": "world"}},
                ]
            }
        }
        results = client.knn_search("idx", [0.1] * 1024, top_k=5)
        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[0]["score"] == 0.95

    def test_text_search(self):
        client = _make_client()
        client._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "c3", "_score": 1.2, "_source": {"content": "search result"}},
                ]
            }
        }
        results = client.text_search("idx", "search query", top_k=10)
        assert len(results) == 1
        assert results[0]["id"] == "c3"

    def test_hybrid_search(self):
        client = _make_client()
        client._client.search.return_value = {
            "hits": {"hits": [{"_id": "c1", "_score": 0.9, "_source": {}}]}
        }
        results = client.hybrid_search("idx", [0.1] * 1024, "query")
        assert "knn" in results
        assert "text" in results
        assert len(results["knn"]) == 1
        assert len(results["text"]) == 1

    def test_knn_search_with_filter(self):
        client = _make_client()
        client._client.search.return_value = {"hits": {"hits": []}}
        client.knn_search(
            "idx", [0.1] * 1024, top_k=5, query_filter={"term": {"document_id": "d1"}}
        )
        call_kwargs = client._client.search.call_args
        body = call_kwargs[1]["body"]
        assert "filter" in body["query"]["knn"]["embedding"]


class TestStats:
    def test_get_index_stats(self):
        client = _make_client()
        client._client.indices.stats.return_value = {
            "indices": {
                "test-index": {
                    "total": {
                        "docs": {"count": 42},
                        "store": {"size_in_bytes": 12345},
                    }
                }
            }
        }
        stats = client.get_index_stats("test-index")
        assert stats["doc_count"] == 42
        assert stats["store_size_bytes"] == 12345

    def test_get_index_stats_not_found(self):
        client = _make_client()
        from opensearchpy.exceptions import NotFoundError

        client._client.indices.stats.side_effect = NotFoundError(404)
        stats = client.get_index_stats("nonexistent")
        assert stats["doc_count"] == 0


class TestHelpers:
    def test_parse_hits(self):
        response = {
            "hits": {
                "hits": [
                    {"_id": "a", "_score": 1.0, "_source": {"content": "x"}},
                    {"_id": "b", "_score": 0.5, "_source": {"content": "y"}},
                ]
            }
        }
        results = OpenSearchClient._parse_hits(response)
        assert len(results) == 2
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 1.0
        assert results[0]["content"] == "x"

    def test_parse_hits_empty(self):
        results = OpenSearchClient._parse_hits({"hits": {"hits": []}})
        assert results == []
