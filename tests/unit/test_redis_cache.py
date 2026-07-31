"""Tests for Redis embedding and query cache (mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np

from rag_pipeline.storage.redis_cache import QueryCache, RedisEmbeddingCache


def _make_embedding_cache() -> RedisEmbeddingCache:
    with patch("rag_pipeline.storage.redis_cache.redis"):
        cache = RedisEmbeddingCache.__new__(RedisEmbeddingCache)
        cache._client = MagicMock()
        cache._ttl = 86400
        return cache


def _make_query_cache() -> QueryCache:
    with patch("rag_pipeline.storage.redis_cache.redis"):
        cache = QueryCache.__new__(QueryCache)
        cache._client = MagicMock()
        cache._ttl = 300
        return cache


class TestRedisEmbeddingCache:
    def test_set_and_get(self):
        cache = _make_embedding_cache()
        vector = [0.1, 0.2, 0.3]
        blob = np.array(vector, dtype=np.float32).tobytes()
        cache._client.get.return_value = blob

        result = cache.get("hello")
        assert result is not None
        assert len(result) == 3
        assert abs(result[0] - 0.1) < 1e-6

    def test_get_miss(self):
        cache = _make_embedding_cache()
        cache._client.get.return_value = None
        assert cache.get("nonexistent") is None

    def test_has(self):
        cache = _make_embedding_cache()
        cache._client.exists.return_value = 1
        assert cache.has("hello") is True
        cache._client.exists.return_value = 0
        assert cache.has("hello") is False

    def test_set_many(self):
        cache = _make_embedding_cache()
        cache._client.pipeline.return_value = MagicMock()
        cache.set_many(["a", "b"], [[0.1], [0.2]])
        cache._client.pipeline.return_value.execute.assert_called_once()

    def test_get_many(self):
        cache = _make_embedding_cache()
        v1 = np.array([0.1], dtype=np.float32).tobytes()
        v2 = np.array([0.2], dtype=np.float32).tobytes()
        cache._client.mget.return_value = [v1, v2, None]

        results = cache.get_many(["a", "b", "c"])
        assert 0 in results
        assert 1 in results
        assert 2 not in results
        assert abs(results[0][0] - 0.1) < 1e-6


class TestQueryCache:
    def test_set_and_get(self):
        import json

        cache = _make_query_cache()
        results = [{"id": "c1", "score": 0.9}]
        cache._client.get.return_value = json.dumps(results)

        cached = cache.get("test query", top_k=5)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["id"] == "c1"

    def test_get_miss(self):
        cache = _make_query_cache()
        cache._client.get.return_value = None
        assert cache.get("missing") is None

    def test_clear(self):
        cache = _make_query_cache()
        cache._client.scan.return_value = (0, ["q1", "q2"])
        cache._client.delete.return_value = 2
        count = cache.clear()
        assert count == 2
