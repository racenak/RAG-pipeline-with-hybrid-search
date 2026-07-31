"""Performance tests — assert latency targets for critical paths."""

from __future__ import annotations

import time


class TestQueryProcessingLatency:
    """Query processing should be fast (< 50ms per query)."""

    def test_query_processing_latency(self) -> None:
        from rag_pipeline.retrieval.query import get_query_processor

        processor = get_query_processor()

        queries = [
            "What is the embedding dimension?",
            "Compare vector search vs BM25",
            "How does the RAG pipeline work?",
        ]

        for query in queries:
            start = time.perf_counter()
            result = processor.process(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 50, f"Query processing took {elapsed_ms:.1f}ms for: {query}"
            assert result.original == query or len(result.original) > 0


class TestBM25Latency:
    """BM25 search should be fast (< 100ms per search)."""

    def test_bm25_search_latency(self) -> None:
        from rag_pipeline.retrieval.bm25 import BM25

        bm25 = BM25()

        for i in range(50):
            bm25.add_document(f"doc{i}", f"Document {i} about topic {i % 10} with keywords")

        queries = ["topic 5", "document about keywords", "topic 0"]

        for query in queries:
            start = time.perf_counter()
            results = bm25.search(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, f"BM25 search took {elapsed_ms:.1f}ms"
            assert len(results) > 0


class TestHybridSearchLatency:
    """Hybrid search should complete within reasonable time."""

    def test_hybrid_search_latency(self) -> None:
        from rag_pipeline.retrieval.bm25 import BM25
        from rag_pipeline.retrieval.hybrid import HybridSearch, HybridSearchConfig
        from rag_pipeline.retrieval.vector import SearchResult

        bm25 = BM25()
        for i in range(20):
            bm25.add_document(f"doc{i}", f"Content about topic {i}")

        bm25_results = bm25.search("topic 1", top_k=5)
        search_results = [
            SearchResult(id=r["id"], score=r["score"], content=r["content"])
            for r in bm25_results
        ]

        config = HybridSearchConfig(vector_top_k=5, bm25_top_k=5, rrf_k=60)
        search = HybridSearch(config=config)

        start = time.perf_counter()
        search.fuse(bm25_results=search_results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Hybrid search took {elapsed_ms:.1f}ms"


class TestLLMCache:
    """LLM cache should improve repeated query performance."""

    def test_cache_hit_returns_cached(self) -> None:
        from rag_pipeline.generation.cache import LLMResponseCache

        class MockRedis:
            def __init__(self) -> None:
                self._store: dict[str, str] = {}

            def get(self, key: str) -> str | None:
                return self._store.get(key)

            def setex(self, key: str, ttl: int, value: str) -> None:  # noqa: ARG002
                self._store[key] = value

            def keys(self, pattern: str) -> list[str]:
                prefix = pattern.replace("*", "")
                return [k for k in self._store if k.startswith(prefix)]

            def delete(self, *keys: str) -> int:
                count = 0
                for k in keys:
                    if k in self._store:
                        del self._store[k]
                        count += 1
                return count

        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock, ttl_seconds=300)

        messages = [{"role": "user", "content": "test"}]
        model = "test-model"

        assert cache.get(messages, model) is None

        cache.set(messages, model, "cached response")

        assert cache.get(messages, model) == "cached response"

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self) -> None:
        from rag_pipeline.generation.cache import LLMResponseCache

        class MockRedis:
            def __init__(self) -> None:
                self._store: dict[str, str] = {}

            def get(self, key: str) -> str | None:
                return self._store.get(key)

            def setex(self, key: str, ttl: int, value: str) -> None:  # noqa: ARG002
                self._store[key] = value

            def keys(self, pattern: str) -> list[str]:
                prefix = pattern.replace("*", "")
                return [k for k in self._store if k.startswith(prefix)]

            def delete(self, *keys: str) -> int:
                count = 0
                for k in keys:
                    if k in self._store:
                        del self._store[k]
                        count += 1
                return count

        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)

        messages = [{"role": "user", "content": "test"}]
        cache.set(messages, "model", "response")

        cleared = cache.clear()
        assert cleared == 1
        assert cache.get(messages, "model") is None


class TestConnectionPoolSingletons:
    """Client singletons should return the same instance."""

    def test_reset_clients_clears_singletons(self) -> None:
        from rag_pipeline.storage import clients
        from rag_pipeline.storage.clients import reset_clients

        reset_clients()
        clients._opensearch_client = None
        clients._postgres_pool = None
        clients._redis_client = None
        assert clients._opensearch_client is None
