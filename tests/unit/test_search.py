"""Tests for unified search engine."""


from rag_pipeline.retrieval.bm25 import BM25
from rag_pipeline.retrieval.hybrid import HybridSearchConfig
from rag_pipeline.retrieval.search import SearchEngine
from rag_pipeline.retrieval.vector import FAISSVectorSearch


class TestSearchEngine:
    def _make_engine(self) -> SearchEngine:
        # FAISS backend
        faiss = FAISSVectorSearch(dimension=4)
        faiss.index_vectors(
            ["d1", "d2", "d3"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ["hello world", "foo bar", "baz qux"],
            metadatas=[{"type": "a"}, {"type": "b"}, {"type": "a"}],
        )

        # BM25 backend
        bm25 = BM25()
        bm25.add_document("d1", "hello world", metadata={"type": "a"})
        bm25.add_document("d2", "foo bar", metadata={"type": "b"})
        bm25.add_document("d3", "baz qux", metadata={"type": "a"})

        config = HybridSearchConfig(vector_top_k=3, bm25_top_k=3)
        return SearchEngine(vector_search=faiss, bm25_search=bm25, hybrid_config=config)

    def test_vector_search(self):
        engine = self._make_engine()
        results = engine.search(
            query="hello",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
            mode="vector",
        )
        assert len(results) > 0
        assert results[0].id == "d1"

    def test_bm25_search(self):
        engine = self._make_engine()
        results = engine.search(query="hello", top_k=3, mode="bm25")
        assert len(results) > 0
        assert results[0].id == "d1"

    def test_hybrid_search(self):
        engine = self._make_engine()
        results = engine.search(
            query="hello",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
            mode="hybrid",
        )
        assert len(results) > 0

    def test_hybrid_fuses_results(self):
        engine = self._make_engine()
        # "hello" matches d1 in both vector and BM25
        results = engine.search(
            query="hello",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
            mode="hybrid",
        )
        # d1 should be first (appears in both)
        assert results[0].id == "d1"

    def test_unknown_mode_raises(self):
        engine = self._make_engine()
        try:
            engine.search(query="hello", mode="unknown")
            raise AssertionError("Should have raised")
        except ValueError:
            pass

    def test_vector_no_backend(self):
        engine = SearchEngine(bm25_search=BM25())
        results = engine.search(query="hello", query_vector=[0.1] * 4, mode="vector")
        assert results == []

    def test_bm25_no_backend(self):
        engine = SearchEngine(vector_search=FAISSVectorSearch(dimension=4))
        results = engine.search(query="hello", mode="bm25")
        assert results == []

    def test_metadata_filter(self):
        engine = self._make_engine()
        results = engine.search(
            query="hello",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
            mode="bm25",
            metadata_filter={"type": "a"},
        )
        for r in results:
            assert r.metadata["type"] == "a"
