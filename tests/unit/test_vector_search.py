"""Tests for vector search (FAISS backend)."""

import numpy as np

from rag_pipeline.retrieval.vector import FAISSVectorSearch, SearchResult


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(id="d1", score=0.9, content="hello", metadata={"k": "v"})
        d = r.to_dict()
        assert d["id"] == "d1"
        assert d["score"] == 0.9
        assert d["metadata"]["k"] == "v"

    def test_to_dict_no_metadata(self):
        r = SearchResult(id="d1", score=0.9, content="hello")
        d = r.to_dict()
        assert "metadata" not in d


class TestFAISSVectorSearch:
    def _make_vectors(self, n: int = 5, dim: int = 128) -> list[list[float]]:
        rng = np.random.default_rng(42)
        return [rng.standard_normal(dim).tolist() for _ in range(n)]

    def test_index_and_search(self):
        faiss = FAISSVectorSearch(dimension=128)
        vectors = self._make_vectors(5, 128)
        ids = [f"d{i}" for i in range(5)]
        contents = [f"doc {i}" for i in range(5)]

        count = faiss.index_vectors(ids, vectors, contents)
        assert count == 5
        assert faiss.doc_count == 5

        results = faiss.search(vectors[0], top_k=3)
        assert len(results) > 0
        assert results[0].id == "d0"  # exact match should be first

    def test_search_empty(self):
        faiss = FAISSVectorSearch(dimension=128)
        results = faiss.search([0.0] * 128, top_k=5)
        assert results == []

    def test_index_empty(self):
        faiss = FAISSVectorSearch(dimension=128)
        count = faiss.index_vectors([], [], [])
        assert count == 0

    def test_search_with_threshold(self):
        faiss = FAISSVectorSearch(dimension=128)
        vectors = self._make_vectors(5, 128)
        faiss.index_vectors([f"d{i}" for i in range(5)], vectors, [f"c{i}" for i in range(5)])

        results = faiss.search(vectors[0], top_k=5, threshold=0.99)
        # Only exact match should pass high threshold
        assert len(results) <= 1

    def test_search_with_metadata_filter(self):
        faiss = FAISSVectorSearch(dimension=128)
        vectors = self._make_vectors(5, 128)
        metadatas = [{"type": "a"}, {"type": "b"}, {"type": "a"}, {"type": "b"}, {"type": "a"}]
        faiss.index_vectors(
            [f"d{i}" for i in range(5)],
            vectors,
            [f"c{i}" for i in range(5)],
            metadatas=metadatas,
        )

        results = faiss.search(vectors[0], top_k=5, metadata_filter={"type": "a"})
        for r in results:
            assert r.metadata["type"] == "a"

    def test_clear(self):
        faiss = FAISSVectorSearch(dimension=128)
        faiss.index_vectors(["d1"], [[0.1] * 128], ["c1"])
        assert faiss.doc_count == 1
        faiss.clear()
        assert faiss.doc_count == 0

    def test_cosine_similarity_ordering(self):
        faiss = FAISSVectorSearch(dimension=4)
        # v0 and v1 are similar, v2 is different
        v0 = [1.0, 0.0, 0.0, 0.0]
        v1 = [0.9, 0.1, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0, 1.0]
        faiss.index_vectors(["d0", "d1", "d2"], [v0, v1, v2], ["c0", "c1", "c2"])

        results = faiss.search(v0, top_k=3)
        assert results[0].id == "d0"
        assert results[1].id == "d1"
        assert results[2].id == "d2"
