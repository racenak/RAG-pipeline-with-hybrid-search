"""Tests for hybrid search with RRF fusion."""

from rag_pipeline.retrieval.hybrid import (
    HybridSearch,
    HybridSearchConfig,
    reciprocal_rank_fusion,
    score_fusion,
)
from rag_pipeline.retrieval.vector import SearchResult


def _results(ids: list[str], scores: list[float] | None = None) -> list[SearchResult]:
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        SearchResult(id=i, score=s, content=f"content {i}")
        for i, s in zip(ids, scores, strict=True)
    ]


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        list1 = _results(["d1", "d2", "d3"])
        list2 = _results(["d2", "d3", "d4"])
        fused = reciprocal_rank_fusion([list1, list2])

        ids = [r.id for r in fused]
        # d2 and d3 appear in both lists, should rank high
        assert "d2" in ids
        assert "d3" in ids
        assert len(ids) == 4  # d1, d2, d3, d4

    def test_single_list(self):
        list1 = _results(["d1", "d2"])
        fused = reciprocal_rank_fusion([list1])
        assert len(fused) == 2
        assert fused[0].id == "d1"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []

    def test_weights(self):
        list1 = _results(["d1", "d2"])
        list2 = _results(["d2", "d1"])
        # Give more weight to list2
        fused = reciprocal_rank_fusion([list1, list2], weights=[1.0, 2.0])
        # d2 is rank 0 in list2 (weight 2.0), should be first
        assert fused[0].id == "d2"

    def test_k_parameter(self):
        list1 = _results(["d1", "d2", "d3"])
        list2 = _results(["d2", "d3", "d4"])

        fused_low_k = reciprocal_rank_fusion([list1, list2], k=1)
        fused_high_k = reciprocal_rank_fusion([list1, list2], k=100)

        # Lower k gives more weight to top ranks
        # Both should have same ordering but different score magnitudes
        assert len(fused_low_k) == len(fused_high_k)

    def test_duplicate_handling(self):
        list1 = _results(["d1", "d1", "d2"])
        fused = reciprocal_rank_fusion([list1])
        ids = [r.id for r in fused]
        assert ids.count("d1") == 1  # deduplicated

    def test_score_accumulation(self):
        # d1 is rank 0 in both lists
        list1 = _results(["d1", "d2"])
        list2 = _results(["d1", "d3"])
        fused = reciprocal_rank_fusion([list1, list2])
        # d1 should be first (appears in both)
        assert fused[0].id == "d1"
        # Score should be sum of both RRF scores
        assert fused[0].score > 1 / 61  # > single list score

    def test_weight_mismatch_raises(self):
        list1 = _results(["d1"])
        try:
            reciprocal_rank_fusion([list1], weights=[1.0, 2.0])
            raise AssertionError("Should have raised")
        except ValueError:
            pass


class TestScoreFusion:
    def test_basic_fusion(self):
        list1 = _results(["d1", "d2", "d3"])
        list2 = _results(["d2", "d3", "d4"])
        fused = score_fusion([list1, list2])
        ids = [r.id for r in fused]
        assert "d2" in ids
        assert "d3" in ids

    def test_empty_lists(self):
        assert score_fusion([]) == []

    def test_single_list(self):
        list1 = _results(["d1", "d2"])
        fused = score_fusion([list1])
        assert len(fused) == 2


class TestHybridSearch:
    def test_fuse_both(self):
        hs = HybridSearch()
        vec = _results(["d1", "d2", "d3"])
        bm25 = _results(["d2", "d3", "d4"])
        fused = hs.fuse(vec, bm25)
        ids = [r.id for r in fused]
        assert len(ids) == 4

    def test_fuse_vector_only(self):
        hs = HybridSearch()
        vec = _results(["d1", "d2"])
        fused = hs.fuse(vec, None)
        assert len(fused) == 2

    def test_fuse_bm25_only(self):
        hs = HybridSearch()
        bm25 = _results(["d1", "d2"])
        fused = hs.fuse(None, bm25)
        assert len(fused) == 2

    def test_fuse_none(self):
        hs = HybridSearch()
        assert hs.fuse(None, None) == []

    def test_vector_disabled(self):
        config = HybridSearchConfig(vector_enabled=False)
        hs = HybridSearch(config)
        vec = _results(["d1", "d2"])
        bm25 = _results(["d2", "d3"])
        fused = hs.fuse(vec, bm25)
        # Only BM25 results
        ids = [r.id for r in fused]
        assert "d1" not in ids

    def test_bm25_disabled(self):
        config = HybridSearchConfig(bm25_enabled=False)
        hs = HybridSearch(config)
        vec = _results(["d1", "d2"])
        bm25 = _results(["d2", "d3"])
        fused = hs.fuse(vec, bm25)
        ids = [r.id for r in fused]
        assert "d3" not in ids

    def test_score_fusion_strategy(self):
        config = HybridSearchConfig(fusion_strategy="score")
        hs = HybridSearch(config)
        vec = _results(["d1", "d2"])
        bm25 = _results(["d2", "d3"])
        fused = hs.fuse(vec, bm25)
        assert len(fused) > 0

    def test_unknown_strategy_raises(self):
        config = HybridSearchConfig(fusion_strategy="unknown")
        hs = HybridSearch(config)
        try:
            hs.fuse(_results(["d1"]), _results(["d1"]))
            raise AssertionError("Should have raised")
        except ValueError:
            pass

    def test_custom_weights(self):
        config = HybridSearchConfig(vector_weight=2.0, bm25_weight=1.0)
        hs = HybridSearch(config)
        vec = _results(["d1", "d2"])
        bm25 = _results(["d2", "d3"])
        fused = hs.fuse(vec, bm25)
        # d2 appears in both, should be first
        assert fused[0].id == "d2"
