"""Tests for retrieval evaluation framework."""

import pytest

from rag_pipeline.evaluation.retrieval import (
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["doc1", "doc2", "doc3"]
        expected = ["doc1", "doc2"]
        assert precision_at_k(retrieved, expected, 3) == pytest.approx(2 / 3)

    def test_zero_precision(self):
        retrieved = ["doc4", "doc5"]
        expected = ["doc1", "doc2"]
        assert precision_at_k(retrieved, expected, 2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["doc1"], 5) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["doc1", "doc2", "doc3"]
        expected = ["doc1", "doc2"]
        assert recall_at_k(retrieved, expected, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["doc1", "doc3"]
        expected = ["doc1", "doc2"]
        assert recall_at_k(retrieved, expected, 2) == 0.5

    def test_zero_recall(self):
        retrieved = ["doc3", "doc4"]
        expected = ["doc1", "doc2"]
        assert recall_at_k(retrieved, expected, 2) == 0.0


class TestMRR:
    def test_first_rank(self):
        retrieved = ["doc1", "doc2"]
        expected = ["doc1"]
        assert mean_reciprocal_rank(retrieved, expected) == 1.0

    def test_second_rank(self):
        retrieved = ["doc2", "doc1"]
        expected = ["doc1"]
        assert mean_reciprocal_rank(retrieved, expected) == 0.5

    def test_no_match(self):
        retrieved = ["doc3", "doc4"]
        expected = ["doc1"]
        assert mean_reciprocal_rank(retrieved, expected) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        retrieved = ["doc1", "doc2"]
        expected = ["doc1", "doc2"]
        assert ndcg_at_k(retrieved, expected, 2) == pytest.approx(1.0)

    def test_imperfect_ndcg(self):
        # With binary relevance, having some irrelevant docs reduces NDCG
        retrieved = ["doc1", "doc3"]  # doc3 is not relevant
        expected = ["doc1", "doc2"]
        score = ndcg_at_k(retrieved, expected, 2)
        assert 0 < score < 1


class TestHitRate:
    def test_hit(self):
        retrieved = ["doc1", "doc3"]
        expected = ["doc1"]
        assert hit_rate(retrieved, expected) == 1.0

    def test_miss(self):
        retrieved = ["doc3", "doc4"]
        expected = ["doc1"]
        assert hit_rate(retrieved, expected) == 0.0


class TestMAP:
    def test_perfect_map(self):
        retrieved = ["doc1", "doc2"]
        expected = ["doc1", "doc2"]
        assert mean_average_precision(retrieved, expected) == pytest.approx(1.0)

    def test_partial_map(self):
        retrieved = ["doc1", "doc3"]
        expected = ["doc1", "doc2"]
        score = mean_average_precision(retrieved, expected)
        assert 0 < score < 1
