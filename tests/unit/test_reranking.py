"""Tests for reranking — CrossEncoderReranker and NoopReranker."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag_pipeline.retrieval.reranking import (
    CrossEncoderReranker,
    NoopReranker,
    get_reranker,
)
from rag_pipeline.retrieval.vector import SearchResult

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _results(ids: list[str], scores: list[float] | None = None) -> list[SearchResult]:
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(ids))]
    return [
        SearchResult(id=i, score=s, content=f"content {i}")
        for i, s in zip(ids, scores, strict=True)
    ]


# ------------------------------------------------------------------ #
#  NoopReranker
# ------------------------------------------------------------------ #


class TestNoopReranker:
    def test_returns_results_unchanged(self):
        reranker = NoopReranker()
        results = _results(["d1", "d2", "d3"])
        reranked = reranker.rerank("query", results, top_k=10)
        assert len(reranked) == 3
        assert [r.id for r in reranked] == ["d1", "d2", "d3"]

    def test_truncates_to_top_k(self):
        reranker = NoopReranker()
        results = _results(["d1", "d2", "d3", "d4", "d5"])
        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2
        assert reranked[0].id == "d1"
        assert reranked[1].id == "d2"

    def test_empty_results(self):
        reranker = NoopReranker()
        reranked = reranker.rerank("query", [], top_k=10)
        assert reranked == []

    def test_preserves_metadata(self):
        reranker = NoopReranker()
        results = [
            SearchResult(id="d1", score=1.0, content="hello", metadata={"source": "test"}),
        ]
        reranked = reranker.rerank("query", results, top_k=10)
        assert reranked[0].metadata == {"source": "test"}


# ------------------------------------------------------------------ #
#  CrossEncoderReranker — with mocked model
# ------------------------------------------------------------------ #


class TestCrossEncoderReranker:
    def test_reranks_by_score_descending(self):
        """Cross-encoder scores should reorder results."""
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2", "d3"])

        # Mock model: give d3 highest score, d1 lowest
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.5, 0.9]
        reranker._model = mock_model

        reranked = reranker.rerank("query", results, top_k=10)

        assert len(reranked) == 3
        assert reranked[0].id == "d3"  # highest score
        assert reranked[1].id == "d2"
        assert reranked[2].id == "d1"  # lowest score
        # Scores should be updated to cross-encoder scores
        assert reranked[0].score == 0.9

    def test_truncates_to_top_k(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2", "d3", "d4", "d5"])

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        reranker._model = mock_model

        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_empty_results(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        reranked = reranker.rerank("query", [], top_k=10)
        assert reranked == []

    def test_empty_query_returns_original_order(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2"])

        reranked = reranker.rerank("", results, top_k=10)
        assert len(reranked) == 2
        assert reranked[0].id == "d1"

    def test_whitespace_query_returns_original_order(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2"])

        reranked = reranker.rerank("   ", results, top_k=10)
        assert len(reranked) == 2

    def test_model_load_failure_returns_original_order(self):
        """If model fails to load, reranker should fall back gracefully."""
        reranker = CrossEncoderReranker(model_name="nonexistent-model")
        results = _results(["d1", "d2"])

        # _load_model will fail because model doesn't exist
        reranked = reranker.rerank("query", results, top_k=10)
        # Should fall back to original order
        assert len(reranked) == 2
        assert reranked[0].id == "d1"

    def test_scoring_exception_returns_original_order(self):
        """If predict() raises, reranker should fall back gracefully."""
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2"])

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU out of memory")
        reranker._model = mock_model

        reranked = reranker.rerank("query", results, top_k=10)
        # Should fall back to original order
        assert len(reranked) == 2
        assert reranked[0].id == "d1"

    def test_preserves_metadata(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = [
            SearchResult(id="d1", score=1.0, content="hello", metadata={"source": "test"}),
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        reranker._model = mock_model

        reranked = reranker.rerank("query", results, top_k=10)
        assert reranked[0].metadata == {"source": "test"}

    def test_predict_receives_correct_pairs(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        results = _results(["d1", "d2"])

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3]
        reranker._model = mock_model

        reranker.rerank("my query", results, top_k=10)

        # Check that predict was called with correct pairs
        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("my query", "content d1"), ("my query", "content d2")]

    def test_batch_size_passed_to_predict(self):
        reranker = CrossEncoderReranker(model_name="fake-model", batch_size=16)
        results = _results(["d1"])

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        reranker._model = mock_model

        reranker.rerank("query", results, top_k=10)

        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_model_loaded_lazily(self):
        """Model should not be loaded until first rerank call."""
        reranker = CrossEncoderReranker(model_name="fake-model")
        assert reranker._model is None

        # After successful rerank, model should be cached
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        reranker._model = mock_model

        reranker.rerank("query", _results(["d1"]), top_k=10)
        # Second call should reuse cached model
        reranker.rerank("query", _results(["d1"]), top_k=10)
        assert mock_model.predict.call_count == 2


# ------------------------------------------------------------------ #
#  get_reranker factory
# ------------------------------------------------------------------ #


class TestGetReranker:
    def test_disabled_returns_noop(self):
        from rag_pipeline.config import RetrievalSettings

        settings = RetrievalSettings(rerank_enabled=False)
        reranker = get_reranker(settings)
        assert isinstance(reranker, NoopReranker)

    def test_enabled_returns_cross_encoder(self):
        from rag_pipeline.config import RetrievalSettings

        settings = RetrievalSettings(rerank_enabled=True)
        reranker = get_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_default_settings(self):
        reranker = get_reranker()
        # Default config has rerank_enabled=True
        assert isinstance(reranker, CrossEncoderReranker)

    def test_custom_model_name(self):
        from rag_pipeline.config import RetrievalSettings

        settings = RetrievalSettings(
            rerank_enabled=True,
            rerank_model="custom/model-name",
        )
        reranker = get_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker._model_name == "custom/model-name"
