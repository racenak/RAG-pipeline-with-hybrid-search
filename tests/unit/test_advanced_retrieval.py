"""Tests for advanced retrieval — LLM-powered strategies and metadata filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_pipeline.retrieval.advanced import (
    AdvancedQueryConfig,
    AdvancedRetrievalPipeline,
    LLMHydeGenerator,
    LLMQueryExpander,
    LLMMultiQueryGenerator,
    LLMStepBackGenerator,
    MetadataFilter,
)
from rag_pipeline.retrieval.vector import SearchResult


# ------------------------------------------------------------------ #
#  Helper: mock LLM backend
# ------------------------------------------------------------------ #


@dataclass
class FakeBackend:
    """Minimal fake LLMBackend for testing."""

    _response: str = ""

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: Any = None,
    ) -> str:
        return self._response

    async def stream(
        self,
        messages: list[dict[str, str]],
        config: Any = None,
    ):
        yield self._response


def make_search_result(
    doc_id: str = "1",
    score: float = 0.9,
    content: str = "test content",
    metadata: dict[str, Any] | None = None,
) -> SearchResult:
    return SearchResult(id=doc_id, score=score, content=content, metadata=metadata)


# ------------------------------------------------------------------ #
#  LLMQueryExpander
# ------------------------------------------------------------------ #


class TestLLMQueryExpander:
    def test_fallback_splits_words(self):
        expander = LLMQueryExpander(backend=None)
        result = _run(expander.expand("machine learning algorithm"))
        # Should return expanded query with content words repeated
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "machine learning algorithm" in result[0]

    def test_fallback_filters_stop_words(self):
        expander = LLMQueryExpander(backend=None)
        result = _run(expander.expand("the is are was"))
        # Stop words only — just returns original
        assert result == ["the is are was"]

    def test_fallback_empty_query(self):
        expander = LLMQueryExpander(backend=None)
        result = _run(expander.expand(""))
        assert result == [""]

    def test_with_mock_backend(self):
        backend = FakeBackend(_response="AI, neural networks, deep learning")
        expander = LLMQueryExpander(backend=backend)
        result = _run(expander.expand("What is machine learning?"))
        assert len(result) >= 1
        assert "machine learning" in result[0]
        assert "AI" in result[0] or "neural" in result[0]

    def test_with_failed_llm_fallback(self):
        backend = AsyncMock()
        backend.generate.side_effect = RuntimeError("LLM unavailable")
        expander = LLMQueryExpander(backend=backend)
        result = _run(expander.expand("fix the error"))
        # Should fall back to deterministic expansion
        assert isinstance(result, list)
        assert len(result) >= 1


# ------------------------------------------------------------------ #
#  LLMHydeGenerator
# ------------------------------------------------------------------ #


class TestLLMHydeGenerator:
    def test_fallback_prepends_prefix(self):
        hyde = LLMHydeGenerator(backend=None)
        result = _run(hyde.generate("What is RAG?"))
        assert result.startswith("This document provides a detailed answer about:")
        assert "What is RAG?" in result

    def test_with_mock_backend(self):
        backend = FakeBackend(_response="RAG combines retrieval augmented generation.")
        hyde = LLMHydeGenerator(backend=backend)
        result = _run(hyde.generate("What is RAG?"))
        assert "retrieval augmented generation" in result.lower()

    def test_with_failed_llm_fallback(self):
        backend = AsyncMock()
        backend.generate.side_effect = RuntimeError("fail")
        hyde = LLMHydeGenerator(backend=backend)
        result = _run(hyde.generate("test query"))
        assert "This document provides" in result


# ------------------------------------------------------------------ #
#  LLMMultiQueryGenerator
# ------------------------------------------------------------------ #


class TestLLMMultiQueryGenerator:
    def test_fallback_generates_variations(self):
        mq = LLMMultiQueryGenerator(backend=None, num_queries=4)
        result = _run(mq.generate("What is RAG?"))
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0] == "What is RAG?"

    def test_fallback_minimum_two(self):
        mq = LLMMultiQueryGenerator(backend=None, num_queries=1)
        result = _run(mq.generate("test"))
        assert len(result) >= 2

    def test_with_mock_backend(self):
        backend = FakeBackend(
            _response="Explain RAG technology.\nHow does RAG work?\nRAG overview."
        )
        mq = LLMMultiQueryGenerator(backend=backend, num_queries=4)
        result = _run(mq.generate("What is RAG?"))
        assert result[0] == "What is RAG?"
        assert len(result) >= 2

    def test_with_failed_llm_fallback(self):
        backend = AsyncMock()
        backend.generate.side_effect = RuntimeError("fail")
        mq = LLMMultiQueryGenerator(backend=backend, num_queries=3)
        result = _run(mq.generate("test query"))
        assert isinstance(result, list)
        assert len(result) >= 2


# ------------------------------------------------------------------ #
#  LLMStepBackGenerator
# ------------------------------------------------------------------ #


class TestLLMStepBackGenerator:
    def test_fallback_extracts_key_terms(self):
        sb = LLMStepBackGenerator(backend=None)
        result = _run(sb.generate("How do I configure Redis cache TTL in Python?"))
        assert "Overview" in result or "background" in result
        assert result != "How do I configure Redis cache TTL in Python?"

    def test_fallback_stop_words_only(self):
        sb = LLMStepBackGenerator(backend=None)
        result = _run(sb.generate("is the?"))
        assert "General information" in result or "topic" in result

    def test_with_mock_backend(self):
        backend = FakeBackend(
            _response="What is the general concept of caching in distributed systems?"
        )
        sb = LLMStepBackGenerator(backend=backend)
        result = _run(sb.generate("How do I configure Redis cache TTL?"))
        assert "caching" in result.lower()

    def test_with_failed_llm_fallback(self):
        backend = AsyncMock()
        backend.generate.side_effect = RuntimeError("fail")
        sb = LLMStepBackGenerator(backend=backend)
        result = _run(sb.generate("What is machine learning?"))
        assert "Overview" in result or "background" in result


# ------------------------------------------------------------------ #
#  MetadataFilter
# ------------------------------------------------------------------ #


class TestMetadataFilter:
    def test_date_range_in_range(self):
        f = MetadataFilter(date_from="2024-01-01", date_to="2024-12-31")
        results = [
            make_search_result("1", metadata={"date": "2024-06-15"}),
            make_search_result("2", metadata={"date": "2024-09-20"}),
        ]
        filtered = f.filter(results)
        assert len(filtered) == 2

    def test_date_range_out_of_range(self):
        f = MetadataFilter(date_from="2024-06-01", date_to="2024-12-31")
        results = [
            make_search_result("1", metadata={"date": "2024-03-15"}),
            make_search_result("2", metadata={"date": "2024-09-20"}),
        ]
        filtered = f.filter(results)
        assert len(filtered) == 1
        assert filtered[0].id == "2"

    def test_date_range_no_date_field(self):
        f = MetadataFilter(date_from="2024-01-01", date_to="2024-12-31")
        results = [make_search_result("1", metadata={"other": "value"})]
        filtered = f.filter(results)
        assert len(filtered) == 0

    def test_date_range_datetime_objects(self):
        f = MetadataFilter(date_from="2024-01-01", date_to="2024-12-31")
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        results = [make_search_result("1", metadata={"date": dt})]
        filtered = f.filter(results)
        assert len(filtered) == 1

    def test_document_types(self):
        f = MetadataFilter(document_types=["pdf", "article"])
        results = [
            make_search_result("1", metadata={"type": "pdf"}),
            make_search_result("2", metadata={"type": "video"}),
            make_search_result("3", metadata={"type": "article"}),
        ]
        filtered = f.filter(results)
        assert len(filtered) == 2
        ids = [r.id for r in filtered]
        assert "1" in ids
        assert "3" in ids

    def test_document_types_no_match(self):
        f = MetadataFilter(document_types=["pdf"])
        results = [make_search_result("1", metadata={"type": "video"})]
        filtered = f.filter(results)
        assert len(filtered) == 0

    def test_custom_filters(self):
        f = MetadataFilter(custom_filters={"author": "John", "lang": "en"})
        results = [
            make_search_result("1", metadata={"author": "John", "lang": "en"}),
            make_search_result("2", metadata={"author": "Jane", "lang": "en"}),
            make_search_result("3", metadata={"author": "John", "lang": "fr"}),
        ]
        filtered = f.filter(results)
        assert len(filtered) == 1
        assert filtered[0].id == "1"

    def test_custom_filters_missing_key(self):
        f = MetadataFilter(custom_filters={"author": "John"})
        results = [make_search_result("1", metadata={"other": "value"})]
        filtered = f.filter(results)
        assert len(filtered) == 0

    def test_multiple_filters_combined(self):
        f = MetadataFilter(
            date_from="2024-01-01",
            date_to="2024-12-31",
            document_types=["pdf"],
            custom_filters={"lang": "en"},
        )
        results = [
            make_search_result("1", metadata={"date": "2024-06-15", "type": "pdf", "lang": "en"}),
            make_search_result("2", metadata={"date": "2024-06-15", "type": "video", "lang": "en"}),
            make_search_result("3", metadata={"date": "2023-01-01", "type": "pdf", "lang": "en"}),
            make_search_result("4", metadata={"date": "2024-06-15", "type": "pdf", "lang": "fr"}),
        ]
        filtered = f.filter(results)
        assert len(filtered) == 1
        assert filtered[0].id == "1"

    def test_no_filters_returns_all(self):
        f = MetadataFilter()
        results = [make_search_result("1"), make_search_result("2")]
        filtered = f.filter(results)
        assert len(filtered) == 2

    def test_empty_results(self):
        f = MetadataFilter(date_from="2024-01-01")
        assert f.filter([]) == []

    def test_unparseable_date_string(self):
        f = MetadataFilter(date_from="2024-01-01", date_to="2024-12-31")
        results = [make_search_result("1", metadata={"date": "not-a-date"})]
        filtered = f.filter(results)
        assert len(filtered) == 0


# ------------------------------------------------------------------ #
#  AdvancedQueryConfig
# ------------------------------------------------------------------ #


class TestAdvancedQueryConfig:
    def test_defaults(self):
        cfg = AdvancedQueryConfig()
        assert cfg.expand_enabled is True
        assert cfg.hyde_enabled is True
        assert cfg.multi_query_enabled is True
        assert cfg.step_back_enabled is True
        assert cfg.metadata_filter_enabled is False
        assert cfg.multi_query_count == 4
        assert cfg.date_from is None
        assert cfg.date_to is None
        assert cfg.document_types == []
        assert cfg.custom_filters == {}

    def test_custom_config(self):
        cfg = AdvancedQueryConfig(
            expand_enabled=False,
            multi_query_count=3,
            date_from="2024-01-01",
            document_types=["pdf"],
        )
        assert cfg.expand_enabled is False
        assert cfg.multi_query_count == 3
        assert cfg.date_from == "2024-01-01"
        assert cfg.document_types == ["pdf"]


# ------------------------------------------------------------------ #
#  AdvancedRetrievalPipeline
# ------------------------------------------------------------------ #


class TestAdvancedRetrievalPipeline:
    def test_all_strategies_enabled(self):
        backend = FakeBackend(_response="related terms, AI, ML")
        cfg = AdvancedQueryConfig(
            expand_enabled=True,
            hyde_enabled=True,
            multi_query_enabled=True,
            step_back_enabled=True,
        )
        pipeline = AdvancedRetrievalPipeline(backend=backend, config=cfg)
        result = _run(pipeline.process("What is machine learning?"))
        assert "expanded_queries" in result
        assert "hyde_query" in result
        assert "multi_queries" in result
        assert "step_back_query" in result
        assert len(result["expanded_queries"]) >= 1
        assert len(result["multi_queries"]) >= 2
        assert result["hyde_query"] != ""
        assert result["step_back_query"] != ""

    def test_all_strategies_disabled(self):
        cfg = AdvancedQueryConfig(
            expand_enabled=False,
            hyde_enabled=False,
            multi_query_enabled=False,
            step_back_enabled=False,
        )
        pipeline = AdvancedRetrievalPipeline(backend=None, config=cfg)
        result = _run(pipeline.process("What is RAG?"))
        assert result["expanded_queries"] == []
        assert result["hyde_query"] == ""
        assert result["multi_queries"] == []
        assert result["step_back_query"] == ""

    def test_metadata_filter_enabled(self):
        cfg = AdvancedQueryConfig(
            expand_enabled=False,
            hyde_enabled=False,
            multi_query_enabled=False,
            step_back_enabled=False,
            metadata_filter_enabled=True,
            document_types=["pdf"],
        )
        pipeline = AdvancedRetrievalPipeline(backend=None, config=cfg)
        results = [
            make_search_result("1", metadata={"type": "pdf"}),
            make_search_result("2", metadata={"type": "video"}),
        ]
        result = _run(pipeline.process("test", results=results))
        assert len(result["filtered_results"]) == 1

    def test_metrics_tracked(self):
        cfg = AdvancedQueryConfig(
            expand_enabled=True,
            hyde_enabled=False,
            multi_query_enabled=False,
            step_back_enabled=False,
        )
        pipeline = AdvancedRetrievalPipeline(backend=None, config=cfg)
        result = _run(pipeline.process("test query"))
        metrics = result["metrics"]
        assert "expand" in metrics.strategies_used
        assert metrics.total_latency_ms >= 0

    def test_failed_strategy_graceful(self):
        backend = AsyncMock()
        backend.generate.side_effect = RuntimeError("LLM down")
        pipeline = AdvancedRetrievalPipeline(backend=backend)
        result = _run(pipeline.process("test query"))
        # Should not raise — failures are logged and skipped
        assert "metrics" in result


# ------------------------------------------------------------------ #
#  Helper
# ------------------------------------------------------------------ #


def _run(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


import asyncio  # noqa: E402
