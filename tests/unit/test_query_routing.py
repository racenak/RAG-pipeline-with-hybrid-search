"""Tests for query routing — classification and strategy routing."""

from __future__ import annotations

import pytest

from rag_pipeline.retrieval.routing import (
    QueryClassifier,
    QueryClassification,
    QueryRouter,
    RetrievalStrategy,
    RoutingConfig,
    RoutingDecision,
)


# ------------------------------------------------------------------ #
#  QueryClassifier — type detection
# ------------------------------------------------------------------ #


class TestQueryClassifierType:
    def test_factual_who(self):
        assert QueryClassifier.classify_type("Who invented Python?") == "factual"

    def test_factual_what(self):
        assert QueryClassifier.classify_type("What is machine learning?") == "factual"

    def test_factual_when(self):
        assert QueryClassifier.classify_type("When was the internet created?") == "factual"

    def test_factual_where(self):
        assert QueryClassifier.classify_type("Where is the Eiffel Tower?") == "factual"

    def test_factual_define(self):
        assert QueryClassifier.classify_type("Define neural network") == "factual"

    def test_factual_how_many(self):
        assert QueryClassifier.classify_type("How many planets are there?") == "factual"

    def test_summarize(self):
        assert QueryClassifier.classify_type("Summarize the document") == "summarization"

    def test_summary(self):
        assert QueryClassifier.classify_type("Give me a summary of the report") == "summarization"

    def test_briefly_explain(self):
        assert QueryClassifier.classify_type("Briefly explain quantum computing") == "summarization"

    def test_overview(self):
        assert QueryClassifier.classify_type("Overview of Docker") == "summarization"

    def test_compare(self):
        assert QueryClassifier.classify_type("Compare React and Vue") == "comparison"

    def test_vs(self):
        assert QueryClassifier.classify_type("Python vs JavaScript") == "comparison"

    def test_difference_between(self):
        assert QueryClassifier.classify_type("Difference between TCP and UDP") == "comparison"

    def test_pros_and_cons(self):
        assert QueryClassifier.classify_type("Pros and cons of microservices") == "comparison"

    def test_how_do(self):
        assert QueryClassifier.classify_type("How do I deploy to production?") == "how_to"

    def test_how_can(self):
        assert QueryClassifier.classify_type("How can I fix this bug?") == "how_to"

    def test_steps_to(self):
        assert QueryClassifier.classify_type("Steps to set up Kubernetes") == "how_to"

    def test_guide_to(self):
        assert QueryClassifier.classify_type("Guide to writing tests") == "how_to"

    def test_tutorial(self):
        assert QueryClassifier.classify_type("Tutorial on Docker networking") == "how_to"

    def test_general_random(self):
        assert QueryClassifier.classify_type("random stuff") == "general"

    def test_general_single_word(self):
        assert QueryClassifier.classify_type("Python") == "general"

    def test_case_insensitive(self):
        assert QueryClassifier.classify_type("SUMMARIZE this") == "summarization"

    def test_no_match_returns_general(self):
        assert QueryClassifier.classify_type("the quick brown fox") == "general"


# ------------------------------------------------------------------ #
#  QueryClassifier — complexity
# ------------------------------------------------------------------ #


class TestQueryClassifierComplexity:
    def test_simple_short(self):
        assert QueryClassifier.classify_complexity("What is Python?") == "simple"

    def test_simple_very_short(self):
        assert QueryClassifier.classify_complexity("Python") == "simple"

    def test_moderate(self):
        assert QueryClassifier.classify_complexity(
            "What is the difference between machine learning versus deep learning?"
        ) == "moderate"

    def test_complex_long(self):
        assert QueryClassifier.classify_complexity(
            "How do I configure a Kubernetes cluster with autoscaling and load balancing "
            "in a multi-region deployment?"
        ) == "complex"

    def test_complex_with_and(self):
        assert QueryClassifier.classify_complexity(
            "What are the best practices and common pitfalls of microservices?"
        ) == "complex"


# ------------------------------------------------------------------ #
#  QueryClassifier — specificity
# ------------------------------------------------------------------ #


class TestQueryClassifierSpecificity:
    def test_broad_short(self):
        assert QueryClassifier.classify_specificity("Python") == "broad"

    def test_broad_four_words(self):
        assert QueryClassifier.classify_specificity("What is Python?") == "broad"

    def test_medium(self):
        assert QueryClassifier.classify_specificity(
            "How does a neural network work?"
        ) == "medium"

    def test_narrow(self):
        assert QueryClassifier.classify_specificity(
            "How do I configure Redis cache TTL for session storage in Python?"
        ) == "narrow"


# ------------------------------------------------------------------ #
#  QueryClassifier — full classify
# ------------------------------------------------------------------ #


class TestQueryClassifierClassify:
    def test_returns_classification(self):
        result = QueryClassifier.classify("What is RAG?")
        assert isinstance(result, QueryClassification)
        assert result.query_type == "factual"
        assert result.complexity == "simple"
        assert result.specificity == "broad"

    def test_complex_narrow_how_to(self):
        result = QueryClassifier.classify(
            "How do I set up a secure Kubernetes cluster with RBAC and network policies?"
        )
        assert result.query_type == "how_to"
        assert result.complexity == "complex"
        assert result.specificity == "narrow"


# ------------------------------------------------------------------ #
#  RetrievalStrategy enum
# ------------------------------------------------------------------ #


class TestRetrievalStrategy:
    def test_vector_only(self):
        assert RetrievalStrategy.VECTOR_ONLY.value == "vector_only"

    def test_bm25_only(self):
        assert RetrievalStrategy.BM25_ONLY.value == "bm25_only"

    def test_hybrid(self):
        assert RetrievalStrategy.HYBRID.value == "hybrid"

    def test_hybrid_rerank(self):
        assert RetrievalStrategy.HYBRID_RERANK.value == "hybrid_rerank"

    def test_step_back(self):
        assert RetrievalStrategy.STEP_BACK.value == "step_back"

    def test_multi_query(self):
        assert RetrievalStrategy.MULTI_QUERY.value == "multi_query"

    def test_all_members(self):
        assert len(RetrievalStrategy) == 6


# ------------------------------------------------------------------ #
#  RoutingConfig
# ------------------------------------------------------------------ #


class TestRoutingConfig:
    def test_defaults(self):
        cfg = RoutingConfig()
        assert cfg.auto_route is True
        assert cfg.default_top_k == 5
        assert RetrievalStrategy.HYBRID_RERANK in cfg.strategy_map.values()
        assert cfg.top_k_map["factual"] == 5
        assert cfg.rerank_map["factual"] is True

    def test_custom_config(self):
        cfg = RoutingConfig(
            auto_route=False,
            default_top_k=10,
            strategy_map={"factual": RetrievalStrategy.VECTOR_ONLY},
        )
        assert cfg.auto_route is False
        assert cfg.default_top_k == 10
        assert cfg.strategy_map["factual"] == RetrievalStrategy.VECTOR_ONLY


# ------------------------------------------------------------------ #
#  QueryRouter
# ------------------------------------------------------------------ #


class TestQueryRouter:
    def test_default_config_factual(self):
        router = QueryRouter()
        decision = router.route("What is Python?")
        assert decision.strategy == RetrievalStrategy.HYBRID_RERANK
        assert decision.top_k == 10  # broad query gets +3, capped at 10
        assert decision.rerank is True

    def test_default_config_summarization(self):
        router = QueryRouter()
        decision = router.route("Summarize the document")
        assert decision.strategy == RetrievalStrategy.HYBRID
        assert decision.rerank is False

    def test_default_config_comparison(self):
        router = QueryRouter()
        decision = router.route("Compare React and Vue")
        assert decision.strategy == RetrievalStrategy.HYBRID

    def test_default_config_how_to(self):
        router = QueryRouter()
        decision = router.route("How do I deploy to production?")
        assert decision.strategy == RetrievalStrategy.HYBRID_RERANK
        assert decision.rerank is True

    def test_default_config_general(self):
        router = QueryRouter()
        decision = router.route("random stuff")
        assert decision.strategy == RetrievalStrategy.HYBRID

    def test_auto_route_disabled(self):
        cfg = RoutingConfig(auto_route=False)
        router = QueryRouter(config=cfg)
        decision = router.route("How do I deploy to production?")
        # With auto_route off, complex adjustments don't apply
        assert decision.top_k == cfg.default_top_k

    def test_auto_route_disabled_uses_config_top_k(self):
        cfg = RoutingConfig(auto_route=False, default_top_k=15)
        router = QueryRouter(config=cfg)
        decision = router.route("anything")
        assert decision.top_k == 15

    def test_complex_query_broadens_top_k(self):
        router = QueryRouter()
        decision = router.route(
            "How do I configure a Kubernetes cluster with autoscaling "
            "and load balancing in production?"
        )
        assert decision.top_k > 5

    def test_broad_query_increases_top_k(self):
        router = QueryRouter()
        decision = router.route("Python")
        assert decision.top_k >= 10

    def test_returns_classification(self):
        router = QueryRouter()
        decision = router.route("What is RAG?")
        assert isinstance(decision.classification, QueryClassification)
        assert decision.classification.query_type == "factual"

    def test_custom_strategy_map(self):
        cfg = RoutingConfig(
            strategy_map={"factual": RetrievalStrategy.VECTOR_ONLY}
        )
        router = QueryRouter(config=cfg)
        decision = router.route("What is Python?")
        assert decision.strategy == RetrievalStrategy.VECTOR_ONLY
