"""Query routing — classify queries and route to optimal retrieval strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ------------------------------------------------------------------ #
#  Retrieval strategies
# ------------------------------------------------------------------ #


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    STEP_BACK = "step_back"
    MULTI_QUERY = "multi_query"


# ------------------------------------------------------------------ #
#  Routing config
# ------------------------------------------------------------------ #


@dataclass
class RoutingConfig:
    """Configuration for query routing: strategy maps, top-k, rerank settings."""

    strategy_map: dict[str, RetrievalStrategy] = field(default_factory=lambda: {
        "factual": RetrievalStrategy.HYBRID_RERANK,
        "summarization": RetrievalStrategy.HYBRID,
        "comparison": RetrievalStrategy.HYBRID,
        "how_to": RetrievalStrategy.HYBRID_RERANK,
        "general": RetrievalStrategy.HYBRID,
    })
    top_k_map: dict[str, int] = field(default_factory=lambda: {
        "factual": 5,
        "summarization": 10,
        "comparison": 10,
        "how_to": 8,
        "general": 5,
    })
    rerank_map: dict[str, bool] = field(default_factory=lambda: {
        "factual": True,
        "summarization": False,
        "comparison": True,
        "how_to": True,
        "general": False,
    })
    default_top_k: int = 5
    auto_route: bool = True


# ------------------------------------------------------------------ #
#  Query classifier
# ------------------------------------------------------------------ #

_QUERY_TYPE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "factual": [
        re.compile(r"\b(who|what|when|where|which|how many|how much)\b", re.I),
        re.compile(r"\b(define|definition of|meaning of)\b", re.I),
    ],
    "summarization": [
        re.compile(r"\b(summarize|summary|summarise|briefly explain)\b", re.I),
        re.compile(r"\b(overview of|tldr|tl;dr)\b", re.I),
    ],
    "comparison": [
        re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.I),
        re.compile(r"\b(pros and cons|advantages and disadvantages)\b", re.I),
    ],
    "how_to": [
        re.compile(r"\b(how (do|can|to|should)|steps to|guide to)\b", re.I),
        re.compile(r"\b(tutorial|walkthrough|instructions for)\b", re.I),
    ],
}


@dataclass
class QueryClassification:
    """Result of classifying a query."""

    query_type: str = "general"
    complexity: str = "simple"
    specificity: str = "medium"


class QueryClassifier:
    """Classify queries by type, complexity, and specificity using regex."""

    @staticmethod
    def classify_type(query: str) -> str:
        for qtype, patterns in _QUERY_TYPE_PATTERNS.items():
            for pat in patterns:
                if pat.search(query):
                    return qtype
        return "general"

    @staticmethod
    def classify_complexity(query: str) -> str:
        word_count = len(query.split())
        if word_count > 15 or " and " in query.lower():
            return "complex"
        if word_count > 8:
            return "moderate"
        return "simple"

    @staticmethod
    def classify_specificity(query: str) -> str:
        word_count = len(query.split())
        if word_count <= 4:
            return "broad"
        if word_count >= 10:
            return "narrow"
        return "medium"

    @classmethod
    def classify(cls, query: str) -> QueryClassification:
        return QueryClassification(
            query_type=cls.classify_type(query),
            complexity=cls.classify_complexity(query),
            specificity=cls.classify_specificity(query),
        )


# ------------------------------------------------------------------ #
#  Query router
# ------------------------------------------------------------------ #


@dataclass
class RoutingDecision:
    """Output of query routing."""

    strategy: RetrievalStrategy
    top_k: int
    rerank: bool
    classification: QueryClassification


class QueryRouter:
    """Route queries to optimal retrieval strategies based on classification."""

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self._config = config or RoutingConfig()
        self._classifier = QueryClassifier()

    def route(self, query: str) -> RoutingDecision:
        cls = self._classifier.classify(query)

        if not self._config.auto_route:
            default_strategy = self._config.strategy_map.get(
                cls.query_type, RetrievalStrategy.HYBRID
            )
            return RoutingDecision(
                strategy=default_strategy,
                top_k=self._config.default_top_k,
                rerank=self._config.rerank_map.get(cls.query_type, False),
                classification=cls,
            )

        strategy = self._config.strategy_map.get(
            cls.query_type, RetrievalStrategy.HYBRID
        )
        top_k = self._config.top_k_map.get(
            cls.query_type, self._config.default_top_k
        )
        rerank = self._config.rerank_map.get(cls.query_type, False)

        # Adjust for complex queries — broaden results
        if cls.complexity == "complex":
            top_k = min(top_k + 5, 20)
            strategy = RetrievalStrategy.HYBRID

        # Adjust for broad queries — use more results
        if cls.specificity == "broad":
            top_k = max(top_k + 3, 10)

        return RoutingDecision(
            strategy=strategy,
            top_k=top_k,
            rerank=rerank,
            classification=cls,
        )
