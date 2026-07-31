"""Advanced retrieval — LLM-powered query rewriting and metadata filtering.

Strategies that use an optional LLM backend to generate richer queries.
When no backend is provided, deterministic fallback methods are used.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rag_pipeline.generation.llm import LLMBackend
from rag_pipeline.retrieval.vector import SearchResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #


@dataclass
class AdvancedQueryConfig:
    """Toggles for each advanced retrieval strategy + metadata filter options."""

    expand_enabled: bool = True
    hyde_enabled: bool = True
    multi_query_enabled: bool = True
    step_back_enabled: bool = True
    metadata_filter_enabled: bool = False
    multi_query_count: int = 4
    date_from: str | None = None
    date_to: str | None = None
    document_types: list[str] = field(default_factory=list)
    custom_filters: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ #
#  LLM Query Expander
# ------------------------------------------------------------------ #

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very", "just",
})


class LLMQueryExpander:
    """Expand queries with related terms using LLM or deterministic fallback."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    def _fallback_expand(self, query: str) -> list[str]:
        words = query.split()
        expanded: list[str] = []
        for word in words:
            clean = word.strip(".,;:!?").lower()
            if clean and clean not in _STOP_WORDS and len(clean) > 2:
                expanded.append(clean)
        if expanded:
            return [f"{query} {' '.join(expanded[:3])}"]
        return [query]

    async def expand(self, query: str) -> list[str]:
        if self._backend is None:
            return self._fallback_expand(query)

        try:
            messages = [
                {"role": "system", "content": (
                    "Generate 2-3 related search terms for the following query. "
                    "Return only the terms separated by commas, no explanation."
                )},
                {"role": "user", "content": query},
            ]
            response = await self._backend.generate(messages)
            terms = [t.strip() for t in response.split(",") if t.strip()]
            if terms:
                return [f"{query} {' '.join(terms[:3])}"]
            return self._fallback_expand(query)
        except Exception:
            logger.warning("LLM query expansion failed, using fallback")
            return self._fallback_expand(query)


# ------------------------------------------------------------------ #
#  LLM HyDE Generator
# ------------------------------------------------------------------ #


class LLMHydeGenerator:
    """Generate hypothetical documents using LLM or deterministic fallback."""

    _PREFIX = "This document provides a detailed answer about: "

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    def _fallback_hyde(self, query: str) -> str:
        return f"{self._PREFIX}{query}"

    async def generate(self, query: str) -> str:
        if self._backend is None:
            return self._fallback_hyde(query)

        try:
            messages = [
                {"role": "system", "content": (
                    "Write a short hypothetical document (2-3 sentences) that would "
                    "contain the answer to the following question. "
                    "Return only the document text."
                )},
                {"role": "user", "content": query},
            ]
            response = await self._backend.generate(messages)
            return response.strip() if response.strip() else self._fallback_hyde(query)
        except Exception:
            logger.warning("LLM HyDE generation failed, using fallback")
            return self._fallback_hyde(query)


# ------------------------------------------------------------------ #
#  LLM Multi-Query Generator
# ------------------------------------------------------------------ #


class LLMMultiQueryGenerator:
    """Generate paraphrased query variations using LLM or templates."""

    _TEMPLATES = [
        "What is {q}?",
        "How does {q} work?",
        "Explain {q} in detail.",
        "Tell me about {q}.",
        "What should I know about {q}?",
        "Give me an overview of {q}.",
    ]

    def __init__(
        self, backend: LLMBackend | None = None, num_queries: int = 4
    ) -> None:
        self._backend = backend
        self._num_queries = max(2, min(num_queries, 6))

    def _fallback_multi(self, query: str) -> list[str]:
        clean_q = query.rstrip("?.!").strip()
        queries = [query]
        for tmpl in self._TEMPLATES[: self._num_queries - 1]:
            queries.append(tmpl.format(q=clean_q))
        return queries

    async def generate(self, query: str) -> list[str]:
        if self._backend is None:
            return self._fallback_multi(query)

        try:
            messages = [
                {"role": "system", "content": (
                    f"Generate {self._num_queries - 1} paraphrased versions of the "
                    "following search query. Return only the paraphrases separated "
                    "by newlines, no numbering or explanation."
                )},
                {"role": "user", "content": query},
            ]
            response = await self._backend.generate(messages)
            variations = [v.strip() for v in response.split("\n") if v.strip()]
            if variations:
                return [query] + variations[: self._num_queries - 1]
            return self._fallback_multi(query)
        except Exception:
            logger.warning("LLM multi-query generation failed, using fallback")
            return self._fallback_multi(query)


# ------------------------------------------------------------------ #
#  LLM Step-Back Generator
# ------------------------------------------------------------------ #


class LLMStepBackGenerator:
    """Generate broader step-back queries using LLM or stop-word extraction."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    def _fallback_step_back(self, query: str) -> str:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query)
        key_terms = [w for w in words if w.lower() not in _STOP_WORDS]
        if not key_terms:
            return "General information and context about the topic"
        broad_query = " ".join(key_terms[:5])
        return f"Overview and background information about {broad_query}"

    async def generate(self, query: str) -> str:
        if self._backend is None:
            return self._fallback_step_back(query)

        try:
            messages = [
                {"role": "system", "content": (
                    "Rewrite the following query as a broader, more general question "
                    "that would retrieve background information. Return only the "
                    "rewritten query."
                )},
                {"role": "user", "content": query},
            ]
            response = await self._backend.generate(messages)
            return response.strip() if response.strip() else self._fallback_step_back(query)
        except Exception:
            logger.warning("LLM step-back generation failed, using fallback")
            return self._fallback_step_back(query)


# ------------------------------------------------------------------ #
#  Metadata Filter
# ------------------------------------------------------------------ #


class MetadataFilter:
    """Filter search results by date range, document type, and custom key=value pairs."""

    def __init__(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        document_types: list[str] | None = None,
        custom_filters: dict[str, str] | None = None,
    ) -> None:
        self._date_from = self._parse_date(date_from) if date_from else None
        self._date_to = self._parse_date(date_to) if date_to else None
        self._document_types = set(document_types) if document_types else set()
        self._custom_filters = custom_filters or {}

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unparseable date format: {date_str}")

    def _matches_date(self, result: SearchResult) -> bool:
        if self._date_from is None and self._date_to is None:
            return True
        meta = result.metadata or {}
        date_val = meta.get("date") or meta.get("created_at") or meta.get("timestamp")
        if date_val is None:
            return False
        if isinstance(date_val, str):
            try:
                result_date = MetadataFilter._parse_date(date_val)
            except ValueError:
                return False
        elif isinstance(date_val, datetime):
            result_date = date_val.replace(tzinfo=timezone.utc) if date_val.tzinfo is None else date_val
        else:
            return False
        if self._date_from and result_date < self._date_from:
            return False
        if self._date_to and result_date > self._date_to:
            return False
        return True

    def _matches_type(self, result: SearchResult) -> bool:
        if not self._document_types:
            return True
        meta = result.metadata or {}
        doc_type = meta.get("type") or meta.get("document_type", "")
        return str(doc_type) in self._document_types

    def _matches_custom(self, result: SearchResult) -> bool:
        if not self._custom_filters:
            return True
        meta = result.metadata or {}
        for key, expected in self._custom_filters.items():
            actual = meta.get(key)
            if actual is None or str(actual) != expected:
                return False
        return True

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        filtered: list[SearchResult] = []
        for r in results:
            if self._matches_date(r) and self._matches_type(r) and self._matches_custom(r):
                filtered.append(r)
        return filtered


# ------------------------------------------------------------------ #
#  Advanced Retrieval Pipeline
# ------------------------------------------------------------------ #


@dataclass
class RetrievalMetrics:
    """Latency and strategy tracking for advanced retrieval."""

    strategies_used: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    strategy_latencies: dict[str, float] = field(default_factory=dict)


class AdvancedRetrievalPipeline:
    """Orchestrate advanced retrieval strategies with async processing."""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        config: AdvancedQueryConfig | None = None,
    ) -> None:
        self._config = config or AdvancedQueryConfig()
        self._expander = LLMQueryExpander(backend)
        self._hyde = LLMHydeGenerator(backend)
        self._multi_query = LLMMultiQueryGenerator(
            backend, self._config.multi_query_count
        )
        self._step_back = LLMStepBackGenerator(backend)
        self._metadata_filter = MetadataFilter(
            date_from=self._config.date_from,
            date_to=self._config.date_to,
            document_types=self._config.document_types,
            custom_filters=self._config.custom_filters,
        )

    async def expand(self, query: str) -> list[str]:
        start = time.perf_counter()
        result = await self._expander.expand(query)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("LLMQueryExpander: %.1fms", elapsed)
        return result

    async def hyde(self, query: str) -> str:
        start = time.perf_counter()
        result = await self._hyde.generate(query)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("LLMHydeGenerator: %.1fms", elapsed)
        return result

    async def multi_query(self, query: str) -> list[str]:
        start = time.perf_counter()
        result = await self._multi_query.generate(query)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("LLMMultiQueryGenerator: %.1fms", elapsed)
        return result

    async def step_back(self, query: str) -> str:
        start = time.perf_counter()
        result = await self._step_back.generate(query)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("LLMStepBackGenerator: %.1fms", elapsed)
        return result

    def filter_results(self, results: list[SearchResult]) -> list[SearchResult]:
        return self._metadata_filter.filter(results)

    async def process(
        self, query: str, results: list[SearchResult] | None = None
    ) -> dict[str, Any]:
        """Run all enabled strategies and return combined output.

        Args:
            query: The user query.
            results: Optional search results to filter with metadata.

        Returns:
            Dict with expanded_queries, hyde_query, multi_queries,
            step_back_query, filtered_results, and metrics.
        """
        start = time.perf_counter()
        metrics = RetrievalMetrics()
        cfg = self._config

        expanded_queries: list[str] = []
        hyde_query: str = ""
        multi_queries: list[str] = []
        step_back_query: str = ""
        filtered_results: list[SearchResult] = []

        tasks: dict[str, Any] = {}
        if cfg.expand_enabled:
            tasks["expand"] = self.expand(query)
        if cfg.hyde_enabled:
            tasks["hyde"] = self.hyde(query)
        if cfg.multi_query_enabled:
            tasks["multi_query"] = self.multi_query(query)
        if cfg.step_back_enabled:
            tasks["step_back"] = self.step_back(query)

        if tasks:
            keys = list(tasks.keys())
            coros = [tasks[k] for k in keys]
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            for key, result in zip(keys, gathered, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Strategy %s failed: %s", key, result)
                else:
                    metrics.strategies_used.append(key)
                    if key == "expand":
                        expanded_queries = result
                    elif key == "hyde":
                        hyde_query = result
                    elif key == "multi_query":
                        multi_queries = result
                    elif key == "step_back":
                        step_back_query = result

        if cfg.metadata_filter_enabled and results is not None:
            filter_start = time.perf_counter()
            filtered_results = self.filter_results(results)
            filter_elapsed = (time.perf_counter() - filter_start) * 1000
            metrics.strategies_used.append("metadata_filter")
            metrics.strategy_latencies["metadata_filter"] = filter_elapsed

        metrics.total_latency_ms = (time.perf_counter() - start) * 1000

        return {
            "expanded_queries": expanded_queries,
            "hyde_query": hyde_query,
            "multi_queries": multi_queries,
            "step_back_query": step_back_query,
            "filtered_results": filtered_results,
            "metrics": metrics,
        }
