"""Evaluation pipeline orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from rag_pipeline.evaluation.retrieval import (
    QueryResult,
    RetrievalMetrics,
    evaluate_dataset,
    evaluate_query,
)

if TYPE_CHECKING:
    from evaluation.dataset_schema import EvalCase


@dataclass
class EvaluationReport:
    """Full evaluation report with aggregated and per-query results."""

    total_queries: int
    metrics: RetrievalMetrics
    per_query: list[QueryResult]
    category_breakdown: dict[str, RetrievalMetrics]
    latency_ms: float


class RetrievalEvaluator:
    """Orchestrate retrieval evaluation across a dataset."""

    def __init__(self, search_engine: Any | None = None) -> None:
        """Initialize evaluator.

        Args:
            search_engine: Optional SearchEngine instance for integration tests.
        """
        self._search_engine = search_engine

    def evaluate(
        self,
        eval_cases: list[Any],
        search_fn: Callable[[str, int], list[str]] | None = None,
        k_values: list[int] | None = None,
    ) -> EvaluationReport:
        """Run evaluation on all cases.

        Args:
            eval_cases: List of EvalCase from dataset_schema.
            search_fn: Callable(query, top_k) -> list of doc IDs.
            k_values: Cutoff values for metrics.

        Returns:
            EvaluationReport with all results.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        if search_fn is None:
            search_fn = self._default_search_fn

        start_time = time.perf_counter()
        per_query: list[QueryResult] = []

        for case in eval_cases:
            max_k = max(k_values) if k_values else 10
            retrieved_doc_ids = search_fn(case.query, max_k)
            result = evaluate_query(
                query_id=case.id,
                query=case.query,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_doc_ids=case.expected_documents,
                k_values=k_values,
            )
            per_query.append(result)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return self.generate_report(per_query, latency_ms=latency_ms)

    def generate_report(
        self,
        results: list[QueryResult],
        latency_ms: float = 0.0,
    ) -> EvaluationReport:
        """Aggregate results into a report.

        Args:
            results: List of per-query results.
            latency_ms: Total evaluation latency in milliseconds.

        Returns:
            EvaluationReport with aggregated metrics and category breakdown.
        """
        metrics = evaluate_dataset(results)

        # Category breakdown — group by query prefix (first word) as a simple heuristic
        category_groups: dict[str, list[QueryResult]] = {}
        for qr in results:
            # Use a category key from the query if available, else "all"
            category = _derive_category(qr.query)
            category_groups.setdefault(category, []).append(qr)

        category_breakdown: dict[str, RetrievalMetrics] = {}
        for cat, cat_results in category_groups.items():
            category_breakdown[cat] = evaluate_dataset(cat_results)

        return EvaluationReport(
            total_queries=len(results),
            metrics=metrics,
            per_query=results,
            category_breakdown=category_breakdown,
            latency_ms=latency_ms,
        )

    def _default_search_fn(self, query: str, top_k: int) -> list[str]:
        """Fallback search function when no search_fn or search_engine is provided."""
        return []


def _derive_category(query: str) -> str:
    """Derive a simple category label from query text."""
    words = query.strip().split()
    if not words:
        return "unknown"
    first_word = words[0].lower()
    # Simple heuristic: if query starts with a question word
    question_words = {"what", "who", "where", "when", "why", "how", "which"}
    if first_word in question_words and len(words) > 1:
        return f"q_{first_word}"
    return "other"
