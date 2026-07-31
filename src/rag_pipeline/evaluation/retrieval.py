"""Standard information retrieval evaluation metrics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """Metrics for a single query or aggregated across queries."""

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    map_score: float = 0.0


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    query_id: str
    query: str
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    metrics: RetrievalMetrics


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: Set of relevant document IDs.
        k: Cutoff depth.

    Returns:
        Precision@k score in [0, 1].
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    expected_set = set(expected)
    relevant_count = sum(1 for doc_id in top_k if doc_id in expected_set)
    return relevant_count / len(top_k)


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected docs found in top-k retrieved.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: List of relevant document IDs.
        k: Cutoff depth.

    Returns:
        Recall@k score in [0, 1].
    """
    if k <= 0 or not expected:
        return 0.0
    top_k = retrieved[:k]
    expected_set = set(expected)
    found = sum(1 for doc_id in expected_set if doc_id in top_k)
    return found / len(expected_set)


def mean_reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1/rank of first relevant result.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: List of relevant document IDs.

    Returns:
        MRR score in [0, 1]. 0.0 if no relevant doc found.
    """
    expected_set = set(expected)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k with binary relevance.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: List of relevant document IDs.
        k: Cutoff depth.

    Returns:
        NDCG@k score in [0, 1].
    """
    if k <= 0 or not expected:
        return 0.0
    expected_set = set(expected)
    top_k = retrieved[:k]

    # DCG: sum of rel_i / log2(i+1) where i is 1-indexed
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in expected_set else 0.0
        dcg += rel / math.log2(i + 1)

    # IDCG: ideal DCG (all relevant docs at the top)
    ideal_relevant = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved: list[str], expected: list[str]) -> float:
    """1.0 if any retrieved doc is in expected, 0.0 otherwise.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: List of relevant document IDs.

    Returns:
        Hit rate: 1.0 or 0.0.
    """
    expected_set = set(expected)
    for doc_id in retrieved:
        if doc_id in expected_set:
            return 1.0
    return 0.0


def mean_average_precision(retrieved: list[str], expected: list[str]) -> float:
    """Average precision = sum(precision@i * rel_i) / |expected|.

    Args:
        retrieved: Ranked list of retrieved document IDs.
        expected: List of relevant document IDs.

    Returns:
        MAP score in [0, 1].
    """
    if not expected:
        return 0.0
    expected_set = set(expected)
    relevant_count = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected_set:
            relevant_count += 1
            precision_sum += relevant_count / i
    return precision_sum / len(expected_set)


def evaluate_query(
    query_id: str,
    query: str,
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
    k_values: list[int] | None = None,
) -> QueryResult:
    """Compute all metrics for a single query.

    Args:
        query_id: Unique query identifier.
        query: Query text.
        retrieved_doc_ids: Ranked list of retrieved document IDs.
        expected_doc_ids: List of relevant document IDs.
        k_values: List of k values for cutoff metrics.

    Returns:
        QueryResult with all computed metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = RetrievalMetrics(
        precision_at_k={
            k: precision_at_k(retrieved_doc_ids, expected_doc_ids, k) for k in k_values
        },
        recall_at_k={
            k: recall_at_k(retrieved_doc_ids, expected_doc_ids, k) for k in k_values
        },
        mrr=mean_reciprocal_rank(retrieved_doc_ids, expected_doc_ids),
        ndcg_at_k={
            k: ndcg_at_k(retrieved_doc_ids, expected_doc_ids, k) for k in k_values
        },
        hit_rate=hit_rate(retrieved_doc_ids, expected_doc_ids),
        map_score=mean_average_precision(retrieved_doc_ids, expected_doc_ids),
    )

    return QueryResult(
        query_id=query_id,
        query=query,
        retrieved_doc_ids=retrieved_doc_ids,
        expected_doc_ids=expected_doc_ids,
        metrics=metrics,
    )


def evaluate_dataset(query_results: list[QueryResult]) -> RetrievalMetrics:
    """Aggregate metrics: mean of per-query metrics. Also compute median and p95.

    Note: This function computes mean values. Median/p95 are available via
    separate aggregation if needed.

    Args:
        query_results: List of per-query evaluation results.

    Returns:
        Aggregated RetrievalMetrics with mean values across all queries.
    """
    if not query_results:
        return RetrievalMetrics()

    n = len(query_results)

    # Aggregate scalar metrics
    mrr_sum = sum(qr.metrics.mrr for qr in query_results)
    hit_rate_sum = sum(qr.metrics.hit_rate for qr in query_results)
    map_sum = sum(qr.metrics.map_score for qr in query_results)

    # Aggregate dict metrics — find union of all k values
    all_k_values: set[int] = set()
    for qr in query_results:
        all_k_values.update(qr.metrics.precision_at_k.keys())
        all_k_values.update(qr.metrics.recall_at_k.keys())
        all_k_values.update(qr.metrics.ndcg_at_k.keys())

    precision_means: dict[int, float] = {}
    recall_means: dict[int, float] = {}
    ndcg_means: dict[int, float] = {}

    for k in sorted(all_k_values):
        p_vals = [
            qr.metrics.precision_at_k.get(k, 0.0) for qr in query_results
        ]
        r_vals = [
            qr.metrics.recall_at_k.get(k, 0.0) for qr in query_results
        ]
        n_vals = [
            qr.metrics.ndcg_at_k.get(k, 0.0) for qr in query_results
        ]
        precision_means[k] = sum(p_vals) / n
        recall_means[k] = sum(r_vals) / n
        ndcg_means[k] = sum(n_vals) / n

    return RetrievalMetrics(
        precision_at_k=precision_means,
        recall_at_k=recall_means,
        mrr=mrr_sum / n,
        ndcg_at_k=ndcg_means,
        hit_rate=hit_rate_sum / n,
        map_score=map_sum / n,
    )


def compute_percentiles(
    query_results: list[QueryResult],
    k_values: list[int] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute median and p95 for each metric across queries.

    Returns:
        Dict of metric_name -> {k: {"median": val, "p95": val}} for dict metrics,
        and metric_name -> {"median": val, "p95": val} for scalar metrics.
    """
    if not query_results:
        return {}

    if k_values is None:
        k_values = [1, 3, 5, 10]

    result: dict[str, dict] = {}

    # Scalar metrics
    mrr_vals = [qr.metrics.mrr for qr in query_results]
    hit_vals = [qr.metrics.hit_rate for qr in query_results]
    map_vals = [qr.metrics.map_score for qr in query_results]

    result["mrr"] = {
        "median": statistics.median(mrr_vals),
        "p95": _percentile(mrr_vals, 95),
    }
    result["hit_rate"] = {
        "median": statistics.median(hit_vals),
        "p95": _percentile(hit_vals, 95),
    }
    result["map_score"] = {
        "median": statistics.median(map_vals),
        "p95": _percentile(map_vals, 95),
    }

    # Dict metrics
    for k in k_values:
        p_vals = [qr.metrics.precision_at_k.get(k, 0.0) for qr in query_results]
        r_vals = [qr.metrics.recall_at_k.get(k, 0.0) for qr in query_results]
        n_vals = [qr.metrics.ndcg_at_k.get(k, 0.0) for qr in query_results]

        if "precision_at_k" not in result:
            result["precision_at_k"] = {}
        if "recall_at_k" not in result:
            result["recall_at_k"] = {}
        if "ndcg_at_k" not in result:
            result["ndcg_at_k"] = {}

        result["precision_at_k"][k] = {
            "median": statistics.median(p_vals),
            "p95": _percentile(p_vals, 95),
        }
        result["recall_at_k"][k] = {
            "median": statistics.median(r_vals),
            "p95": _percentile(r_vals, 95),
        }
        result["ndcg_at_k"][k] = {
            "median": statistics.median(n_vals),
            "p95": _percentile(n_vals, 95),
        }

    return result


def _percentile(values: list[float], percentile: float) -> float:
    """Compute the given percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (percentile / 100.0) * (len(sorted_vals) - 1)
    lower = int(math.floor(k))
    upper = int(math.ceil(k))
    if lower == upper:
        return sorted_vals[lower]
    fraction = k - lower
    return sorted_vals[lower] * (1 - fraction) + sorted_vals[upper] * fraction
