"""Hybrid search — combines vector + BM25 with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_pipeline.retrieval.vector import SearchResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Reciprocal Rank Fusion
# ------------------------------------------------------------------ #


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ weight_i / (k + rank_i(d))

    Args:
        result_lists: List of ranked result lists from different retrievers.
        k: RRF parameter (default 60). Higher k = less weight on top ranks.
        weights: Optional weights per retriever (default: equal weights).

    Returns:
        Fused and re-ranked list of SearchResults.
    """
    if not result_lists:
        return []

    n_lists = len(result_lists)
    if weights is None:
        weights = [1.0] * n_lists

    if len(weights) != n_lists:
        raise ValueError(f"weights length ({len(weights)}) must match result_lists ({n_lists})")

    # Accumulate RRF scores
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for weight, results in zip(weights, result_lists, strict=True):
        for rank, result in enumerate(results):
            rrf_score = weight / (k + rank + 1)  # rank is 0-indexed, so +1

            if result.id in doc_scores:
                doc_scores[result.id] += rrf_score
            else:
                doc_scores[result.id] = rrf_score
                doc_map[result.id] = result

    # Sort by fused score descending
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        original = doc_map[doc_id]
        fused_results.append(SearchResult(
            id=original.id,
            score=doc_scores[doc_id],
            content=original.content,
            metadata=original.metadata,
        ))

    return fused_results


# ------------------------------------------------------------------ #
#  Score-based fusion
# ------------------------------------------------------------------ #


def score_fusion(
    result_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Fuse results by weighted sum of normalized scores.

    Scores are min-max normalized per retriever before combining.
    """
    if not result_lists:
        return []

    n_lists = len(result_lists)
    if weights is None:
        weights = [1.0] * n_lists

    doc_scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for weight, results in zip(weights, result_lists, strict=True):
        if not results:
            continue

        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0

        for result in results:
            normalized = (result.score - min_s) / range_s
            weighted = weight * normalized

            if result.id in doc_scores:
                doc_scores[result.id] += weighted
            else:
                doc_scores[result.id] = weighted
                doc_map[result.id] = result

    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    return [
        SearchResult(
            id=doc_id,
            score=doc_scores[doc_id],
            content=doc_map[doc_id].content,
            metadata=doc_map[doc_id].metadata,
        )
        for doc_id in sorted_ids
    ]


# ------------------------------------------------------------------ #
#  Hybrid search orchestrator
# ------------------------------------------------------------------ #


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    vector_top_k: int = 20
    bm25_top_k: int = 20
    fusion_strategy: str = "rrf"  # "rrf" or "score"
    vector_enabled: bool = True
    bm25_enabled: bool = True


class HybridSearch:
    """Hybrid search orchestrator.

    Combines vector (dense) and BM25 (sparse) search results
    using Reciprocal Rank Fusion or score-based fusion.
    """

    def __init__(self, config: HybridSearchConfig | None = None) -> None:
        self.config = config or HybridSearchConfig()

    def fuse(
        self,
        vector_results: list[SearchResult] | None = None,
        bm25_results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        """Fuse results from vector and BM25 search.

        Args:
            vector_results: Results from dense vector search.
            bm25_results: Results from BM25 text search.

        Returns:
            Fused and re-ranked results.
        """
        active_lists: list[list[SearchResult]] = []
        active_weights: list[float] = []

        if self.config.vector_enabled and vector_results:
            active_lists.append(vector_results)
            active_weights.append(self.config.vector_weight)

        if self.config.bm25_enabled and bm25_results:
            active_lists.append(bm25_results)
            active_weights.append(self.config.bm25_weight)

        if not active_lists:
            return []

        if len(active_lists) == 1:
            return active_lists[0]

        if self.config.fusion_strategy == "rrf":
            return reciprocal_rank_fusion(
                active_lists,
                k=self.config.rrf_k,
                weights=active_weights,
            )
        if self.config.fusion_strategy == "score":
            return score_fusion(active_lists, weights=active_weights)
        raise ValueError(f"Unknown fusion strategy: {self.config.fusion_strategy}")
