"""Unified search interface — routes to vector, BM25, or hybrid search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rag_pipeline.retrieval.hybrid import HybridSearch, HybridSearchConfig
from rag_pipeline.retrieval.vector import SearchResult

if TYPE_CHECKING:
    from rag_pipeline.retrieval.bm25 import BM25
    from rag_pipeline.retrieval.bm25_index import OpenSearchBM25
    from rag_pipeline.retrieval.vector import VectorSearch

logger = logging.getLogger(__name__)


class SearchEngine:
    """Unified search interface.

    Supports vector-only, BM25-only, or hybrid search modes.
    """

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        bm25_search: BM25 | OpenSearchBM25 | None = None,
        hybrid_config: HybridSearchConfig | None = None,
    ) -> None:
        self._vector = vector_search
        self._bm25 = bm25_search
        self._hybrid = HybridSearch(hybrid_config)

    def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        top_k: int = 20,
        mode: str = "hybrid",
        metadata_filter: dict[str, Any] | None = None,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search using the specified mode.

        Args:
            query: Search query text (for BM25).
            query_vector: Query embedding (for vector search).
            top_k: Number of results to return.
            mode: "hybrid", "vector", or "bm25".
            metadata_filter: Optional metadata filter.
            threshold: Minimum similarity score (vector only).

        Returns:
            Ranked list of SearchResults.
        """
        if mode == "vector":
            return self._vector_search(query_vector, top_k, threshold, metadata_filter)
        if mode == "bm25":
            return self._bm25_search(query, top_k, metadata_filter)
        if mode == "hybrid":
            return self._hybrid_search(
                query, query_vector, top_k, threshold, metadata_filter
            )
        raise ValueError(f"Unknown search mode: {mode}")

    def _vector_search(
        self,
        query_vector: list[float] | None,
        top_k: int,
        threshold: float,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if self._vector is None:
            logger.warning("Vector search requested but no vector backend configured")
            return []
        if query_vector is None:
            logger.warning("Vector search requested but no query vector provided")
            return []
        return self._vector.search(
            query_vector, top_k=top_k, threshold=threshold,
            metadata_filter=metadata_filter,
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if self._bm25 is None:
            logger.warning("BM25 search requested but no BM25 backend configured")
            return []

        raw_results = self._bm25.search(query, top_k=top_k, metadata_filter=metadata_filter)

        # Convert to SearchResult
        return [
            SearchResult(
                id=r["id"],
                score=r["score"],
                content=r.get("content", ""),
                metadata=r.get("metadata"),
            )
            for r in raw_results
        ]

    def _hybrid_search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int,
        threshold: float,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        # Gather results from both backends
        vector_results: list[SearchResult] | None = None
        bm25_results: list[SearchResult] | None = None

        if self._vector is not None and query_vector is not None:
            vector_results = self._vector.search(
                query_vector,
                top_k=self._hybrid.config.vector_top_k,
                threshold=threshold,
                metadata_filter=metadata_filter,
            )

        if self._bm25 is not None:
            raw_bm25 = self._bm25.search(
                query,
                top_k=self._hybrid.config.bm25_top_k,
                metadata_filter=metadata_filter,
            )
            bm25_results = [
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    content=r.get("content", ""),
                    metadata=r.get("metadata"),
                )
                for r in raw_bm25
            ]

        fused = self._hybrid.fuse(vector_results, bm25_results)
        return fused[:top_k]
