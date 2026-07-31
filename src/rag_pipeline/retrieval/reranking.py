"""Reranking — Cross-encoder reranking for retrieval results."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rag_pipeline.retrieval.vector import SearchResult

if TYPE_CHECKING:
    from rag_pipeline.config import RetrievalSettings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Abstract interface
# ------------------------------------------------------------------ #


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Rerank results using the cross-encoder model.

        Args:
            query: The search query.
            results: Candidate results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of SearchResults, truncated to top_k.
        """


# ------------------------------------------------------------------ #
#  Cross-encoder reranker
# ------------------------------------------------------------------ #


class CrossEncoderReranker(Reranker):
    """Reranker using a sentence-transformers CrossEncoder model.

    Scores each (query, content) pair and returns results sorted
    by cross-encoder score in descending order.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _load_model(self) -> bool:
        """Lazy-load the cross-encoder model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("Loaded cross-encoder model: %s", self._model_name)
            return True
        except Exception:
            logger.exception(
                "Failed to load cross-encoder model %s", self._model_name
            )
            return False

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Rerank results using the cross-encoder model.

        Args:
            query: The search query.
            results: Candidate results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of SearchResults, truncated to top_k.
            Falls back to original order if model is unavailable.
        """
        if not results:
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided to reranker, returning original order")
            return results[:top_k]

        if not self._load_model():
            logger.warning(
                "Cross-encoder model unavailable, returning original order"
            )
            return results[:top_k]

        assert self._model is not None  # for type checker

        # Build (query, content) pairs for batch scoring
        pairs = [(query, result.content) for result in results]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("Cross-encoder scoring failed, returning original order")
            return results[:top_k]

        # Pair results with their cross-encoder scores and sort descending
        scored_results = list(zip(results, scores, strict=True))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results with updated scores
        reranked: list[SearchResult] = []
        for result, score in scored_results[:top_k]:
            reranked.append(SearchResult(
                id=result.id,
                score=float(score),
                content=result.content,
                metadata=result.metadata,
            ))

        return reranked


# ------------------------------------------------------------------ #
#  No-op reranker (fallback)
# ------------------------------------------------------------------ #


class NoopReranker(Reranker):
    """Reranker that returns results unchanged.

    Used as a fallback when cross-encoder is unavailable or disabled.
    """

    def rerank(
        self,
        query: str,  # noqa: ARG002
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Return results truncated to top_k without reranking."""
        return results[:top_k]


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #


def get_reranker(settings: RetrievalSettings | None = None) -> Reranker:
    """Create a reranker based on configuration.

    Args:
        settings: Retrieval settings. Uses defaults if None.

    Returns:
        CrossEncoderReranker if enabled, NoopReranker otherwise.
    """
    from rag_pipeline.config import RetrievalSettings

    if settings is None:
        settings = RetrievalSettings()

    if not settings.rerank_enabled:
        logger.info("Reranking disabled, using NoopReranker")
        return NoopReranker()

    return CrossEncoderReranker(
        model_name=settings.rerank_model,
    )
