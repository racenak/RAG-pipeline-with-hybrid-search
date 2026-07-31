"""Embedding generator — main entry point for text → vector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_pipeline.embeddings.cache import EmbeddingCache

if TYPE_CHECKING:
    from rag_pipeline.data.chunking import Chunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using a configurable backend.

    Supports sentence-transformers (local) with optional file-based caching.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 64,
        cache_enabled: bool = True,
        cache_dir: str = ".cache/embeddings",
    ) -> None:
        from rag_pipeline.embeddings.backends.sentence_transformers import (
            SentenceTransformerBackend,
        )

        self._backend = SentenceTransformerBackend(
            model_name=model_name,
            device=device,
            normalize=normalize,
        )
        self._batch_size = batch_size
        self._cache_enabled = cache_enabled
        self._cache = EmbeddingCache(cache_dir) if cache_enabled else None

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, using cache where possible."""
        if not texts:
            return []

        # Check cache
        vectors: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self._cache:
            cached = self._cache.get_many(texts)
            for idx, vector in cached.items():
                vectors[idx] = vector
            uncached_indices = [i for i, v in enumerate(vectors) if v is None]
            uncached_texts = [texts[i] for i in uncached_indices]
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Compute missing embeddings
        if uncached_texts:
            logger.debug(
                "Computing %d embeddings (batch_size=%d)", len(uncached_texts), self._batch_size
            )
            new_vectors = self._backend.encode(
                uncached_texts,
                batch_size=self._batch_size,
            )
            for idx, vector in zip(uncached_indices, new_vectors, strict=True):
                vectors[idx] = vector

            # Store in cache
            if self._cache:
                self._cache.set_many(uncached_texts, new_vectors)

        return [v for v in vectors if v is not None]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        result = self.embed([text])
        return result[0] if result else []

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed chunks and attach vectors to them.

        Returns the same chunk objects with .embedding populated.
        """
        texts = [chunk.content for chunk in chunks]
        vectors = self.embed(texts)

        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector  # type: ignore[attr-defined]

        return chunks

    def clear_cache(self) -> int:
        """Clear the embedding cache."""
        if self._cache:
            return self._cache.clear()
        return 0

    @property
    def cache_size(self) -> int:
        if self._cache:
            return self._cache.size
        return 0
