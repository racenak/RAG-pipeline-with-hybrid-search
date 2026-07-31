"""File-based embedding cache — content-addressed vectors."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache embeddings to disk using content-addressed storage.

    Each unique text is hashed to produce a cache key. The vector is
    stored as a .npy file under cache_dir/{key}.npy.
    """

    def __init__(self, cache_dir: str | Path = ".cache/embeddings") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def has(self, text: str) -> bool:
        """Check if text has a cached embedding."""
        return self._path(self._hash(text)).exists()

    def get(self, text: str) -> list[float] | None:
        """Retrieve cached embedding for text, or None if not cached."""
        path = self._path(self._hash(text))
        if not path.exists():
            return None
        try:
            vector: np.ndarray = np.load(path)
            return vector.tolist()
        except Exception as e:
            logger.warning("Failed to load cache %s: %s", path, e)
            return None

    def set(self, text: str, vector: list[float]) -> None:
        """Store embedding for text in cache."""
        path = self._path(self._hash(text))
        try:
            np.save(path, np.array(vector, dtype=np.float32))
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", path, e)

    def get_many(self, texts: list[str]) -> dict[int, list[float]]:
        """Retrieve cached embeddings for multiple texts.

        Returns dict mapping index → vector for cache hits.
        """
        results: dict[int, list[float]] = {}
        for i, text in enumerate(texts):
            vector = self.get(text)
            if vector is not None:
                results[i] = vector
        return results

    def set_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        """Store multiple embeddings in cache."""
        for text, vector in zip(texts, vectors, strict=True):
            self.set(text, vector)

    def clear(self) -> int:
        """Delete all cached files. Returns number of files deleted."""
        count = 0
        for path in self.cache_dir.glob("*.npy"):
            path.unlink()
            count += 1
        logger.info("Cleared %d cached embeddings", count)
        return count

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return sum(1 for _ in self.cache_dir.glob("*.npy"))
