"""Sentence-transformers embedding backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerBackend:
    """Embedding backend using sentence-transformers (HuggingFace)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded model %s (dim=%d, device=%s)", model_name, self._dimension, device)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Encode a list of texts into embedding vectors."""
        if not texts:
            return []

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings.tolist()
