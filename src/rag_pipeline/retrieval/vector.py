"""Dense vector search — OpenSearch kNN and FAISS in-memory backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rag_pipeline.storage.opensearch import OpenSearchClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Search result model
# ------------------------------------------------------------------ #


@dataclass
class SearchResult:
    """A single search result with score and content."""

    id: str
    score: float
    content: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "score": self.score, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ------------------------------------------------------------------ #
#  Abstract interface
# ------------------------------------------------------------------ #


class VectorSearch(ABC):
    """Abstract vector search backend."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbors."""

    @abstractmethod
    def index_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Index vectors. Returns count indexed."""


# ------------------------------------------------------------------ #
#  OpenSearch kNN backend
# ------------------------------------------------------------------ #


class OpenSearchVectorSearch(VectorSearch):
    """Vector search using OpenSearch kNN."""

    def __init__(
        self,
        client: OpenSearchClient,
        index_name: str,
    ) -> None:
        self._client = client
        self._index_name = index_name

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k,
                    }
                }
            },
        }

        if metadata_filter:
            body["query"]["knn"]["embedding"]["filter"] = {
                "bool": {
                    "filter": [
                        {"term": {k: v}} for k, v in metadata_filter.items()
                    ]
                }
            }

        response = self._client._client.search(  # noqa: SLF001
            index=self._index_name, body=body
        )
        hits = response.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            score = hit["_score"]
            if score < threshold:
                continue
            source = hit["_source"]
            results.append(SearchResult(
                id=hit["_id"],
                score=score,
                content=source.get("content", ""),
                metadata=source.get("metadata"),
            ))
        return results

    def index_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Bulk index via OpenSearch bulk API."""
        from rag_pipeline.data.chunking import Chunk

        chunks = []
        for i, (cid, vec, content) in enumerate(zip(ids, vectors, contents, strict=True)):
            chunks.append(Chunk(
                id=cid,
                document_id="",
                content=content,
                index=i,
                token_count=0,
                embedding=vec,
                metadata=metadatas[i] if metadatas else {},
            ))
        return self._client.index_chunks(self._index_name, chunks)


# ------------------------------------------------------------------ #
#  FAISS in-memory backend (dev/testing)
# ------------------------------------------------------------------ #


class FAISSVectorSearch(VectorSearch):
    """In-memory vector search using FAISS.

    Useful for dev/testing without OpenSearch.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)

        # Cosine similarity (vectors should be L2-normalized)
        scores = (self._vectors @ query.T).flatten()

        # Apply metadata filter (post-filter)
        if metadata_filter:
            for i, meta in enumerate(self._metadatas):
                if meta is None:
                    scores[i] = -np.inf
                    continue
                for k, v in metadata_filter.items():
                    if meta.get(k) != v:
                        scores[i] = -np.inf
                        break

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            results.append(SearchResult(
                id=self._ids[idx],
                score=score,
                content=self._contents[idx],
                metadata=self._metadatas[idx],
            ))
        return results

    def index_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not ids:
            return 0

        self._ids.extend(ids)
        self._contents.extend(contents)
        self._metadatas.extend(metadatas or [{}] * len(ids))

        new_vectors = np.array(vectors, dtype=np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_vectors = new_vectors / np.maximum(norms, 1e-10)

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        return len(ids)

    def clear(self) -> None:
        """Clear all indexed vectors."""
        self._ids.clear()
        self._contents.clear()
        self._metadatas.clear()
        self._vectors = None

    @property
    def doc_count(self) -> int:
        return len(self._ids)
