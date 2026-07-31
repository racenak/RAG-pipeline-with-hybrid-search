"""OpenSearch client — indexing and retrieval for vectors + text."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError, RequestError

if TYPE_CHECKING:
    from rag_pipeline.data.chunking import Chunk

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client for RAG indexing and search."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        retry_on_timeout: bool = True,
    ) -> None:
        kwargs: dict[str, Any] = {
            "hosts": [{"host": host, "port": port}],
            "scheme": scheme,
            "connection_class": RequestsHttpConnection,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_on_timeout": retry_on_timeout,
        }
        if username and password:
            kwargs["http_auth"] = (username, password)

        self._client = OpenSearch(**kwargs)
        self._host = host
        self._port = port

    def health_check(self) -> bool:
        """Check cluster health. Returns True if healthy."""
        try:
            health = self._client.cluster.health()
            return health["status"] in ("green", "yellow")
        except Exception as e:
            logger.warning("OpenSearch health check failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    #  Index management
    # ------------------------------------------------------------------ #

    def create_index(
        self,
        index_name: str,
        dimension: int = 1024,
    ) -> None:
        """Create an index with knn_vector + text mapping."""
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                        },
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "chunk_index": {"type": "integer"},
                    "created_at": {"type": "date"},
                }
            },
        }

        try:
            self._client.indices.create(index=index_name, body=body)
            logger.info("Created index: %s", index_name)
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.debug("Index %s already exists", index_name)
            else:
                raise

    def delete_index(self, index_name: str) -> None:
        """Delete an index if it exists."""
        try:
            self._client.indices.delete(index=index_name)
            logger.info("Deleted index: %s", index_name)
        except NotFoundError:
            logger.debug("Index %s does not exist", index_name)

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        return self._client.indices.exists(index=index_name)

    def alias_swap(
        self,
        alias: str,
        new_index: str,
        old_index: str | None = None,
    ) -> None:
        """Atomically swap alias from old_index to new_index."""
        actions: list[dict[str, Any]] = [{"add": {"index": new_index, "alias": alias}}]
        if old_index:
            actions.append({"remove": {"index": old_index, "alias": alias}})
        self._client.indices.update_aliases(body={"actions": actions})
        logger.info("Alias %s → %s (removed %s)", alias, new_index, old_index)

    # ------------------------------------------------------------------ #
    #  Document operations
    # ------------------------------------------------------------------ #

    def index_chunks(self, index_name: str, chunks: list[Chunk]) -> int:
        """Bulk-index chunks with embeddings. Returns count indexed."""
        if not chunks:
            return 0

        actions: list[dict[str, Any]] = []
        for chunk in chunks:
            doc = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "embedding": chunk.embedding or [],
                "metadata": chunk.metadata,
                "chunk_index": chunk.index,
                "created_at": datetime.now(UTC).isoformat(),
            }
            actions.append({"index": {"_index": index_name, "_id": chunk.id}})
            actions.append(doc)

        response = self._client.bulk(body=actions)

        errors = response.get("errors", False)
        if errors:
            logger.error("Bulk indexing errors: %s", response["items"][:5])
        else:
            logger.info("Indexed %d chunks into %s", len(chunks), index_name)

        return len(chunks)

    def delete_chunks(self, index_name: str, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""
        if not chunk_ids:
            return

        actions: list[dict[str, Any]] = []
        for cid in chunk_ids:
            actions.append({"delete": {"_index": index_name, "_id": cid}})

        self._client.bulk(body=actions)
        logger.info("Deleted %d chunks from %s", len(chunk_ids), index_name)

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def knn_search(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 20,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """k-NN vector search. Returns list of {id, score, ...}."""
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
        if query_filter:
            body["query"]["knn"]["embedding"]["filter"] = query_filter

        response = self._client.search(index=index_name, body=body)
        return self._parse_hits(response)

    def text_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """BM25 text search. Returns list of {id, score, ...}."""
        body = {
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "or",
                    }
                }
            },
        }
        response = self._client.search(index=index_name, body=body)
        return self._parse_hits(response)

    def hybrid_search(
        self,
        index_name: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 20,
    ) -> dict[str, list[dict[str, Any]]]:
        """Combined knn + text search.

        Returns both result sets for downstream RRF fusion (Phase 8).
        """
        knn_results = self.knn_search(index_name, query_vector, top_k)
        text_results = self.text_search(index_name, query_text, top_k)
        return {"knn": knn_results, "text": text_results}

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_index_stats(self, index_name: str) -> dict[str, Any]:
        """Get index document count and size."""
        try:
            stats = self._client.indices.stats(index=index_name)
            index_stats = stats["indices"][index_name]["total"]
            return {
                "doc_count": index_stats["docs"]["count"],
                "store_size_bytes": index_stats["store"]["size_in_bytes"],
            }
        except NotFoundError:
            return {"doc_count": 0, "store_size_bytes": 0}

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract hits from OpenSearch response."""
        hits = response.get("hits", {}).get("hits", [])
        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"],
            }
            for hit in hits
        ]

    def close(self) -> None:
        """Close the underlying client transport."""
        self._client.transport.close()
