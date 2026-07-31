"""BM25 retrieval backed by OpenSearch.

Uses OpenSearch's built-in BM25 scoring via match queries.
This is the production-ready BM25 implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_pipeline.storage.opensearch import OpenSearchClient

logger = logging.getLogger(__name__)


class OpenSearchBM25:
    """BM25 search using OpenSearch's built-in scoring.

    Delegates to OpenSearch's match query which uses BM25 internally.
    """

    def __init__(
        self,
        client: OpenSearchClient,
        index_name: str,
    ) -> None:
        self._client = client
        self._index_name = index_name

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 text search via OpenSearch.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            metadata_filter: Optional filter dict (e.g., {"document_id": "d1"}).

        Returns:
            List of {id, score, content, metadata, ...} dicts.
        """
        body: dict[str, Any] = {
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

        if metadata_filter:
            body["query"] = {
                "bool": {
                    "must": [body["query"]],
                    "filter": [{"term": {k: v}} for k, v in metadata_filter.items()],
                }
            }

        response = self._client._client.search(index=self._index_name, body=body)  # noqa: SLF001
        return self._parse_hits(response)

    def multi_field_search(
        self,
        query: str,
        fields: list[str] | None = None,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search across multiple fields.

        Args:
            query: Search query text.
            fields: Fields to search (default: ["content"]).
            top_k: Number of results to return.
            metadata_filter: Optional filter dict.

        Returns:
            List of {id, score, content, ...} dicts.
        """
        fields = fields or ["content"]

        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                    "operator": "or",
                }
            },
        }

        if metadata_filter:
            body["query"] = {
                "bool": {
                    "must": [body["query"]],
                    "filter": [{"term": {k: v}} for k, v in metadata_filter.items()],
                }
            }

        response = self._client._client.search(index=self._index_name, body=body)  # noqa: SLF001
        return self._parse_hits(response)

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
