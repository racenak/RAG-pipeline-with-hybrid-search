"""Storage — OpenSearch, PostgreSQL, Redis, and S3."""

from rag_pipeline.data.storage import S3Storage
from rag_pipeline.storage.clients import (
    close_all_clients,
    get_opensearch_async_client,
    get_opensearch_client,
    get_postgres_pool,
    get_redis_client,
    reset_clients,
)
from rag_pipeline.storage.opensearch import OpenSearchClient
from rag_pipeline.storage.postgres import (
    ChunkRecord,
    DocumentRecord,
    PostgresClient,
)
from rag_pipeline.storage.redis_cache import QueryCache, RedisEmbeddingCache

__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "OpenSearchClient",
    "PostgresClient",
    "QueryCache",
    "RedisEmbeddingCache",
    "S3Storage",
    "close_all_clients",
    "get_opensearch_async_client",
    "get_opensearch_client",
    "get_postgres_pool",
    "get_redis_client",
    "reset_clients",
]
