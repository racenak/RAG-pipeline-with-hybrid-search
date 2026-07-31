"""Connection pooling — singleton clients for OpenSearch, PostgreSQL, Redis."""

from __future__ import annotations

import logging
from typing import Any

from rag_pipeline.reliability.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Lazy singletons
_opensearch_client = None
_opensearch_async_client = None
_postgres_pool = None
_redis_client = None

# Circuit breakers for external services
_opensearch_breaker = CircuitBreaker(
    failure_threshold=5, recovery_timeout=30, name="opensearch"
)
_postgres_breaker = CircuitBreaker(
    failure_threshold=5, recovery_timeout=30, name="postgres"
)
_redis_breaker = CircuitBreaker(
    failure_threshold=5, recovery_timeout=30, name="redis"
)


def get_opensearch_client():
    """Get or create a singleton OpenSearch client with connection pooling."""
    global _opensearch_client
    if _opensearch_client is None:
        from opensearchpy import OpenSearch
        from rag_pipeline.config import get_settings
        settings = get_settings()

        _opensearch_client = OpenSearch(
            hosts=[{
                "host": settings.storage.opensearch_host,
                "port": settings.storage.opensearch_port,
            }],
            http_compress=True,
            use_ssl=False,
            max_retries=settings.storage.opensearch_max_retries,
            retry_on_timeout=settings.storage.opensearch_retry_on_timeout,
            timeout=settings.storage.opensearch_timeout,
        )
        logger.info(
            "OpenSearch client created: %s:%s",
            settings.storage.opensearch_host,
            settings.storage.opensearch_port,
        )
    return _opensearch_client


def get_opensearch_async_client():
    """Get or create a singleton async OpenSearch client."""
    global _opensearch_async_client
    if _opensearch_async_client is None:
        from opensearchpy import AsyncOpenSearch
        from rag_pipeline.config import get_settings
        settings = get_settings()

        _opensearch_async_client = AsyncOpenSearch(
            hosts=[{
                "host": settings.storage.opensearch_host,
                "port": settings.storage.opensearch_port,
            }],
            http_compress=True,
            use_ssl=False,
            max_retries=settings.storage.opensearch_max_retries,
            retry_on_timeout=settings.storage.opensearch_retry_on_timeout,
            timeout=settings.storage.opensearch_timeout,
        )
        logger.info("Async OpenSearch client created")
    return _opensearch_async_client


def get_postgres_pool():
    """Get or create a singleton PostgreSQL connection pool."""
    global _postgres_pool
    if _postgres_pool is None:
        import psycopg2.pool
        from rag_pipeline.config import get_settings
        settings = get_settings()

        _postgres_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=settings.storage.postgres_pool_size,
            dsn=settings.storage.postgres_dsn,
        )
        logger.info("PostgreSQL pool created: %s", settings.storage.postgres_host)
    return _postgres_pool


def get_redis_client():
    """Get or create a singleton Redis client."""
    global _redis_client
    if _redis_client is None:
        import redis
        from rag_pipeline.config import get_settings
        settings = get_settings()

        _redis_client = redis.Redis(
            host=settings.storage.redis_host,
            port=settings.storage.redis_port,
            db=settings.storage.redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        logger.info("Redis client created: %s", settings.storage.redis_host)
    return _redis_client


def close_all_clients():
    """Close all singleton clients (for graceful shutdown)."""
    global _opensearch_client, _opensearch_async_client, _postgres_pool, _redis_client

    if _opensearch_client:
        _opensearch_client.transport.close()
        _opensearch_client = None
        logger.info("OpenSearch client closed")

    if _opensearch_async_client:
        _opensearch_async_client = None
        logger.info("Async OpenSearch client released")

    if _postgres_pool:
        _postgres_pool.closeall()
        _postgres_pool = None
        logger.info("PostgreSQL pool closed")

    if _redis_client:
        _redis_client.close()
        _redis_client = None
        logger.info("Redis client closed")


def reset_clients():
    """Reset all singletons (for testing)."""
    global _opensearch_client, _opensearch_async_client, _postgres_pool, _redis_client
    _opensearch_client = None
    _opensearch_async_client = None
    _postgres_pool = None
    _redis_client = None
