"""Tests for connection pooling singletons."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_pipeline.storage import clients


class TestSingletons:
    """Verify each getter returns the same instance across calls."""

    def setup_method(self) -> None:
        clients.reset_clients()

    def test_opensearch_singleton(self):
        mock_client = MagicMock()
        with patch("opensearchpy.OpenSearch", return_value=mock_client):
            a = clients.get_opensearch_client()
            b = clients.get_opensearch_client()
        assert a is b

    def test_opensearch_async_singleton(self):
        mock_client = MagicMock()
        with patch("opensearchpy.AsyncOpenSearch", return_value=mock_client):
            a = clients.get_opensearch_async_client()
            b = clients.get_opensearch_async_client()
        assert a is b

    def test_postgres_pool_singleton(self):
        mock_pool = MagicMock()
        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
            a = clients.get_postgres_pool()
            b = clients.get_postgres_pool()
        assert a is b

    def test_redis_singleton(self):
        mock_redis = MagicMock()
        with patch("redis.Redis", return_value=mock_redis):
            a = clients.get_redis_client()
            b = clients.get_redis_client()
        assert a is b


class TestResetClients:
    """Verify reset_clients clears all singletons."""

    def setup_method(self) -> None:
        clients.reset_clients()

    def test_reset_opensearch(self):
        mock_client = MagicMock()
        with patch("opensearchpy.OpenSearch", return_value=mock_client):
            first = clients.get_opensearch_client()
        clients.reset_clients()
        mock_client2 = MagicMock()
        with patch("opensearchpy.OpenSearch", return_value=mock_client2):
            second = clients.get_opensearch_client()
        assert first is not second

    def test_reset_postgres_pool(self):
        mock_pool = MagicMock()
        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
            first = clients.get_postgres_pool()
        clients.reset_clients()
        mock_pool2 = MagicMock()
        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool2):
            second = clients.get_postgres_pool()
        assert first is not second

    def test_reset_redis(self):
        mock_redis = MagicMock()
        with patch("redis.Redis", return_value=mock_redis):
            first = clients.get_redis_client()
        clients.reset_clients()
        mock_redis2 = MagicMock()
        with patch("redis.Redis", return_value=mock_redis2):
            second = clients.get_redis_client()
        assert first is not second

    def test_reset_all_at_once(self):
        """After creating all clients, reset_clients clears all four singletons."""
        mock_os = MagicMock()
        mock_async_os = MagicMock()
        mock_pool = MagicMock()
        mock_redis = MagicMock()

        with patch("opensearchpy.OpenSearch", return_value=mock_os), \
             patch("opensearchpy.AsyncOpenSearch", return_value=mock_async_os), \
             patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool), \
             patch("redis.Redis", return_value=mock_redis):
            clients.get_opensearch_client()
            clients.get_opensearch_async_client()
            clients.get_postgres_pool()
            clients.get_redis_client()

        clients.reset_clients()

        # After reset, calling getters should create new instances
        mock_os2 = MagicMock()
        mock_async_os2 = MagicMock()
        mock_pool2 = MagicMock()
        mock_redis2 = MagicMock()

        with patch("opensearchpy.OpenSearch", return_value=mock_os2), \
             patch("opensearchpy.AsyncOpenSearch", return_value=mock_async_os2), \
             patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool2), \
             patch("redis.Redis", return_value=mock_redis2):
            assert clients.get_opensearch_client() is mock_os2
            assert clients.get_opensearch_async_client() is mock_async_os2
            assert clients.get_postgres_pool() is mock_pool2
            assert clients.get_redis_client() is mock_redis2


class TestCloseAllClients:
    """Verify close_all_clients tears down singletons correctly."""

    def setup_method(self) -> None:
        clients.reset_clients()

    def test_close_opensearch_calls_transport_close(self):
        mock_client = MagicMock()
        with patch("opensearchpy.OpenSearch", return_value=mock_client):
            clients.get_opensearch_client()

        clients.close_all_clients()
        mock_client.transport.close.assert_called_once()
        assert clients._opensearch_client is None

    def test_close_postgres_calls_closeall(self):
        mock_pool = MagicMock()
        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
            clients.get_postgres_pool()

        clients.close_all_clients()
        mock_pool.closeall.assert_called_once()
        assert clients._postgres_pool is None

    def test_close_redis_calls_close(self):
        mock_redis = MagicMock()
        with patch("redis.Redis", return_value=mock_redis):
            clients.get_redis_client()

        clients.close_all_clients()
        mock_redis.close.assert_called_once()
        assert clients._redis_client is None

    def test_close_all_when_none_created(self):
        """close_all_clients should be a no-op when no clients exist."""
        clients.close_all_clients()  # Should not raise
        assert clients._opensearch_client is None
        assert clients._postgres_pool is None
        assert clients._redis_client is None
