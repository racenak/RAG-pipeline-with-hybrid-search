"""Redis cache — embedding cache + query result cache."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import redis

logger = logging.getLogger(__name__)


class RedisEmbeddingCache:
    """Redis-backed embedding cache.

    Same interface as EmbeddingCache but backed by Redis for
    shared access across processes and containers.
    """

    PREFIX = "emb:"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 86400,  # 24h default
    ) -> None:
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()
        return f"{RedisEmbeddingCache.PREFIX}{h}"

    def has(self, text: str) -> bool:
        return self._client.exists(self._key(text)) > 0

    def get(self, text: str) -> list[float] | None:
        data = self._client.get(self._key(text))
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).tolist()

    def set(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        blob = np.array(vector, dtype=np.float32).tobytes()
        self._client.setex(key, self._ttl, blob)

    def get_many(self, texts: list[str]) -> dict[int, list[float]]:
        if not texts:
            return {}
        keys = [self._key(t) for t in texts]
        values = self._client.mget(keys)
        results: dict[int, list[float]] = {}
        for i, val in enumerate(values):
            if val is not None:
                results[i] = np.frombuffer(val, dtype=np.float32).tolist()
        return results

    def set_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        if not texts:
            return
        pipe = self._client.pipeline()
        for text, vector in zip(texts, vectors, strict=True):
            key = self._key(text)
            blob = np.array(vector, dtype=np.float32).tobytes()
            pipe.setex(key, self._ttl, blob)
        pipe.execute()

    def clear(self) -> int:
        """Delete all embedding keys. Returns count deleted."""
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=500)
            if keys:
                count += self._client.delete(*keys)
            if cursor == 0:
                break
        logger.info("Cleared %d cached embeddings from Redis", count)
        return count

    @property
    def _prefix(self) -> str:
        return self.PREFIX

    @property
    def size(self) -> int:
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=500)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def close(self) -> None:
        self._client.close()


class QueryCache:
    """Redis-backed query result cache for search results."""

    PREFIX = "query:"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 300,  # 5min default
    ) -> None:
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(query: str, top_k: int) -> str:
        h = hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()
        return f"{QueryCache.PREFIX}{h}"

    def get(self, query: str, top_k: int = 20) -> list[dict[str, Any]] | None:
        """Retrieve cached search results."""
        data = self._client.get(self._key(query, top_k))
        if data is None:
            return None
        return json.loads(data)

    def set(self, query: str, results: list[dict[str, Any]], top_k: int = 20) -> None:
        """Cache search results."""
        key = self._key(query, top_k)
        self._client.setex(key, self._ttl, json.dumps(results))

    def clear(self) -> int:
        """Delete all query cache entries. Returns count deleted."""
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self.PREFIX}*", count=500)
            if keys:
                count += self._client.delete(*keys)
            if cursor == 0:
                break
        logger.info("Cleared %d cached queries from Redis", count)
        return count

    def close(self) -> None:
        self._client.close()
