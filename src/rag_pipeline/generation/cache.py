"""LLM response cache — cache LLM outputs to save API costs during evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """Redis-based cache for LLM responses. Content-addressed by prompt hash."""

    def __init__(
        self, redis_client: Any = None, ttl_seconds: int = 3600, prefix: str = "llm_cache"
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = prefix
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages: list[dict[str, str]], model: str) -> str:
        """Create cache key from messages + model."""
        content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self._prefix}:{hash_val}"

    def get(self, messages: list[dict[str, str]], model: str) -> str | None:
        """Get cached response. Returns None if miss."""
        if not self._redis:
            return None

        key = self._make_key(messages, model)
        try:
            cached = self._redis.get(key)
            if cached:
                self._hits += 1
                logger.debug("LLM cache hit: %s", key)
                return cached
            self._misses += 1
            return None
        except Exception:
            return None

    def set(self, messages: list[dict[str, str]], model: str, response: str) -> None:
        """Cache an LLM response."""
        if not self._redis:
            return

        key = self._make_key(messages, model)
        try:
            self._redis.setex(key, self._ttl, response)
            logger.debug("LLM cache set: %s", key)
        except Exception:
            logger.warning("Failed to cache LLM response")

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        if not self._redis:
            return 0
        try:
            keys = self._redis.keys(f"{self._prefix}:*")
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception:
            return 0
