"""Tests for LLM response cache and CachedLLMBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag_pipeline.generation.cache import LLMResponseCache
from rag_pipeline.generation.cached_llm import CachedLLMBackend
from rag_pipeline.generation.llm import GenerationConfig, LLMBackend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


class MockRedis:
    """Minimal Redis-like dict for testing."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:  # noqa: ARG002
        self._store[key] = value

    def keys(self, pattern: str) -> list[str]:
        prefix = pattern.replace("*", "")
        return [k for k in self._store if k.startswith(prefix)]

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                count += 1
        return count


class FakeLLMBackend(LLMBackend):
    """Test double that records calls and returns a fixed response."""

    def __init__(self, response: str = "fake answer") -> None:
        self._response = response
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []
        self.last_config: GenerationConfig | None = None

    async def generate(
        self, messages: list[dict[str, str]], config: GenerationConfig | None = None
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_config = config
        return self._response

    async def stream(
        self, messages: list[dict[str, str]], config: GenerationConfig | None = None
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        self.last_messages = messages
        self.last_config = config
        for word in self._response.split():
            yield word + " "


# ------------------------------------------------------------------ #
#  LLMResponseCache — no Redis
# ------------------------------------------------------------------ #


class TestLLMResponseCacheNoRedis:
    """Cache with no Redis client should silently no-op."""

    def test_get_returns_none(self) -> None:
        cache = LLMResponseCache()
        assert cache.get([{"role": "user", "content": "hi"}], "m") is None

    def test_set_does_not_raise(self) -> None:
        cache = LLMResponseCache()
        cache.set([{"role": "user", "content": "hi"}], "m", "response")

    def test_clear_returns_zero(self) -> None:
        cache = LLMResponseCache()
        assert cache.clear() == 0

    def test_stats_initial(self) -> None:
        cache = LLMResponseCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0


# ------------------------------------------------------------------ #
#  LLMResponseCache — with MockRedis
# ------------------------------------------------------------------ #


class TestLLMResponseCacheWithRedis:
    """Cache with a mock Redis backend."""

    def test_set_and_get(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        messages = [{"role": "user", "content": "hello"}]
        cache.set(messages, "gpt-4", "world")
        assert cache.get(messages, "gpt-4") == "world"

    def test_get_miss(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        assert cache.get([{"role": "user", "content": "missing"}], "m") is None

    def test_key_deterministic(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        messages = [{"role": "user", "content": "test"}]
        key1 = cache._make_key(messages, "model")
        key2 = cache._make_key(messages, "model")
        assert key1 == key2

    def test_key_differs_by_model(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        messages = [{"role": "user", "content": "test"}]
        key1 = cache._make_key(messages, "gpt-4")
        key2 = cache._make_key(messages, "gpt-3.5")
        assert key1 != key2

    def test_key_differs_by_messages(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        key1 = cache._make_key([{"role": "user", "content": "a"}], "m")
        key2 = cache._make_key([{"role": "user", "content": "b"}], "m")
        assert key1 != key2

    def test_stats_tracking(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        messages = [{"role": "user", "content": "q"}]

        cache.get(messages, "m")  # miss
        cache.get(messages, "m")  # miss
        cache.set(messages, "m", "answer")
        cache.get(messages, "m")  # hit
        cache.get(messages, "m")  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5

    def test_clear(self) -> None:
        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)
        cache.set([{"role": "user", "content": "a"}], "m", "1")
        cache.set([{"role": "user", "content": "b"}], "m", "2")

        cleared = cache.clear()
        assert cleared == 2
        assert cache.get([{"role": "user", "content": "a"}], "m") is None

    def test_prefix_isolation(self) -> None:
        mock = MockRedis()
        cache1 = LLMResponseCache(redis_client=mock, prefix="cache_a")
        cache2 = LLMResponseCache(redis_client=mock, prefix="cache_b")

        messages = [{"role": "user", "content": "x"}]
        cache1.set(messages, "m", "from_a")
        cache2.set(messages, "m", "from_b")

        assert cache1.get(messages, "m") == "from_a"
        assert cache2.get(messages, "m") == "from_b"


# ------------------------------------------------------------------ #
#  CachedLLMBackend
# ------------------------------------------------------------------ #


class TestCachedLLMBackend:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_backend(self) -> None:
        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)
        fake = FakeLLMBackend("generated")
        cached = CachedLLMBackend(fake, cache)

        messages = [{"role": "user", "content": "q"}]

        # First call — cache miss, calls backend
        result = await cached.generate(messages, GenerationConfig(model="m"))
        assert result == "generated"
        assert fake.call_count == 1

        # Second call — cache hit, backend NOT called
        result2 = await cached.generate(messages, GenerationConfig(model="m"))
        assert result2 == "generated"
        assert fake.call_count == 1  # still 1

    @pytest.mark.asyncio
    async def test_cache_miss_calls_backend(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        fake = FakeLLMBackend("new response")
        cached = CachedLLMBackend(fake, cache)

        result = await cached.generate(
            [{"role": "user", "content": "q"}], GenerationConfig(model="m")
        )
        assert result == "new response"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_passes_through(self) -> None:
        cache = LLMResponseCache(redis_client=MockRedis())
        fake = FakeLLMBackend("a b c")
        cached = CachedLLMBackend(fake, cache)

        tokens = [t async for t in cached.stream([{"role": "user", "content": "q"}])]
        assert len(tokens) == 3
        assert tokens[0].strip() == "a"

    @pytest.mark.asyncio
    async def test_stream_does_not_cache(self) -> None:
        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)
        fake = FakeLLMBackend("output")
        cached = CachedLLMBackend(fake, cache)

        _ = [t async for t in cached.stream([{"role": "user", "content": "q"}])]

        # Nothing should be in cache after streaming
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_empty_response_not_cached(self) -> None:
        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)
        fake = FakeLLMBackend("")
        cached = CachedLLMBackend(fake, cache)

        await cached.generate(
            [{"role": "user", "content": "q"}], GenerationConfig(model="m")
        )

        # Empty response should not be cached
        await cached.generate(
            [{"role": "user", "content": "q"}], GenerationConfig(model="m")
        )
        assert fake.call_count == 2  # called again because empty wasn't cached

    @pytest.mark.asyncio
    async def test_different_models_use_separate_cache(self) -> None:
        mock = MockRedis()
        cache = LLMResponseCache(redis_client=mock)
        fake = FakeLLMBackend("response")
        cached = CachedLLMBackend(fake, cache)

        messages = [{"role": "user", "content": "q"}]

        await cached.generate(messages, GenerationConfig(model="gpt-4"))
        await cached.generate(messages, GenerationConfig(model="gpt-3.5"))

        # Both should have been cache misses (different models)
        assert fake.call_count == 2

        stats = cache.get_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0
