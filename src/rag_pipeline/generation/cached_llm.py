"""Cache-aware LLM wrapper — transparently caches LLM responses."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_pipeline.generation.llm import GenerationConfig, LLMBackend

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from rag_pipeline.generation.cache import LLMResponseCache

logger = logging.getLogger(__name__)


class CachedLLMBackend(LLMBackend):
    """Wrapper that adds caching to any LLM backend."""

    def __init__(self, backend: LLMBackend, cache: LLMResponseCache) -> None:
        self._backend = backend
        self._cache = cache

    async def generate(
        self, messages: list[dict[str, str]], config: GenerationConfig | None = None
    ) -> str:
        cfg = config or GenerationConfig()

        cached = self._cache.get(messages, cfg.model)
        if cached:
            return cached

        response = await self._backend.generate(messages, config)

        if response:
            self._cache.set(messages, cfg.model, response)

        return response

    async def stream(
        self, messages: list[dict[str, str]], config: GenerationConfig | None = None
    ) -> AsyncGenerator[str, None]:
        async for token in self._backend.stream(messages, config):
            yield token
