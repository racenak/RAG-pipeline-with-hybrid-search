"""LLM backends — OpenAI and Ollama implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator  # noqa: TC003
from dataclasses import dataclass
from typing import Any

import httpx

from rag_pipeline.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """LLM generation parameters."""

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a complete response from messages."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield response tokens as they arrive."""


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible backend using the ``openai`` library."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI

        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        response = await self._client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or GenerationConfig()
        response = await self._client.chat.completions.create(
            model=cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class OllamaBackend(LLMBackend):
    """Ollama backend using httpx for direct API calls."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
            },
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")

    async def stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or GenerationConfig()
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
            },
        }
        async with (
            httpx.AsyncClient(timeout=120) as client,
            client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp,
        ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content


def get_llm_backend(config: Any = None, use_cache: bool = False) -> LLMBackend:  # noqa: ARG001
    """Factory — create an LLM backend from application settings."""
    settings = get_settings()
    provider = settings.generation.provider.lower()

    if provider in ("openai", "openrouter"):
        backend = OpenAIBackend(
            api_key=settings.openrouter_api_key or None,
            base_url=settings.generation.base_url or None,
        )
    elif provider == "ollama":
        backend = OllamaBackend()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    if use_cache:
        from rag_pipeline.generation.cache import LLMResponseCache
        from rag_pipeline.generation.cached_llm import CachedLLMBackend
        from rag_pipeline.storage.clients import get_redis_client

        cache = LLMResponseCache(redis_client=get_redis_client())
        backend = CachedLLMBackend(backend, cache)

    return backend
