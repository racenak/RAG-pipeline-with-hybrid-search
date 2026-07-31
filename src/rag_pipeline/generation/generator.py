"""RAG generator — orchestrates context building, prompting, and LLM generation."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator  # noqa: TC003
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.data.chunking import Chunk
from rag_pipeline.generation.citations import CitationBundle, CitationManager
from rag_pipeline.generation.context import ContextBuilder, ContextConfig
from rag_pipeline.generation.llm import GenerationConfig, LLMBackend, get_llm_backend
from rag_pipeline.generation.prompt import PromptBuilder, PromptConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a RAG generation call."""

    answer: str
    context_used: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    citations: CitationBundle | None = None


class RAGGenerator:
    """Orchestrates the full RAG generation pipeline."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend
        self._context_builder = ContextBuilder()
        self._prompt_builder = PromptBuilder()
        self._citation_manager = CitationManager()

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = get_llm_backend()
        return self._backend

    def generate(
        self,
        query: str,
        chunks: list[Chunk],
        config: GenerationConfig | None = None,
        context_config: ContextConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> GenerationResult:
        """Run the full RAG generation pipeline synchronously.

        Builds context from chunks, assembles the prompt, and calls the LLM.
        """
        cfg = config or GenerationConfig()
        ctx_config = context_config or ContextConfig()
        p_config = prompt_config or PromptConfig()

        context = self._context_builder.build_context(chunks, ctx_config)
        messages = self._prompt_builder.build_prompt(query, context, p_config)

        msg_list = [{"role": "system", "content": messages["system"]}, {"role": "user", "content": messages["user"]}]

        t0 = time.monotonic()
        backend = self._get_backend()
        import asyncio

        answer = asyncio.get_event_loop().run_until_complete(backend.generate(msg_list, cfg))
        latency = (time.monotonic() - t0) * 1000

        citations = self._citation_manager.process(answer, chunks)

        return GenerationResult(
            answer=answer,
            context_used=context,
            model=cfg.model,
            latency_ms=round(latency, 1),
            citations=citations,
        )

    async def agenerate(
        self,
        query: str,
        chunks: list[Chunk],
        config: GenerationConfig | None = None,
        context_config: ContextConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> GenerationResult:
        """Run the full RAG generation pipeline asynchronously."""
        cfg = config or GenerationConfig()
        ctx_config = context_config or ContextConfig()
        p_config = prompt_config or PromptConfig()

        context = self._context_builder.build_context(chunks, ctx_config)
        messages = self._prompt_builder.build_prompt(query, context, p_config)

        msg_list = [{"role": "system", "content": messages["system"]}, {"role": "user", "content": messages["user"]}]

        t0 = time.monotonic()
        backend = self._get_backend()
        answer = await backend.generate(msg_list, cfg)
        latency = (time.monotonic() - t0) * 1000

        citations = self._citation_manager.process(answer, chunks)

        return GenerationResult(
            answer=answer,
            context_used=context,
            model=cfg.model,
            latency_ms=round(latency, 1),
            citations=citations,
        )

    async def stream_generate(
        self,
        query: str,
        chunks: list[Chunk],
        config: GenerationConfig | None = None,
        context_config: ContextConfig | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield response tokens as they arrive from the LLM."""
        cfg = config or GenerationConfig()
        ctx_config = context_config or ContextConfig()
        p_config = prompt_config or PromptConfig()

        context = self._context_builder.build_context(chunks, ctx_config)
        messages = self._prompt_builder.build_prompt(query, context, p_config)

        msg_list = [{"role": "system", "content": messages["system"]}, {"role": "user", "content": messages["user"]}]

        backend = self._get_backend()
        async for token in backend.stream(msg_list, cfg):
            yield token
