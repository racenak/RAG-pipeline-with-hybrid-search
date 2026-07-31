"""Tests for generation module — context, prompts, LLM backends, and generator."""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003

import pytest

from rag_pipeline.data.chunking import Chunk
from rag_pipeline.generation.context import ContextBuilder, ContextConfig, estimate_tokens
from rag_pipeline.generation.generator import GenerationResult, RAGGenerator
from rag_pipeline.generation.llm import GenerationConfig, LLMBackend, OllamaBackend
from rag_pipeline.generation.prompt import PromptBuilder, PromptConfig

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _make_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i}",
            document_id="doc-1",
            content=f"This is chunk number {i} with some content about topic {i}.",
            index=i,
            token_count=20,
            metadata={"score": 0.9 - i * 0.1},
        )
        for i in range(n)
    ]


class FakeLLMBackend(LLMBackend):
    """Test double that returns a fixed response."""

    def __init__(self, response: str = "fake answer") -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []
        self.last_config: GenerationConfig | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        self.last_messages = messages
        self.last_config = config
        return self._response

    async def stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        self.last_messages = messages
        self.last_config = config
        for word in self._response.split():
            yield word + " "


# ------------------------------------------------------------------ #
#  estimate_tokens
# ------------------------------------------------------------------ #


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        assert estimate_tokens("hello") >= 1

    def test_longer_text(self) -> None:
        short = estimate_tokens("hi")
        long = estimate_tokens("this is a much longer sentence with many words")
        assert long > short

    def test_consistent_results(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        assert estimate_tokens(text) == estimate_tokens(text)


# ------------------------------------------------------------------ #
#  ContextBuilder
# ------------------------------------------------------------------ #


class TestContextBuilder:
    def test_empty_chunks(self) -> None:
        builder = ContextBuilder()
        result = builder.build_context([])
        assert result == ""

    def test_single_chunk_xml(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(1)
        result = builder.build_context(chunks)
        assert "<context>" in result
        assert "</context>" in result
        assert "chunk number 0" in result

    def test_multiple_chunks_xml(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(3)
        result = builder.build_context(chunks, ContextConfig(separator="xml"))
        assert result.count("<chunk") == 3

    def test_markdown_separator(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(2)
        result = builder.build_context(chunks, ContextConfig(separator="markdown"))
        assert "### Source [1]" in result
        assert "### Source [2]" in result
        assert "<context>" not in result

    def test_numbered_separator(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(2)
        result = builder.build_context(chunks, ContextConfig(separator="numbered"))
        assert "[1]" in result
        assert "[2]" in result

    def test_deduplication(self) -> None:
        builder = ContextBuilder()
        chunk = Chunk(id="a", document_id="d", content="duplicate text", index=0, token_count=5)
        duplicate = Chunk(id="b", document_id="d", content="duplicate text", index=1, token_count=5)
        result = builder.build_context([chunk, duplicate])
        assert result.count("duplicate text") == 1

    def test_order_by_score(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(3)
        result = builder.build_context(chunks, ContextConfig(order_by="score"))
        # Highest score chunk (index 0, score 0.9) should appear first
        pos_0 = result.index("chunk number 0")
        pos_2 = result.index("chunk number 2")
        assert pos_0 < pos_2

    def test_order_by_position(self) -> None:
        builder = ContextBuilder()
        chunks = list(reversed(_make_chunks(3)))
        result = builder.build_context(chunks, ContextConfig(order_by="position"))
        pos_0 = result.index("chunk number 0")
        pos_2 = result.index("chunk number 2")
        assert pos_0 < pos_2

    def test_token_budget_respected(self) -> None:
        builder = ContextBuilder()
        chunks = _make_chunks(10)
        result = builder.build_context(chunks, ContextConfig(max_tokens=100))
        assert estimate_tokens(result) <= 120  # small margin for wrappers

    def test_no_metadata_score_defaults_zero(self) -> None:
        builder = ContextBuilder()
        chunk = Chunk(id="x", document_id="d", content="no score here", index=0, token_count=5)
        result = builder.build_context([chunk], ContextConfig(order_by="score"))
        assert "no score here" in result


# ------------------------------------------------------------------ #
#  PromptBuilder
# ------------------------------------------------------------------ #


class TestPromptBuilder:
    def test_default_prompts(self) -> None:
        builder = PromptBuilder()
        result = builder.build_prompt("What is AI?", "AI is artificial intelligence.")
        assert "system" in result
        assert "user" in result
        assert "AI?" in result["user"]
        assert "AI is artificial intelligence" in result["user"]

    def test_custom_system_prompt(self) -> None:
        builder = PromptBuilder()
        config = PromptConfig(system_prompt="Be concise.")
        result = builder.build_prompt("query", "ctx", config)
        assert result["system"] == "Be concise."

    def test_custom_user_template(self) -> None:
        builder = PromptBuilder()
        config = PromptConfig(user_template="Q: {query}\nC: {context}")
        result = builder.build_prompt("hello", "world", config)
        assert result["user"] == "Q: hello\nC: world"

    def test_citation_instruction_appended(self) -> None:
        builder = PromptBuilder()
        config = PromptConfig(citation_instruction="Always cite [1], [2].")
        result = builder.build_prompt("q", "c", config)
        assert "Always cite [1], [2]." in result["system"]

    def test_empty_citation_not_added(self) -> None:
        builder = PromptBuilder()
        config = PromptConfig(citation_instruction="")
        result = builder.build_prompt("q", "c", config)
        assert "Cite your sources" in result["system"]


# ------------------------------------------------------------------ #
#  LLMBackend / GenerationConfig
# ------------------------------------------------------------------ #


class TestGenerationConfig:
    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024
        assert cfg.top_p == 0.9

    def test_custom_values(self) -> None:
        cfg = GenerationConfig(model="gpt-3.5-turbo", temperature=0.0, max_tokens=500)
        assert cfg.model == "gpt-3.5-turbo"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 500


class TestFakeLLMBackend:
    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        backend = FakeLLMBackend("hello world")
        result = await backend.generate([{"role": "user", "content": "hi"}])
        assert result == "hello world"
        assert len(backend.last_messages) == 1

    @pytest.mark.asyncio
    async def test_stream(self) -> None:
        backend = FakeLLMBackend("a b c")
        tokens = [t async for t in backend.stream([{"role": "user", "content": "hi"}])]
        assert len(tokens) == 3
        assert tokens[0].strip() == "a"


# ------------------------------------------------------------------ #
#  RAGGenerator
# ------------------------------------------------------------------ #


class TestRAGGenerator:
    @pytest.mark.asyncio
    async def test_agenerate(self) -> None:
        backend = FakeLLMBackend("The answer is 42.")
        generator = RAGGenerator(backend=backend)
        chunks = _make_chunks(2)
        result = await generator.agenerate("What is the answer?", chunks)

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is 42."
        assert result.model == "gpt-4"
        assert "<context>" in result.context_used
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_agenerate_custom_config(self) -> None:
        backend = FakeLLMBackend("custom")
        generator = RAGGenerator(backend=backend)
        config = GenerationConfig(model="gpt-3.5-turbo", temperature=0.0)
        result = await generator.agenerate("q", _make_chunks(1), config=config)
        assert result.model == "gpt-3.5-turbo"
        assert backend.last_config is not None
        assert backend.last_config.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_stream_generate(self) -> None:
        backend = FakeLLMBackend("token1 token2 token3")
        generator = RAGGenerator(backend=backend)
        tokens = [t async for t in generator.stream_generate("q", _make_chunks(1))]
        assert len(tokens) == 3

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        backend = FakeLLMBackend("no context")
        generator = RAGGenerator(backend=backend)
        result = await generator.agenerate("q", [])
        assert result.answer == "no context"
        assert result.context_used == ""

    @pytest.mark.asyncio
    async def test_messages_structure(self) -> None:
        backend = FakeLLMBackend("ok")
        generator = RAGGenerator(backend=backend)
        await generator.agenerate("test query", _make_chunks(1))
        assert backend.last_messages[0]["role"] == "system"
        assert backend.last_messages[1]["role"] == "user"
        assert "test query" in backend.last_messages[1]["content"]


# ------------------------------------------------------------------ #
#  OllamaBackend (unit tests — no real server)
# ------------------------------------------------------------------ #


class TestOllamaBackend:
    def test_init_default_url(self) -> None:
        backend = OllamaBackend()
        assert backend._base_url == "http://localhost:11434"

    def test_init_custom_url(self) -> None:
        backend = OllamaBackend(base_url="http://custom:9999")
        assert backend._base_url == "http://custom:9999"
