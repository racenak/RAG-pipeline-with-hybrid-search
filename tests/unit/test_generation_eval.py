"""Tests for generation evaluation — metrics, hallucination, LLM judge, latency."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from rag_pipeline.evaluation.generation import (
    GenerationMetrics,
    GenerationResult,
    LLMJudge,
    _heuristic_completeness,
    _heuristic_faithfulness,
    _heuristic_relevance,
    _lcs_length,
    bleu,
    detect_hallucination,
    rouge_l,
    rouge_1,
    word_overlap,
)
from rag_pipeline.evaluation.latency import LatencyMetrics, LatencyTracker
from rag_pipeline.generation.llm import GenerationConfig, LLMBackend


# ------------------------------------------------------------------ #
#  Fake LLM backend for judge tests
# ------------------------------------------------------------------ #


class FakeLLMBackend(LLMBackend):
    def __init__(self, response: str = "0.8") -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        self.last_messages = messages
        return self._response

    async def stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        yield self._response


# ------------------------------------------------------------------ #
#  rouge_1
# ------------------------------------------------------------------ #


class TestRouge1:
    def test_identical(self) -> None:
        assert rouge_1("the cat sat", "the cat sat") == 1.0

    def test_completely_different(self) -> None:
        assert rouge_1("dog ran fast", "cat sat still") == 0.0

    def test_partial_overlap(self) -> None:
        score = rouge_1("the cat sat on the mat", "the cat is on the mat")
        assert 0.5 < score < 1.0

    def test_empty_answer(self) -> None:
        assert rouge_1("", "hello world") == 0.0

    def test_empty_reference(self) -> None:
        assert rouge_1("hello world", "") == 0.0

    def test_both_empty(self) -> None:
        assert rouge_1("", "") == 0.0

    def test_symmetry(self) -> None:
        assert rouge_1("a b c", "a b d") == rouge_1("a b d", "a b c")


# ------------------------------------------------------------------ #
#  rouge_l
# ------------------------------------------------------------------ #


class TestRougeL:
    def test_identical(self) -> None:
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self) -> None:
        assert rouge_l("a b c", "x y z") == 0.0

    def test_known_lcs(self) -> None:
        # LCS("abcde", "ace") = 3 (a, c, e)
        assert _lcs_length(list("abcde"), list("ace")) == 3

    def test_subsequence(self) -> None:
        score = rouge_l("a b c d", "a c")
        assert score > 0.5

    def test_empty(self) -> None:
        assert rouge_l("", "hello") == 0.0


# ------------------------------------------------------------------ #
#  bleu
# ------------------------------------------------------------------ #


class TestBleu:
    def test_identical(self) -> None:
        score = bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

    def test_no_overlap(self) -> None:
        score = bleu("dog ran", "cat sat")
        assert score == 0.0

    def test_empty_answer(self) -> None:
        assert bleu("", "hello world") == 0.0

    def test_empty_reference(self) -> None:
        assert bleu("hello world", "") == 0.0

    def test_brevity_penalty(self) -> None:
        short = bleu("cat", "the cat sat on the mat")
        long = bleu("the cat sat on the mat", "the cat sat on the mat")
        assert short < long


# ------------------------------------------------------------------ #
#  word_overlap
# ------------------------------------------------------------------ #


class TestWordOverlap:
    def test_identical(self) -> None:
        assert word_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert word_overlap("a b c", "x y z") == 0.0

    def test_partial(self) -> None:
        score = word_overlap("a b c", "b c d")
        assert 0.3 < score < 0.7

    def test_empty(self) -> None:
        assert word_overlap("", "") == 0.0


# ------------------------------------------------------------------ #
#  detect_hallucination
# ------------------------------------------------------------------ #


class TestDetectHallucination:
    def test_no_hallucination(self) -> None:
        context = "The cat sat on the mat. The cat was happy."
        answer = "The cat sat on the mat."
        rate, flagged = detect_hallucination(answer, context)
        assert rate == 0.0
        assert flagged == []

    def test_hallucination(self) -> None:
        context = "The cat sat on the mat."
        answer = "The cat sat on the mat. The dragon flew over the mountain."
        rate, flagged = detect_hallucination(answer, context)
        assert rate > 0.0
        assert len(flagged) == 1
        assert "dragon" in flagged[0]

    def test_empty_answer(self) -> None:
        rate, flagged = detect_hallucination("", "some context")
        assert rate == 0.0
        assert flagged == []


# ------------------------------------------------------------------ #
#  Heuristic scoring
# ------------------------------------------------------------------ #


class TestHeuristics:
    def test_faithfulness_high_overlap(self) -> None:
        score = _heuristic_faithfulness("the cat sat", "the cat sat on the mat")
        assert score > 0.5

    def test_faithfulness_low_overlap(self) -> None:
        score = _heuristic_faithfulness("dragon flew", "the cat sat")
        assert score < 0.1

    def test_relevance_high(self) -> None:
        score = _heuristic_relevance("python programming", "Python is a programming language.")
        assert score > 0.3

    def test_relevance_low(self) -> None:
        score = _heuristic_relevance("python programming language", "the weather nice today")
        assert score < 0.1

    def test_completeness_high(self) -> None:
        score = _heuristic_completeness("q", "the answer is 42", "the answer is 42")
        assert score == 1.0

    def test_completeness_low(self) -> None:
        score = _heuristic_completeness("q", "something else", "the answer is 42")
        assert score < 0.2


# ------------------------------------------------------------------ #
#  LLMJudge (heuristic fallback — no backend)
# ------------------------------------------------------------------ #


class TestLLMJudgeHeuristic:
    @pytest.mark.asyncio
    async def test_faithfulness_no_backend(self) -> None:
        judge = LLMJudge()
        score = await judge.score_faithfulness("the cat sat", "the cat sat on the mat")
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    @pytest.mark.asyncio
    async def test_relevance_no_backend(self) -> None:
        judge = LLMJudge()
        score = await judge.score_relevance("what is python?", "Python is a language.")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_completeness_no_backend(self) -> None:
        judge = LLMJudge()
        score = await judge.score_completeness("q", "the answer", "the answer is 42")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_faithfulness_with_backend(self) -> None:
        backend = FakeLLMBackend("0.9")
        judge = LLMJudge(backend=backend)
        score = await judge.score_faithfulness("answer text", "context text")
        assert score == 0.9
        assert len(backend.last_messages) == 1

    @pytest.mark.asyncio
    async def test_relevance_with_backend(self) -> None:
        backend = FakeLLMBackend("0.7")
        judge = LLMJudge(backend=backend)
        score = await judge.score_relevance("query", "answer")
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_completeness_with_backend(self) -> None:
        backend = FakeLLMBackend("0.6")
        judge = LLMJudge(backend=backend)
        score = await judge.score_completeness("q", "a", "e")
        assert score == 0.6


# ------------------------------------------------------------------ #
#  LatencyTracker
# ------------------------------------------------------------------ #


class TestLatencyTracker:
    def test_start_stop(self) -> None:
        tracker = LatencyTracker()
        tracker.start("test")
        elapsed = tracker.stop("test")
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)

    def test_stop_without_start(self) -> None:
        tracker = LatencyTracker()
        assert tracker.stop("missing") == 0.0

    def test_record_and_summary(self) -> None:
        tracker = LatencyTracker()
        tracker.record(LatencyMetrics(total_ms=100.0, generation_ms=50.0))
        tracker.record(LatencyMetrics(total_ms=200.0, generation_ms=100.0))
        summary = tracker.get_summary()
        assert summary.total_ms == 150.0
        assert summary.generation_ms == 75.0

    def test_empty_summary(self) -> None:
        tracker = LatencyTracker()
        summary = tracker.get_summary()
        assert summary.total_ms == 0.0

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        tracker = LatencyTracker()
        async with tracker.track("op"):
            await asyncio.sleep(0.01)
        summary = tracker.get_summary()
        assert summary.total_ms > 0.0

    @pytest.mark.asyncio
    async def test_async_context_manager_exception(self) -> None:
        tracker = LatencyTracker()
        with pytest.raises(ValueError):
            async with tracker.track("op"):
                raise ValueError("boom")
        summary = tracker.get_summary()
        assert summary.total_ms >= 0.0


# ------------------------------------------------------------------ #
#  Dataclass smoke tests
# ------------------------------------------------------------------ #


class TestGenerationMetrics:
    def test_defaults(self) -> None:
        m = GenerationMetrics()
        assert m.faithfulness == 0.0
        assert m.relevance == 0.0
        assert m.completeness == 0.0
        assert m.hallucination_rate == 0.0

    def test_custom(self) -> None:
        m = GenerationMetrics(faithfulness=0.9, relevance=0.8, completeness=0.7, hallucination_rate=0.1)
        assert m.faithfulness == 0.9


class TestGenerationResult:
    def test_construction(self) -> None:
        r = GenerationResult(
            query_id="q1",
            query="what?",
            answer="42",
            expected_answer="42",
            context_used="context",
        )
        assert r.query_id == "q1"
        assert isinstance(r.metrics, GenerationMetrics)
