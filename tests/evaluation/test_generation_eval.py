"""Tests for generation evaluation framework."""

import pytest

from rag_pipeline.evaluation.generation import (
    _heuristic_completeness,
    _heuristic_faithfulness,
    _heuristic_relevance,
    bleu,
    detect_hallucination,
    rouge_1,
    rouge_l,
    word_overlap,
)


class TestROUGE1:
    def test_identical_texts(self):
        assert rouge_1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_completely_different(self):
        score = rouge_1("hello world", "foo bar baz")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = rouge_1("the cat sat on the mat", "the cat is on the mat")
        assert 0 < score < 1


class TestROUGEL:
    def test_identical(self):
        assert rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_different(self):
        score = rouge_l("abc", "xyz")
        assert score == pytest.approx(0.0)


class TestBLEU:
    def test_identical(self):
        score = bleu("the cat sat", "the cat sat")
        assert score == pytest.approx(1.0)

    def test_different(self):
        score = bleu("hello world", "foo bar baz")
        assert score < 0.1


class TestWordOverlap:
    def test_identical(self):
        assert word_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert word_overlap("abc", "xyz") == pytest.approx(0.0)


class TestHallucination:
    def test_no_hallucination(self):
        answer = "The dimension is 1024"
        context = "The embedding dimension is 1024 for the model"
        rate, _sentences = detect_hallucination(answer, context)
        assert rate < 0.5

    def test_detects_hallucination(self):
        answer = "The model uses quantum computing"
        context = "The embedding dimension is 1024"
        rate, _sentences = detect_hallucination(answer, context)
        assert rate > 0.0


class TestHeuristics:
    def test_faithfulness_high(self):
        context = "The embedding dimension is 1024"
        answer = "The dimension is 1024"
        assert _heuristic_faithfulness(answer, context) > 0.5

    def test_faithfulness_low(self):
        context = "The embedding dimension is 1024"
        answer = "Quantum computing is the future"
        assert _heuristic_faithfulness(answer, context) < 0.3

    def test_relevance_high(self):
        query = "embedding dimension"
        answer = "The embedding dimension is 1024"
        assert _heuristic_relevance(query, answer) > 0.3

    def test_completeness(self):
        query = "what is RAG"
        answer = "RAG is retrieval augmented generation"
        expected = "RAG is retrieval augmented generation for LLMs"
        score = _heuristic_completeness(query, answer, expected)
        assert score > 0.5
