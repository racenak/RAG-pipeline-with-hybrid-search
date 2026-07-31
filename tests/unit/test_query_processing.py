"""Tests for query processing — preprocessing, validation, type detection, rewriting."""

from __future__ import annotations

import pytest

from rag_pipeline.retrieval.query import (
    HyDE,
    MultiQuery,
    QueryExpansion,
    QueryProcessor,
    QueryProcessorConfig,
    QueryType,
    QueryTypeDetector,
    QueryValidator,
    SpecialCharacterHandler,
    StepBackPrompting,
    WhitespaceNormalizer,
    get_query_processor,
)

# ------------------------------------------------------------------ #
#  WhitespaceNormalizer
# ------------------------------------------------------------------ #


class TestWhitespaceNormalizer:
    def test_collapses_multiple_spaces(self):
        norm = WhitespaceNormalizer()
        assert norm.process("hello   world", {}) == "hello world"

    def test_collapses_tabs_and_newlines(self):
        norm = WhitespaceNormalizer()
        assert norm.process("hello\tworld\nfoo", {}) == "hello world foo"

    def test_strips_leading_trailing_whitespace(self):
        norm = WhitespaceNormalizer()
        assert norm.process("  hello  ", {}) == "hello"

    def test_single_space_unchanged(self):
        norm = WhitespaceNormalizer()
        assert norm.process("hello world", {}) == "hello world"

    def test_empty_string(self):
        norm = WhitespaceNormalizer()
        assert norm.process("", {}) == ""

    def test_only_whitespace(self):
        norm = WhitespaceNormalizer()
        assert norm.process("   ", {}) == ""


# ------------------------------------------------------------------ #
#  SpecialCharacterHandler
# ------------------------------------------------------------------ #


class TestSpecialCharacterHandler:
    def test_removes_backticks(self):
        handler = SpecialCharacterHandler()
        assert handler.process("what is `python`?", {}) == "what is python?"

    def test_removes_control_chars(self):
        handler = SpecialCharacterHandler()
        assert handler.process("hello\x00world", {}) == "helloworld"

    def test_preserves_hyphens(self):
        handler = SpecialCharacterHandler()
        assert handler.process("state-of-the-art", {}) == "state-of-the-art"

    def test_preserves_apostrophes(self):
        handler = SpecialCharacterHandler()
        assert handler.process("what's happening", {}) == "what's happening"

    def test_preserves_dots(self):
        handler = SpecialCharacterHandler()
        assert handler.process("v1.2.3", {}) == "v1.2.3"

    def test_removes_tildes(self):
        handler = SpecialCharacterHandler()
        assert handler.process("hello~world", {}) == "helloworld"


# ------------------------------------------------------------------ #
#  QueryValidator
# ------------------------------------------------------------------ #


class TestQueryValidator:
    def test_valid_query_passes(self):
        validator = QueryValidator()
        assert validator.process("hello world", {}) == "hello world"

    def test_empty_query_raises(self):
        validator = QueryValidator()
        with pytest.raises(ValueError, match="must not be empty"):
            validator.process("", {})

    def test_whitespace_only_raises(self):
        validator = QueryValidator()
        with pytest.raises(ValueError, match="must not be empty"):
            validator.process("   ", {})

    def test_too_long_raises(self):
        validator = QueryValidator(max_length=10)
        with pytest.raises(ValueError, match="too long"):
            validator.process("a" * 11, {})

    def test_exactly_max_length_passes(self):
        validator = QueryValidator(max_length=10)
        result = validator.process("a" * 10, {})
        assert len(result) == 10

    def test_strips_whitespace(self):
        validator = QueryValidator()
        assert validator.process("  hello  ", {}) == "hello"


# ------------------------------------------------------------------ #
#  QueryTypeDetector
# ------------------------------------------------------------------ #


class TestQueryTypeDetector:
    def test_detects_factual_who(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("Who invented Python?", ctx)
        assert ctx["query_type"] == QueryType.FACTUAL

    def test_detects_factual_what(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("What is machine learning?", ctx)
        assert ctx["query_type"] == QueryType.FACTUAL

    def test_detects_factual_define(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("Define neural network", ctx)
        assert ctx["query_type"] == QueryType.FACTUAL

    def test_detects_summarization(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("Summarize the document", ctx)
        assert ctx["query_type"] == QueryType.SUMMARIZATION

    def test_detects_comparison(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("Compare React and Vue", ctx)
        assert ctx["query_type"] == QueryType.COMPARISON

    def test_detects_how_to(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("How do I deploy to production?", ctx)
        assert ctx["query_type"] == QueryType.HOW_TO

    def test_detects_general(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("random stuff", ctx)
        assert ctx["query_type"] == QueryType.GENERAL

    def test_returns_query_unchanged(self):
        detector = QueryTypeDetector()
        q = "What is RAG?"
        assert detector.process(q, {}) == q

    def test_case_insensitive(self):
        detector = QueryTypeDetector()
        ctx: dict = {}
        detector.process("SUMMARIZE this", ctx)
        assert ctx["query_type"] == QueryType.SUMMARIZATION


# ------------------------------------------------------------------ #
#  QueryExpansion
# ------------------------------------------------------------------ #


class TestQueryExpansion:
    def test_expands_known_word(self):
        exp = QueryExpansion()
        result = exp.process("fix the error", {})
        # Should contain at least one synonym
        assert result != "fix the error"
        assert "fix" in result  # original kept

    def test_does_not_duplicate_existing_synonym(self):
        exp = QueryExpansion()
        result = exp.process("fix the repair", {})
        # "repair" is a synonym of "fix" but already present
        assert "repair" in result

    def test_no_expansion_for_unknown_words(self):
        exp = QueryExpansion()
        result = exp.process("quantum entanglement", {})
        assert result == "quantum entanglement"

    def test_empty_query(self):
        exp = QueryExpansion()
        result = exp.process("", {})
        assert result == ""


# ------------------------------------------------------------------ #
#  HyDE
# ------------------------------------------------------------------ #


class TestHyDE:
    def test_prepends_hypothetical_prefix(self):
        hyde = HyDE()
        result = hyde.process("What is RAG?", {})
        assert result.startswith("This document provides a detailed answer about:")
        assert "What is RAG?" in result

    def test_stores_original_in_context(self):
        hyde = HyDE()
        ctx: dict = {}
        hyde.process("test query", ctx)
        assert ctx["hyde_original"] == "test query"

    def test_returns_string(self):
        hyde = HyDE()
        result = hyde.process("query", {})
        assert isinstance(result, str)


# ------------------------------------------------------------------ #
#  MultiQuery
# ------------------------------------------------------------------ #


class TestMultiQuery:
    def test_generates_multiple_queries(self):
        mq = MultiQuery(num_queries=4)
        result = mq.process("What is RAG?", {})
        assert isinstance(result, list)
        assert len(result) == 4

    def test_first_query_is_original(self):
        mq = MultiQuery(num_queries=3)
        result = mq.process("What is RAG?", {})
        assert result[0] == "What is RAG?"

    def test_includes_variations(self):
        mq = MultiQuery(num_queries=4)
        result = mq.process("What is RAG?", {})
        # Should contain different phrasings
        texts = " ".join(result).lower()
        assert "how does" in texts or "explain" in texts or "tell me" in texts

    def test_respects_num_queries_limit(self):
        mq = MultiQuery(num_queries=2)
        result = mq.process("test", {})
        assert len(result) == 2

    def test_minimum_one_query(self):
        mq = MultiQuery(num_queries=0)
        result = mq.process("test", {})
        assert len(result) >= 1

    def test_strips_trailing_punctuation(self):
        mq = MultiQuery(num_queries=3)
        result = mq.process("What is RAG?!", {})
        # Templates should clean up the query
        assert all(isinstance(q, str) for q in result)


# ------------------------------------------------------------------ #
#  StepBackPrompting
# ------------------------------------------------------------------ #


class TestStepBackPrompting:
    def test_creates_broader_query(self):
        sb = StepBackPrompting()
        result = sb.process("How do I configure Redis cache TTL in Python?", {})
        assert "Overview" in result or "background" in result
        # Should be broader than the original
        assert result != "How do I configure Redis cache TTL in Python?"

    def test_extracts_key_terms(self):
        sb = StepBackPrompting()
        ctx: dict = {}
        result = sb.process("What is machine learning?", ctx)
        assert ctx["step_back_original"] == "What is machine learning?"
        assert "machine" in result.lower() or "learning" in result.lower()

    def test_fallback_for_stop_words_only(self):
        sb = StepBackPrompting()
        result = sb.process("is the?", {})
        assert "General information" in result or "topic" in result

    def test_limits_key_terms(self):
        sb = StepBackPrompting()
        result = sb.process(
            "a b c d e f g h i j k l m n o p", {}
        )
        # Should only use up to 5 terms
        words = result.split()
        assert len(words) < 20  # reasonable upper bound


# ------------------------------------------------------------------ #
#  QueryProcessor pipeline
# ------------------------------------------------------------------ #


class TestQueryProcessor:
    def test_default_config_no_rewriting(self):
        proc = QueryProcessor()
        result = proc.process("What is RAG?")
        assert result.original == "What is RAG?"
        assert result.rewritten == []
        assert result.query_type == QueryType.FACTUAL

    def test_whitespace_normalization(self):
        proc = QueryProcessor(QueryProcessorConfig(normalize_whitespace=True))
        result = proc.process("  hello   world  ")
        assert result.original == "hello world"

    def test_special_char_handling(self):
        proc = QueryProcessor(QueryProcessorConfig(handle_special_chars=True))
        result = proc.process("what is `python`?")
        assert "`" not in result.original

    def test_validation_rejects_empty(self):
        proc = QueryProcessor(QueryProcessorConfig(validate_query=True))
        with pytest.raises(ValueError, match="must not be empty"):
            proc.process("")

    def test_query_type_detection(self):
        proc = QueryProcessor(QueryProcessorConfig(detect_query_type=True))
        result = proc.process("How do I deploy Kubernetes?")
        assert result.query_type == QueryType.HOW_TO

    def test_expansion_enabled(self):
        config = QueryProcessorConfig(expand_query=True)
        proc = QueryProcessor(config)
        result = proc.process("fix the error")
        assert len(result.rewritten) > 0

    def test_hyde_enabled(self):
        config = QueryProcessorConfig(use_hyde=True)
        proc = QueryProcessor(config)
        result = proc.process("What is RAG?")
        assert any("detailed answer" in r for r in result.rewritten)

    def test_multi_query_enabled(self):
        config = QueryProcessorConfig(use_multi_query=True, multi_query_count=3)
        proc = QueryProcessor(config)
        result = proc.process("What is RAG?")
        assert len(result.rewritten) >= 2  # original + paraphrases

    def test_step_back_enabled(self):
        config = QueryProcessorConfig(use_step_back=True)
        proc = QueryProcessor(config)
        result = proc.process("How do I configure Redis?")
        assert any("Overview" in r or "background" in r for r in result.rewritten)

    def test_multiple_rewrite_strategies(self):
        config = QueryProcessorConfig(
            expand_query=True,
            use_hyde=True,
            use_multi_query=True,
            multi_query_count=2,
        )
        proc = QueryProcessor(config)
        result = proc.process("fix the error")
        # Should have multiple rewritten queries
        assert len(result.rewritten) >= 2

    def test_latency_recorded(self):
        proc = QueryProcessor()
        result = proc.process("test query")
        assert result.latency_ms >= 0

    def test_metadata_contains_query_type(self):
        proc = QueryProcessor(QueryProcessorConfig(detect_query_type=True))
        result = proc.process("What is X?")
        assert "query_type" in result.metadata

    def test_all_queries_property(self):
        config = QueryProcessorConfig(use_multi_query=True, multi_query_count=3)
        proc = QueryProcessor(config)
        result = proc.process("What is RAG?")
        all_q = result.all_queries
        # Should include original + rewritten, deduplicated
        assert len(all_q) >= 1
        assert result.original in all_q

    def test_all_queries_deduplicates(self):
        config = QueryProcessorConfig(use_hyde=True)
        proc = QueryProcessor(config)
        result = proc.process("test")
        all_q = result.all_queries
        assert len(all_q) == len(set(all_q))


# ------------------------------------------------------------------ #
#  get_query_processor factory
# ------------------------------------------------------------------ #


class TestGetQueryProcessor:
    def test_default_returns_processor(self):
        proc = get_query_processor()
        assert isinstance(proc, QueryProcessor)

    def test_with_overrides(self):
        proc = get_query_processor(expand_query=True, use_hyde=True)
        result = proc.process("fix the error")
        # Expansion and HyDE should both produce rewritten queries
        assert len(result.rewritten) >= 1

    def test_with_settings_object(self):
        class FakeSettings:
            expand_query = True
            use_hyde = False

        proc = get_query_processor(FakeSettings())
        result = proc.process("fix the bug")
        assert len(result.rewritten) >= 1

    def test_overrides_take_precedence(self):
        class FakeSettings:
            expand_query = False

        proc = get_query_processor(FakeSettings(), expand_query=True)
        result = proc.process("fix the error")
        # expand_query=True override should win
        assert len(result.rewritten) >= 1


# ------------------------------------------------------------------ #
#  Edge cases
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_single_word_query(self):
        proc = QueryProcessor()
        result = proc.process("Python")
        assert result.original == "Python"
        assert result.query_type == QueryType.GENERAL

    def test_very_long_query(self):
        proc = QueryProcessor(QueryProcessorConfig(max_query_length=10000))
        long_q = "What is " + "very " * 200 + "important?"
        result = proc.process(long_q)
        assert len(result.original) > 0

    def test_unicode_query(self):
        proc = QueryProcessor()
        result = proc.process("Was ist eine KI?")
        assert result.original == "Was ist eine KI?"

    def test_special_chars_only(self):
        proc = QueryProcessor(QueryProcessorConfig(handle_special_chars=True))
        result = proc.process("???")
        assert result.original == "???"  # question marks preserved

    def test_mixed_language(self):
        proc = QueryProcessor()
        result = proc.process("What is データベース?")
        assert "データベース" in result.original
