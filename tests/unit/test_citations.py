"""Tests for citation extraction, mapping, validation, and formatting."""

from __future__ import annotations

from rag_pipeline.data.chunking import Chunk
from rag_pipeline.generation.citations import (
    Citation,
    CitationBundle,
    CitationExtractor,
    CitationFormatter,
    CitationManager,
    CitationMapper,
    CitationValidator,
)


def _make_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"Content about topic {i}.",
            index=i,
            token_count=10,
            metadata={"score": 0.9 - i * 0.1, "page": i + 1},
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------ #
#  CitationBundle dataclass
# ------------------------------------------------------------------ #


class TestCitationBundle:
    def test_default_fields(self) -> None:
        bundle = CitationBundle()
        assert bundle.citations == []
        assert bundle.formatted_sources == ""
        assert bundle.validation_warnings == []

    def test_populated_fields(self) -> None:
        cit = Citation(
            marker="[1]",
            source_document="doc-a",
            chunk_id="c1",
            chunk_index=0,
            page=1,
            score=0.95,
            text_snippet="snippet",
        )
        bundle = CitationBundle(
            citations=[cit],
            formatted_sources="[1] doc-a",
            validation_warnings=[],
        )
        assert len(bundle.citations) == 1
        assert bundle.citations[0].marker == "[1]"


# ------------------------------------------------------------------ #
#  CitationExtractor
# ------------------------------------------------------------------ #


class TestCitationExtractor:
    def test_single_citation(self) -> None:
        ext = CitationExtractor()
        assert ext.extract("The answer is [1] correct.") == ["1"]

    def test_multiple_citations(self) -> None:
        ext = CitationExtractor()
        result = ext.extract("This is [1] supported by [2] and [3].")
        assert result == ["1", "2", "3"]

    def test_no_citations(self) -> None:
        ext = CitationExtractor()
        assert ext.extract("No citations here.") == []

    def test_duplicate_citations_deduplicated(self) -> None:
        ext = CitationExtractor()
        result = ext.extract("See [1] and also [1] again.")
        assert result == ["1"]

    def test_preserves_order(self) -> None:
        ext = CitationExtractor()
        result = ext.extract("First [3] then [1] then [2].")
        assert result == ["3", "1", "2"]

    def test_empty_string(self) -> None:
        ext = CitationExtractor()
        assert ext.extract("") == []

    def test_large_numbers(self) -> None:
        ext = CitationExtractor()
        result = ext.extract("Reference [100] and [999].")
        assert result == ["100", "999"]


# ------------------------------------------------------------------ #
#  CitationMapper
# ------------------------------------------------------------------ #


class TestCitationMapper:
    def test_maps_markers_to_chunks(self) -> None:
        chunks = _make_chunks(3)
        mapper = CitationMapper()
        citations = mapper.map_citations(["1", "2"], chunks)
        assert len(citations) == 2
        assert citations[0].marker == "[1]"
        assert citations[0].source_document == "doc-0"
        assert citations[1].marker == "[2]"
        assert citations[1].source_document == "doc-1"

    def test_skips_out_of_range(self) -> None:
        chunks = _make_chunks(2)
        mapper = CitationMapper()
        citations = mapper.map_citations(["1", "5"], chunks)
        assert len(citations) == 1
        assert citations[0].marker == "[1]"

    def test_skips_invalid_markers(self) -> None:
        chunks = _make_chunks(2)
        mapper = CitationMapper()
        citations = mapper.map_citations(["abc", "1"], chunks)
        assert len(citations) == 1

    def test_includes_score_from_metadata(self) -> None:
        chunks = _make_chunks(1)
        mapper = CitationMapper()
        citations = mapper.map_citations(["1"], chunks)
        assert citations[0].score == 0.9

    def test_includes_page_from_metadata(self) -> None:
        chunks = _make_chunks(1)
        mapper = CitationMapper()
        citations = mapper.map_citations(["1"], chunks)
        assert citations[0].page == 1

    def test_empty_chunks(self) -> None:
        mapper = CitationMapper()
        citations = mapper.map_citations(["1"], [])
        assert citations == []


# ------------------------------------------------------------------ #
#  CitationValidator
# ------------------------------------------------------------------ #


class TestCitationValidator:
    def test_no_warnings_valid(self) -> None:
        validator = CitationValidator()
        chunks = _make_chunks(2)
        mapper = CitationMapper()
        citations = mapper.map_citations(["1", "2"], chunks)
        warnings = validator.validate("Answer [1] and [2].", citations)
        assert warnings == []

    def test_hallucinated_marker(self) -> None:
        validator = CitationValidator()
        citations = [
            Citation(
                marker="[1]",
                source_document="doc",
                chunk_id="c1",
                chunk_index=0,
                page=None,
                score=0.9,
                text_snippet="text",
            )
        ]
        warnings = validator.validate("Answer [1] and [5].", citations)
        assert any("[5]" in w for w in warnings)

    def test_missing_marker_in_answer(self) -> None:
        validator = CitationValidator()
        citations = [
            Citation(
                marker="[1]",
                source_document="doc",
                chunk_id="c1",
                chunk_index=0,
                page=None,
                score=0.9,
                text_snippet="text",
            ),
            Citation(
                marker="[2]",
                source_document="doc2",
                chunk_id="c2",
                chunk_index=1,
                page=None,
                score=0.8,
                text_snippet="text2",
            ),
        ]
        warnings = validator.validate("Answer [1] only.", citations)
        assert any("[2]" in w for w in warnings)


# ------------------------------------------------------------------ #
#  CitationFormatter
# ------------------------------------------------------------------ #


class TestCitationFormatter:
    def _citations(self) -> list[Citation]:
        return [
            Citation("[1]", "Paper A", "c1", 0, 1, 0.95, "First snippet"),
            Citation("[2]", "Paper B", "c2", 1, None, 0.80, "Second snippet"),
        ]

    def test_inline_format(self) -> None:
        fmt = CitationFormatter()
        result = fmt.format_inline(self._citations())
        assert result == "[1] Paper A, [2] Paper B"

    def test_footnote_format(self) -> None:
        fmt = CitationFormatter()
        result = fmt.format_footnote(self._citations())
        assert "1. Paper A" in result
        assert "2. Paper B" in result
        assert "p. 1" in result
        assert "score 0.95" in result

    def test_footnote_no_page(self) -> None:
        fmt = CitationFormatter()
        result = fmt.format_footnote(self._citations())
        assert "p." not in result.split("\n")[1]

    def test_source_block_format(self) -> None:
        fmt = CitationFormatter()
        result = fmt.format_source_block(self._citations())
        assert "**Sources:**" in result
        assert "[1]" in result
        assert "First snippet" in result

    def test_source_block_empty(self) -> None:
        fmt = CitationFormatter()
        assert fmt.format_source_block([]) == ""


# ------------------------------------------------------------------ #
#  CitationManager
# ------------------------------------------------------------------ #


class TestCitationManager:
    def test_end_to_end(self) -> None:
        chunks = _make_chunks(3)
        manager = CitationManager()
        bundle = manager.process("The answer is [1] supported by [2].", chunks)
        assert isinstance(bundle, CitationBundle)
        assert len(bundle.citations) == 2
        assert bundle.citations[0].marker == "[1]"
        assert bundle.citations[1].marker == "[2]"
        assert "[1] doc-0" in bundle.formatted_sources
        assert "[2] doc-1" in bundle.formatted_sources
        assert bundle.validation_warnings == []

    def test_with_hallucination(self) -> None:
        chunks = _make_chunks(2)
        manager = CitationManager()
        bundle = manager.process("This is [1] and [99].", chunks)
        assert len(bundle.citations) == 1
        assert any("[99]" in w for w in bundle.validation_warnings)

    def test_custom_formatter(self) -> None:
        chunks = _make_chunks(2)
        fmt = CitationFormatter()
        manager = CitationManager(formatter=fmt)
        assert manager.formatter is fmt
        bundle = manager.process("[1] and [2].", chunks)
        assert "Sources" in bundle.formatted_sources or "[1]" in bundle.formatted_sources

    def test_empty_answer(self) -> None:
        chunks = _make_chunks(2)
        manager = CitationManager()
        bundle = manager.process("No citations.", chunks)
        assert bundle.citations == []
        assert bundle.validation_warnings == []
