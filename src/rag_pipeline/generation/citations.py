"""Citation extraction, mapping, validation, and formatting for RAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.data.chunking import Chunk

_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    """A single citation linking an answer marker to its source chunk."""

    marker: str
    source_document: str
    chunk_id: str
    chunk_index: int
    page: int | None
    score: float
    text_snippet: str


@dataclass
class CitationBundle:
    """Collected citations with formatted output and validation info."""

    citations: list[Citation] = field(default_factory=list)
    formatted_sources: str = ""
    validation_warnings: list[str] = field(default_factory=list)


class CitationExtractor:
    """Extract [N] citation markers from answer text."""

    def extract(self, answer_text: str) -> list[str]:
        """Return deduplicated citation marker numbers in order of appearance."""
        matches = _MARKER_RE.findall(answer_text)
        seen: set[str] = set()
        ordered: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return ordered


class CitationMapper:
    """Map citation markers to source chunks."""

    def map_citations(self, markers: list[str], chunks: list[Chunk]) -> list[Citation]:
        """Map marker numbers to citations.

        Chunks are expected in relevance order (index 0 = most relevant = [1]).
        """
        citations: list[Citation] = []
        for marker in markers:
            try:
                idx = int(marker) - 1
            except (ValueError, TypeError):
                continue
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            score = chunk.metadata.get("score", 0.0)
            page = chunk.metadata.get("page")
            citations.append(
                Citation(
                    marker=f"[{marker}]",
                    source_document=chunk.document_id,
                    chunk_id=chunk.id,
                    chunk_index=chunk.index,
                    page=page if isinstance(page, int) else None,
                    score=float(score),
                    text_snippet=chunk.content[:200],
                )
            )
        return citations


class CitationValidator:
    """Validate citations against the answer text."""

    def validate(self, answer_text: str, citations: list[Citation]) -> list[str]:
        """Return warnings for hallucinated or missing citations."""
        warnings: list[str] = []
        present = {c.marker for c in citations}

        for marker in _MARKER_RE.findall(answer_text):
            tag = f"[{marker}]"
            if tag not in present:
                warnings.append(f"citation {tag} not found in context")

        used = {int(c.marker[1:-1]) for c in citations}
        all_nums = {int(m) for m in _MARKER_RE.findall(answer_text)}
        for num in used - all_nums:
            warnings.append(f"marker [{num}] in citations but not in answer text")

        return warnings


class CitationFormatter:
    """Format citation lists for display."""

    def format_inline(self, citations: list[Citation]) -> str:
        """Inline format: [1] Doc A, [2] Doc B."""
        parts = [f"{c.marker} {c.source_document}" for c in citations]
        return ", ".join(parts)

    def format_footnote(self, citations: list[Citation]) -> str:
        """Numbered footnote list with full source details."""
        lines: list[str] = []
        for i, c in enumerate(citations, 1):
            loc = f", p. {c.page}" if c.page is not None else ""
            lines.append(f"{i}. {c.source_document} (chunk {c.chunk_index}{loc}, score {c.score:.2f})")
        return "\n".join(lines)

    def format_source_block(self, citations: list[Citation]) -> str:
        """Markdown source block with chunk snippets."""
        if not citations:
            return ""
        parts = ["**Sources:**\n"]
        for c in citations:
            loc = f", p. {c.page}" if c.page is not None else ""
            parts.append(f"{c.marker} *{c.source_document}*{loc} — {c.text_snippet}")
        return "\n".join(parts)


class CitationManager:
    """Orchestrates citation extraction, mapping, validation, and formatting."""

    def __init__(self, formatter: CitationFormatter | None = None) -> None:
        self._extractor = CitationExtractor()
        self._mapper = CitationMapper()
        self._validator = CitationValidator()
        self._formatter = formatter or CitationFormatter()

    @property
    def formatter(self) -> CitationFormatter:
        return self._formatter

    def process(self, answer: str, chunks: list[Chunk]) -> CitationBundle:
        """Extract markers, map to chunks, validate, and format."""
        markers = self._extractor.extract(answer)
        citations = self._mapper.map_citations(markers, chunks)
        warnings = self._validator.validate(answer, citations)
        formatted = self._formatter.format_inline(citations)
        return CitationBundle(
            citations=citations,
            formatted_sources=formatted,
            validation_warnings=warnings,
        )
