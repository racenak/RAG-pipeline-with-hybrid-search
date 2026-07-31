"""Semantic text chunking — split documents into sized, coherent chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

# ---- Token counting ----------------------------------------------------------


class TokenCounter:
    """Count tokens using tiktoken with whitespace fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Whitespace fallback: ~4 chars per token
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        if self._encoder is not None:
            return self._encoder.encode(text)
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        if self._encoder is not None:
            return self._encoder.decode(tokens)
        return ""


# ---- Data model --------------------------------------------------------------


@dataclass
class Chunk:
    """A text chunk ready for embedding."""

    id: str
    document_id: str
    content: str
    index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @staticmethod
    def make_id(document_id: str, content: str, index: int) -> str:
        raw = f"{document_id}:{index}:{content}"
        return hashlib.sha256(raw.encode()).hexdigest()


# ---- Config ------------------------------------------------------------------


@dataclass
class ChunkingConfig:
    """Chunking parameters."""

    target_size: int = 512
    max_size: int = 1024
    min_size: int = 100
    overlap: int = 50
    encoding_name: str = "cl100k_base"


# ---- Chunker -----------------------------------------------------------------

# Heading pattern: ## Title, ### Title, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Sentence boundary: period/exclamation/question + space + uppercase
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class TextChunker:
    """Semantic text chunker — splits by headings, paragraphs, sentences."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._counter = TokenCounter(self.config.encoding_name)

    def chunk(
        self,
        text: str,
        document_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Returns a list of Chunk objects with content, token counts, and metadata.
        """
        if not text.strip():
            return []

        meta = metadata or {}
        sections = self._split_by_headings(text)
        raw_chunks: list[tuple[str, dict[str, Any]]] = []

        for section_text, heading_path in sections:
            paragraphs = self._split_by_paragraphs(section_text)
            for para in paragraphs:
                para_meta = {**meta, "heading_path": heading_path}
                if self._counter.count(para) <= self.config.max_size:
                    raw_chunks.append((para, para_meta))
                else:
                    # Split oversized paragraph by sentences
                    for sent_chunk in self._split_by_sentences(para):
                        raw_chunks.append((sent_chunk, para_meta))

        # Merge small chunks and apply overlap
        merged = self._merge_and_overlap(raw_chunks)

        # Build final Chunk objects
        chunks = []
        for i, (content, chunk_meta) in enumerate(merged):
            token_count = self._counter.count(content)
            chunk_id = Chunk.make_id(document_id, content, i)
            chunks.append(Chunk(
                id=chunk_id,
                document_id=document_id,
                content=content,
                index=i,
                token_count=token_count,
                metadata=chunk_meta,
            ))

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split text by heading markers, returning (section_text, heading_path)."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(text, "")]

        sections: list[tuple[str, str]] = []

        # Text before first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((preamble, ""))

        for i, match in enumerate(matches):
            title = match.group(2)
            heading_path = title

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                sections.append((section_text, heading_path))

        return sections

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by double newlines into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split oversized text by sentence boundaries, respecting max_size."""
        sentences = _SENTENCE_RE.split(text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if self._counter.count(candidate) <= self.config.max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    def _merge_and_overlap(
        self,
        raw_chunks: list[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Merge small chunks and apply overlap between adjacent chunks."""
        if not raw_chunks:
            return []

        # Step 1: Merge small chunks
        merged: list[tuple[str, dict[str, Any]]] = []
        current_text, current_meta = raw_chunks[0]

        for text, meta in raw_chunks[1:]:
            combined = f"{current_text}\n\n{text}"
            if self._counter.count(combined) <= self.config.target_size:
                current_text = combined
            else:
                merged.append((current_text, current_meta))
                current_text, current_meta = text, meta

        merged.append((current_text, current_meta))

        # Step 2: Apply overlap
        if self.config.overlap <= 0 or len(merged) <= 1:
            return merged

        overlapped: list[tuple[str, dict[str, Any]]] = [merged[0]]

        for i in range(1, len(merged)):
            prev_text = merged[i - 1][0]
            curr_text = merged[i][0]
            curr_meta = merged[i][1]

            # Get overlap tokens from end of previous chunk
            overlap_text = self._get_overlap_tail(prev_text)
            combined = f"{overlap_text}\n\n{curr_text}" if overlap_text else curr_text

            overlapped.append((combined, curr_meta))

        return overlapped

    def _get_overlap_tail(self, text: str) -> str:
        """Get the last `overlap` tokens worth of text."""
        if self.config.overlap <= 0:
            return ""

        tokens = self._counter.encode(text)
        if len(tokens) <= self.config.overlap:
            return text

        tail_tokens = tokens[-self.config.overlap :]
        return self._counter.decode(tail_tokens)
