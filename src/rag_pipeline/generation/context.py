"""Context builder — assemble retrieved chunks into a prompt-ready context string."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from rag_pipeline.data.chunking import Chunk


@dataclass
class ContextConfig:
    """Configuration for context assembly."""

    max_tokens: int = 4096
    separator: str = "xml"
    order_by: str = "score"


_ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return len(_ENCODING.encode(text))


class ContextBuilder:
    """Builds a context string from retrieved chunks within a token budget."""

    def build_context(self, chunks: list[Chunk], config: ContextConfig | None = None) -> str:
        """Assemble chunks into a single context string.

        Deduplicates by content hash, orders by the configured strategy,
        and truncates to fit within max_tokens.
        """
        cfg = config or ContextConfig()
        if not chunks:
            return ""

        # Deduplicate by content hash
        seen: set[str] = set()
        unique: list[Chunk] = []
        for chunk in chunks:
            h = hashlib.sha256(chunk.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)

        # Order chunks
        ordered = self._order_chunks(unique, cfg.order_by)

        # Build context within token budget
        return self._assemble(ordered, cfg)

    def _order_chunks(self, chunks: list[Chunk], order_by: str) -> list[Chunk]:
        if order_by == "score":
            return sorted(chunks, key=lambda c: c.metadata.get("score", 0.0), reverse=True)
        if order_by == "position":
            return sorted(chunks, key=lambda c: c.index)
        return chunks

    def _assemble(self, chunks: list[Chunk], config: ContextConfig) -> str:
        if not chunks:
            return ""

        parts: list[str] = []
        current_tokens = 0
        budget = config.max_tokens

        # Reserve tokens for separators/overhead
        overhead = estimate_tokens("<context></context>") + 20
        available = budget - overhead
        if available <= 0:
            return ""

        for chunk in chunks:
            chunk_text = self._format_chunk(chunk, len(parts) + 1, config.separator)
            chunk_tokens = estimate_tokens(chunk_text)

            if current_tokens + chunk_tokens > available:
                # Try partial fit for last chunk
                if not parts:
                    remaining_tokens = available
                    truncated = self._truncate_to_tokens(chunk.content, remaining_tokens - 10)
                    if truncated:
                        parts.append(self._format_chunk_text(truncated, 1, config.separator))
                break

            parts.append(chunk_text)
            current_tokens += chunk_tokens

        if not parts:
            return ""

        return self._wrap_context(parts, config.separator)

    def _format_chunk(self, chunk: Chunk, idx: int, separator: str) -> str:
        return self._format_chunk_text(chunk.content, idx, separator)

    def _format_chunk_text(self, text: str, idx: int, separator: str) -> str:
        if separator == "xml":
            return f'<chunk id="{idx}">\n{text}\n</chunk>'
        if separator == "markdown":
            return f"### Source [{idx}]\n\n{text}"
        return f"[{idx}] {text}"

    def _wrap_context(self, parts: list[str], separator: str) -> str:
        body = "\n\n".join(parts)
        if separator == "xml":
            return f"<context>\n{body}\n</context>"
        return body

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _ENCODING.decode(tokens[:max_tokens])
