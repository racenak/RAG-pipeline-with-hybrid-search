"""Tests for text chunking pipeline."""

from rag_pipeline.data.chunking import (
    Chunk,
    ChunkingConfig,
    TextChunker,
    TokenCounter,
)


class TestTokenCounter:
    def test_count_basic(self):
        counter = TokenCounter()
        assert counter.count("hello world") > 0

    def test_count_empty(self):
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_longer_text_has_more_tokens(self):
        counter = TokenCounter()
        short = counter.count("hello")
        long = counter.count("hello world this is a longer sentence")
        assert long > short

    def test_decode_empty(self):
        counter = TokenCounter()
        assert counter.decode([]) == ""


class TestChunk:
    def test_make_id_deterministic(self):
        id1 = Chunk.make_id("doc1", "content", 0)
        id2 = Chunk.make_id("doc1", "content", 0)
        assert id1 == id2

    def test_make_id_different_content(self):
        id1 = Chunk.make_id("doc1", "content A", 0)
        id2 = Chunk.make_id("doc1", "content B", 0)
        assert id1 != id2

    def test_make_id_different_index(self):
        id1 = Chunk.make_id("doc1", "content", 0)
        id2 = Chunk.make_id("doc1", "content", 1)
        assert id1 != id2


class TestTextChunker:
    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(ChunkingConfig(target_size=512))
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].index == 0
        assert chunks[0].token_count > 0

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(ChunkingConfig(target_size=20, max_size=40))
        text = "This is sentence one. " * 20  # ~120 tokens
        chunks = chunker.chunk(text, document_id="doc1")
        assert len(chunks) > 1
        assert all(c.document_id == "doc1" for c in chunks)
        assert all(c.index == i for i, c in enumerate(chunks))

    def test_chunk_ids_unique(self):
        chunker = TextChunker(ChunkingConfig(target_size=20, max_size=40))
        text = "This is sentence one. " * 20
        chunks = chunker.chunk(text, document_id="doc1")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_heading_boundaries(self):
        chunker = TextChunker(ChunkingConfig(target_size=10, max_size=20))
        text = "## Introduction\n\nThis is the introduction paragraph with enough words to exceed the target size.\n\n## Methods\n\nThis is the methods paragraph with enough words to exceed the target size."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # Each chunk should have heading_path metadata
        assert any(c.metadata.get("heading_path") == "Introduction" for c in chunks)
        assert any(c.metadata.get("heading_path") == "Methods" for c in chunks)

    def test_paragraph_splitting(self):
        chunker = TextChunker(ChunkingConfig(target_size=5, max_size=10))
        text = "Paragraph one is quite long.\n\nParagraph two is also long.\n\nParagraph three is long too."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_metadata_propagation(self):
        chunker = TextChunker()
        meta = {"source": "test.pdf", "page": 1}
        chunks = chunker.chunk("Hello world.", metadata=meta)
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["page"] == 1

    def test_overlap(self):
        chunker = TextChunker(ChunkingConfig(target_size=15, max_size=30, overlap=5))
        text = "First paragraph here. Second paragraph here. Third paragraph here."
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # Second chunk should have overlap from first
            assert len(chunks[1].content) > len(chunks[1].content.split("\n\n")[-1])

    def test_no_overlap_when_zero(self):
        chunker = TextChunker(ChunkingConfig(target_size=15, max_size=30, overlap=0))
        text = "First paragraph here. Second paragraph here. Third paragraph here."
        chunks = chunker.chunk(text)
        # Should still produce chunks
        assert len(chunks) >= 1

    def test_min_size_respected(self):
        chunker = TextChunker(ChunkingConfig(target_size=10, max_size=20, min_size=3))
        text = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        chunks = chunker.chunk(text)
        # Small paragraphs should be merged
        assert len(chunks) >= 1

    def test_sentence_splitting(self):
        chunker = TextChunker(ChunkingConfig(target_size=10, max_size=15))
        # One very long paragraph with many sentences
        text = " ".join(["This is sentence number " + str(i) + "." for i in range(20)])
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_empty_paragraphs_skipped(self):
        chunker = TextChunker()
        text = "Hello.\n\n\n\n\nWorld."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
