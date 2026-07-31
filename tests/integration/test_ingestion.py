"""Integration tests — ingestion pipeline."""



class TestIngestionPipeline:
    """Test the full ingestion pipeline with mocked dependencies."""

    def test_parse_txt_file(self, sample_txt_path):
        """Test parsing a TXT file."""
        from rag_pipeline.data.parsers import get_parser

        parser = get_parser(sample_txt_path)
        doc = parser.parse(sample_txt_path)
        assert doc is not None
        assert len(doc.content) > 0
        assert doc.metadata.get("format") == "txt"

    def test_parse_md_file(self, sample_md_path):
        """Test parsing a Markdown file."""
        from rag_pipeline.data.parsers import get_parser

        parser = get_parser(sample_md_path)
        doc = parser.parse(sample_md_path)
        assert doc is not None
        assert len(doc.content) > 0
        assert doc.metadata.get("format") == "markdown"
        assert len(doc.sections) > 0

    def test_parse_html_file(self, sample_html_path):
        """Test parsing an HTML file."""
        from rag_pipeline.data.parsers import get_parser

        parser = get_parser(sample_html_path)
        doc = parser.parse(sample_html_path)
        assert doc is not None
        assert len(doc.content) > 0
        assert "skip this navigation" not in doc.content.lower()
        assert "skip this footer" not in doc.content.lower()

    def test_parse_csv_file(self, sample_csv_path):
        """Test parsing a CSV file."""
        from rag_pipeline.data.parsers import get_parser

        parser = get_parser(sample_csv_path)
        doc = parser.parse(sample_csv_path)
        assert doc is not None
        assert len(doc.content) > 0
        assert doc.metadata.get("format") == "csv"
        assert doc.metadata.get("row_count") == 5

    def test_clean_text(self):
        """Test text cleaning pipeline."""
        from rag_pipeline.data.cleaning import TextCleaner

        cleaner = TextCleaner()
        dirty = "  Hello   world  \x00\x01  with   spaces  "
        clean, stats = cleaner.clean(dirty)
        assert "\x00" not in clean
        assert "\x01" not in clean
        assert "  " not in clean
        assert stats.chars_before > 0
        assert stats.chars_after > 0

    def test_chunk_text(self):
        """Test text chunking."""
        from rag_pipeline.data.chunking import ChunkingConfig, TextChunker

        config = ChunkingConfig(target_size=100, max_size=200, min_size=20, overlap=10)
        chunker = TextChunker(config)

        text = "This is a test document. " * 20  # ~500 chars
        chunks = chunker.chunk(text, document_id="test-doc")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == "test-doc"
            assert len(chunk.content) > 0
            assert chunk.token_count > 0

    def test_ingest_txt_file(self, sample_txt_path):
        """Test ingesting a TXT file through the full pipeline."""
        from rag_pipeline.pipeline import ingest_file_full

        # Pipeline handles missing services gracefully
        result = ingest_file_full(sample_txt_path)
        assert result is not None
        assert result.source == str(sample_txt_path)
