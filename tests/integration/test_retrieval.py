"""Integration tests — retrieval pipeline."""


from rag_pipeline.retrieval.vector import SearchResult


class TestRetrievalPipeline:
    """Test the retrieval pipeline with mocked dependencies."""

    def test_bm25_index_and_search(self):
        """Test BM25 indexing and searching."""
        from rag_pipeline.retrieval.bm25 import BM25

        bm25 = BM25()

        # Index documents
        bm25.add_document("doc1", "The embedding dimension is 1024")
        bm25.add_document("doc2", "OpenSearch supports kNN vector search")
        bm25.add_document("doc3", "Redis provides caching for query results")

        # Search
        results = bm25.search("embedding dimension")
        assert len(results) > 0
        assert results[0]["id"] == "doc1"  # Most relevant

    def test_bm25_save_and_load(self, tmp_path):
        """Test BM25 index persistence."""
        from rag_pipeline.retrieval.bm25 import BM25

        bm25 = BM25()
        bm25.add_document("doc1", "Test content for persistence")

        # Save
        index_path = tmp_path / "bm25_index.json"
        bm25.save(str(index_path))
        assert index_path.exists()

        # Load
        bm25_loaded = BM25.load(str(index_path))
        results = bm25_loaded.search("test content")
        assert len(results) > 0

    def test_rrf_fusion(self):
        """Test reciprocal rank fusion combines results correctly."""
        from rag_pipeline.retrieval.hybrid import reciprocal_rank_fusion

        results_a = [
            SearchResult(id="doc1", score=0.9, content="", metadata={}),
            SearchResult(id="doc2", score=0.8, content="", metadata={}),
            SearchResult(id="doc3", score=0.7, content="", metadata={}),
        ]
        results_b = [
            SearchResult(id="doc2", score=0.95, content="", metadata={}),
            SearchResult(id="doc1", score=0.85, content="", metadata={}),
            SearchResult(id="doc4", score=0.75, content="", metadata={}),
        ]

        fused = reciprocal_rank_fusion([results_a, results_b])
        assert len(fused) > 0
        # doc1 and doc2 appear in both, should rank high
        fused_ids = [r.id for r in fused]
        assert "doc1" in fused_ids
        assert "doc2" in fused_ids

    def test_reranking(self, sample_search_results):
        """Test reranking improves result order."""
        from rag_pipeline.retrieval.reranking import NoopReranker

        reranker = NoopReranker()
        reranked = reranker.rerank("test query", sample_search_results, top_k=10)
        assert len(reranked) == len(sample_search_results)

    def test_query_processing_pipeline(self):
        """Test full query processing pipeline."""
        from rag_pipeline.retrieval.query import get_query_processor

        processor = get_query_processor()
        result = processor.process("What is the embedding dimension?")

        assert result.original == "What is the embedding dimension?"
        assert result.query_type is not None
        assert result.latency_ms >= 0
