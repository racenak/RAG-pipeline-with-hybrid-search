"""Tests for BM25 scoring engine and OpenSearch-backed BM25."""

from unittest.mock import MagicMock

from rag_pipeline.retrieval.bm25 import BM25, tokenize
from rag_pipeline.retrieval.bm25_index import OpenSearchBM25


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_lowercase(self):
        tokens = tokenize("FOO bar", lower=True)
        assert tokens == ["foo", "bar"]

    def test_no_lower(self):
        tokens = tokenize("FOO bar", lower=False)
        assert tokens == ["FOO", "bar"]

    def test_remove_stopwords(self):
        tokens = tokenize("the cat is on the mat", remove_stopwords=True)
        assert "the" not in tokens
        assert "cat" in tokens

    def test_stem(self):
        tokens = tokenize("running quickly", stem=True)
        assert any("run" in t for t in tokens)

    def test_alphanumeric_only(self):
        tokens = tokenize("hello, world! 123")
        assert tokens == ["hello", "world", "123"]


class TestBM25:
    def _make_index(self) -> BM25:
        bm25 = BM25(k1=1.5, b=0.75)
        bm25.add_document("d1", "the cat sat on the mat", metadata={"type": "a"})
        bm25.add_document("d2", "the dog sat on the log", metadata={"type": "b"})
        bm25.add_document("d3", "the cat chased the dog", metadata={"type": "a"})
        return bm25

    def test_add_document(self):
        bm25 = BM25()
        bm25.add_document("d1", "hello world")
        assert bm25.doc_count == 1

    def test_add_documents_bulk(self):
        bm25 = BM25()
        bm25.add_documents(
            [
                {"id": "d1", "content": "hello"},
                {"id": "d2", "content": "world"},
            ]
        )
        assert bm25.doc_count == 2

    def test_remove_document(self):
        bm25 = self._make_index()
        assert bm25.remove_document("d1") is True
        assert bm25.doc_count == 2
        assert bm25.remove_document("nonexistent") is False

    def test_search_basic(self):
        bm25 = self._make_index()
        results = bm25.search("cat")
        assert len(results) > 0
        # d1 and d3 mention "cat"
        ids = [r["id"] for r in results]
        assert "d1" in ids
        assert "d3" in ids

    def test_search_top_k(self):
        bm25 = self._make_index()
        results = bm25.search("cat sat", top_k=1)
        assert len(results) == 1

    def test_search_no_match(self):
        bm25 = self._make_index()
        results = bm25.search("elephant")
        assert len(results) == 0

    def test_search_empty_query(self):
        bm25 = self._make_index()
        results = bm25.search("")
        assert results == []

    def test_metadata_filter(self):
        bm25 = self._make_index()
        results = bm25.search("cat", metadata_filter={"type": "a"})
        assert len(results) > 0
        for r in results:
            assert r["metadata"]["type"] == "a"

    def test_score_ordering(self):
        bm25 = BM25()
        bm25.add_document("d1", "python programming language")
        bm25.add_document("d2", "python is a programming language for data science")
        bm25.add_document("d3", "java programming")
        results = bm25.search("python programming")
        # d1 should score higher than d2 (more concentrated)
        assert results[0]["id"] == "d1"

    def test_idf_higher_for_rare_terms(self):
        bm25 = BM25()
        bm25.add_document("d1", "the cat sat on the mat")
        bm25.add_document("d2", "the dog lay on the rug")
        bm25.add_document("d3", "the bird flew over the tree")
        # "cat" appears in 1 doc, "the" in 3 docs
        idf_cat = bm25._compute_idf("cat")
        idf_the = bm25._compute_idf("the")
        assert idf_cat > idf_the

    def test_save_and_load(self, tmp_path):
        bm25 = self._make_index()
        path = tmp_path / "bm25.json"
        bm25.save(path)

        loaded = BM25.load(path)
        assert loaded.doc_count == bm25.doc_count
        assert loaded.k1 == bm25.k1
        assert loaded.b == bm25.b

        # Search should produce same results
        r1 = bm25.search("cat")
        r2 = loaded.search("cat")
        assert len(r1) == len(r2)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]

    def test_persistence_content_intact(self, tmp_path):
        bm25 = BM25()
        bm25.add_document("d1", "hello world", metadata={"key": "value"})
        bm25.save(tmp_path / "idx.json")
        loaded = BM25.load(tmp_path / "idx.json")
        results = loaded.search("hello")
        assert results[0]["metadata"]["key"] == "value"


class TestOpenSearchBM25:
    def _make_client(self) -> OpenSearchBM25:
        mock_client = MagicMock()
        mock_client._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "c1", "_score": 1.5, "_source": {"content": "hello world"}},
                    {"_id": "c2", "_score": 0.8, "_source": {"content": "foo bar"}},
                ]
            }
        }
        return OpenSearchBM25(client=mock_client, index_name="test-index")

    def test_search(self):
        os_bm25 = self._make_client()
        results = os_bm25.search("hello")
        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[0]["score"] == 1.5

    def test_search_with_filter(self):
        os_bm25 = self._make_client()
        results = os_bm25.search("hello", metadata_filter={"document_id": "d1"})
        assert len(results) == 2
        # Verify filter was passed
        call_kwargs = os_bm25._client._client.search.call_args
        body = call_kwargs[1]["body"]
        assert "bool" in body["query"]

    def test_multi_field_search(self):
        os_bm25 = self._make_client()
        results = os_bm25.multi_field_search("hello", fields=["content", "title"])
        assert len(results) == 2
        call_kwargs = os_bm25._client._client.search.call_args
        body = call_kwargs[1]["body"]
        assert "multi_match" in body["query"]

    def test_parse_hits(self):
        response = {
            "hits": {
                "hits": [
                    {"_id": "a", "_score": 2.0, "_source": {"content": "x"}},
                ]
            }
        }
        results = OpenSearchBM25._parse_hits(response)
        assert len(results) == 1
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 2.0
        assert results[0]["content"] == "x"
