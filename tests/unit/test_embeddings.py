"""Tests for embedding generation and caching."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_pipeline.embeddings.cache import EmbeddingCache
from rag_pipeline.embeddings.generator import EmbeddingGenerator

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestEmbeddingCache:
    def _make_cache(self, tmp_path: Path) -> EmbeddingCache:
        return EmbeddingCache(cache_dir=tmp_path / "emb_cache")

    def test_set_and_get(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        cache.set("hello world", [0.1, 0.2, 0.3])
        result = cache.get("hello world")
        assert result is not None
        assert len(result) == 3
        assert abs(result[0] - 0.1) < 1e-6

    def test_get_miss(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        assert cache.get("nonexistent") is None

    def test_has(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        assert not cache.has("hello")
        cache.set("hello", [0.1])
        assert cache.has("hello")

    def test_deterministic_hash(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        cache.set("hello", [0.1])
        # Same text → same cache file
        assert cache.has("hello")

    def test_different_texts_different_keys(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        cache.set("hello", [0.1])
        cache.set("world", [0.2])
        assert cache.get("hello") != cache.get("world")

    def test_set_many_get_many(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        texts = ["hello", "world"]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        cache.set_many(texts, vectors)

        hits = cache.get_many(["hello", "world", "missing"])
        assert 0 in hits
        assert 1 in hits
        assert 2 not in hits
        assert abs(hits[0][0] - 0.1) < 1e-6

    def test_clear(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        cache.set("a", [0.1])
        cache.set("b", [0.2])
        assert cache.size == 2
        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_size(self, tmp_path: Path):
        cache = self._make_cache(tmp_path)
        assert cache.size == 0
        cache.set("a", [0.1])
        assert cache.size == 1
        cache.set("b", [0.2])
        assert cache.size == 2


class TestEmbeddingGenerator:
    def _make_generator(self) -> EmbeddingGenerator:
        with patch(
            "rag_pipeline.embeddings.generator.EmbeddingGenerator.__init__", return_value=None
        ):
            gen = EmbeddingGenerator.__new__(EmbeddingGenerator)
            gen._batch_size = 64
            gen._cache_enabled = False
            gen._cache = None
            gen._backend = MagicMock()
            gen._backend.dimension = 1024
            gen._backend.encode.return_value = [[0.1] * 1024, [0.2] * 1024]
            return gen

    def test_embed_batch(self):
        gen = self._make_generator()
        vectors = gen.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 1024

    def test_embed_single(self):
        gen = self._make_generator()
        gen._backend.encode.return_value = [[0.1] * 1024]
        vector = gen.embed_single("hello")
        assert len(vector) == 1024

    def test_embed_empty(self):
        gen = self._make_generator()
        assert gen.embed([]) == []

    def test_embed_with_cache(self, tmp_path: Path):
        gen = self._make_generator()
        gen._cache_enabled = True
        gen._cache = EmbeddingCache(tmp_path / "cache")
        gen._backend.encode.return_value = [[0.1] * 1024]

        # First call — cache miss
        vectors = gen.embed(["hello"])
        assert len(vectors) == 1
        assert gen._backend.encode.called

        # Reset mock
        gen._backend.encode.reset_mock()

        # Second call — cache hit
        vectors2 = gen.embed(["hello"])
        assert len(vectors2) == 1
        gen._backend.encode.assert_not_called()

    def test_dimension(self):
        gen = self._make_generator()
        assert gen.dimension == 1024

    def test_clear_cache(self, tmp_path: Path):
        gen = self._make_generator()
        gen._cache_enabled = True
        gen._cache = EmbeddingCache(tmp_path / "cache")
        gen._cache.set("a", [0.1])
        assert gen.cache_size == 1
        gen.clear_cache()
        assert gen.cache_size == 0

    def test_cache_disabled(self):
        gen = self._make_generator()
        gen._cache_enabled = False
        gen._cache = None
        gen._backend.encode.return_value = [[0.1] * 1024]
        vectors = gen.embed(["hello"])
        assert len(vectors) == 1
        assert gen.cache_size == 0
