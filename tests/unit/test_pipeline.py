"""Tests for the full ingestion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

from rag_pipeline.pipeline import PipelineResult, ingest_file_full

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.upload_file.return_value = "s3-key-123"
    return storage


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_chunks.side_effect = lambda chunks: chunks
    return embedder


def _mock_opensearch() -> MagicMock:
    client = MagicMock()
    client.index_chunks.return_value = 3
    return client


def _mock_postgres() -> MagicMock:
    return MagicMock()


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(document_id="d1", source="test", source_type="file", success=True)
        assert r.chunks_count == 0
        assert r.indexed_in_opensearch is False
        assert r.registered_in_postgres is False
        assert r.error is None


class TestIngestFileFull:
    def test_txt_file(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document with enough content to chunk. " * 20)

        result = ingest_file_full(
            path=test_file,
            storage=_mock_storage(),
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=_mock_postgres(),
            index_name="test-index",
        )

        assert result.success is True
        assert result.chunks_count > 0
        assert result.indexed_in_opensearch is True
        assert result.registered_in_postgres is True
        assert result.s3_key == "s3-key-123"
        assert "validate" in result.timings
        assert "parse" in result.timings
        assert "chunk" in result.timings
        assert "embed" in result.timings

    def test_invalid_file(self, tmp_path):
        test_file = tmp_path / "test.xyz"
        test_file.write_text("unsupported format")

        result = ingest_file_full(path=test_file)
        assert result.success is False
        assert "Unsupported" in result.error or "error" in result.error.lower()

    def test_empty_file(self, tmp_path):
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = ingest_file_full(path=test_file)
        # Empty files are rejected by validation
        assert result.success is False
        assert result.chunks_count == 0

    def test_no_storage(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content to ingest. " * 20)

        result = ingest_file_full(
            path=test_file,
            storage=None,
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=_mock_postgres(),
        )
        assert result.success is True
        assert result.s3_key is None

    def test_opensearch_failure(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content to ingest. " * 20)

        failing_os = MagicMock()
        failing_os.index_chunks.side_effect = ConnectionError("OpenSearch down")

        result = ingest_file_full(
            path=test_file,
            embedder=_mock_embedder(),
            opensearch_client=failing_os,
            postgres_client=_mock_postgres(),
        )
        assert result.success is True
        assert result.indexed_in_opensearch is False
        assert result.chunks_count > 0

    def test_postgres_failure(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content to ingest. " * 20)

        failing_pg = MagicMock()
        failing_pg.insert_document.side_effect = ConnectionError("PG down")

        result = ingest_file_full(
            path=test_file,
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=failing_pg,
        )
        assert result.success is True
        assert result.registered_in_postgres is False
        assert result.indexed_in_opensearch is True

    def test_s3_failure_graceful(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content to ingest. " * 20)

        failing_s3 = MagicMock()
        failing_s3.upload_file.side_effect = ConnectionError("S3 down")

        result = ingest_file_full(
            path=test_file,
            storage=failing_s3,
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=_mock_postgres(),
        )
        assert result.success is True
        assert result.s3_key is None

    def test_md_file(self, tmp_path):
        test_file = tmp_path / "readme.md"
        test_file.write_text("# Title\n\nSome content here. " * 20)

        result = ingest_file_full(
            path=test_file,
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=_mock_postgres(),
        )
        assert result.success is True
        assert result.chunks_count > 0

    def test_timings_recorded(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content for timing test. " * 20)

        result = ingest_file_full(
            path=test_file,
            embedder=_mock_embedder(),
            opensearch_client=_mock_opensearch(),
            postgres_client=_mock_postgres(),
        )
        assert isinstance(result.timings, dict)
        assert len(result.timings) > 0
        for v in result.timings.values():
            assert isinstance(v, float)
            assert v >= 0
