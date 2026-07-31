"""Tests for incremental ingestion, content hashing, status tracking, and reindex."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_pipeline.data.incremental import (
    ContentHashTracker,
    IncrementalIngestor,
    IngestionJob,
    IngestionStatus,
    IngestionStatusTracker,
    ReindexManager,
)
from rag_pipeline.storage.postgres import ChunkRecord, DocumentRecord


# ================================================================== #
#  ContentHashTracker
# ================================================================== #


class TestContentHashTracker:
    def test_compute_file_hash_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = ContentHashTracker.compute_file_hash(str(f))
        h2 = ContentHashTracker.compute_file_hash(str(f))
        assert h1 == h2

    def test_compute_file_hash_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert ContentHashTracker.compute_file_hash(str(f1)) != ContentHashTracker.compute_file_hash(str(f2))

    def test_compute_content_hash_same(self):
        h1 = ContentHashTracker.compute_content_hash("test content")
        h2 = ContentHashTracker.compute_content_hash("test content")
        assert h1 == h2

    def test_compute_content_hash_different(self):
        h1 = ContentHashTracker.compute_content_hash("aaa")
        h2 = ContentHashTracker.compute_content_hash("bbb")
        assert h1 != h2

    def test_check_changed_returns_false_when_unchanged(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("unchanged content")
        h = ContentHashTracker.compute_file_hash(str(f))
        assert ContentHashTracker.check_changed(str(f), h) is False

    def test_check_changed_returns_true_when_changed(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("original")
        h = ContentHashTracker.compute_file_hash(str(f))
        f.write_text("modified")
        assert ContentHashTracker.check_changed(str(f), h) is True

    def test_hash_is_sha256_hex(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        h = ContentHashTracker.compute_file_hash(str(f))
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ================================================================== #
#  IngestionStatusTracker
# ================================================================== #


class TestIngestionStatusTracker:
    def test_create_job_pending(self):
        tracker = IngestionStatusTracker()
        job = tracker.create("file.txt")
        assert job.status == IngestionStatus.PENDING
        assert job.document_path == "file.txt"
        assert job.job_id

    def test_start_sets_processing(self):
        tracker = IngestionStatusTracker()
        job = tracker.create("file.txt")
        tracker.start(job.job_id)
        updated = tracker.get(job.job_id)
        assert updated.status == IngestionStatus.PROCESSING
        assert updated.started_at is not None

    def test_complete_sets_indexed(self):
        tracker = IngestionStatusTracker()
        job = tracker.create("file.txt")
        tracker.start(job.job_id)
        tracker.complete(job.job_id, chunks_indexed=5, total_chunks=5)
        updated = tracker.get(job.job_id)
        assert updated.status == IngestionStatus.INDEXED
        assert updated.chunks_indexed == 5
        assert updated.completed_at is not None

    def test_fail_sets_error(self):
        tracker = IngestionStatusTracker()
        job = tracker.create("file.txt")
        tracker.start(job.job_id)
        tracker.fail(job.job_id, "boom")
        updated = tracker.get(job.job_id)
        assert updated.status == IngestionStatus.ERROR
        assert updated.error_message == "boom"
        assert updated.completed_at is not None

    def test_get_returns_none_for_unknown(self):
        tracker = IngestionStatusTracker()
        assert tracker.get("nonexistent") is None

    def test_multiple_jobs_independent(self):
        tracker = IngestionStatusTracker()
        j1 = tracker.create("a.txt")
        j2 = tracker.create("b.txt")
        tracker.start(j1.job_id)
        tracker.complete(j1.job_id, 3, 3)
        assert tracker.get(j1.job_id).status == IngestionStatus.INDEXED
        assert tracker.get(j2.job_id).status == IngestionStatus.PENDING


# ================================================================== #
#  IngestionJob dataclass
# ================================================================== #


class TestIngestionJob:
    def test_defaults(self):
        job = IngestionJob(
            job_id="j1",
            document_path="f.txt",
            status=IngestionStatus.PENDING,
        )
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error_message is None
        assert job.chunks_indexed == 0
        assert job.total_chunks == 0

    def test_timestamps_set_after_complete(self):
        now = datetime.now(timezone.utc)
        job = IngestionJob(
            job_id="j1",
            document_path="f.txt",
            status=IngestionStatus.INDEXED,
            started_at=now,
            completed_at=now,
            chunks_indexed=10,
            total_chunks=10,
        )
        assert job.chunks_indexed == 10


# ================================================================== #
#  IncrementalIngestor (mocked)
# ================================================================== #


def _mock_postgres(existing_hash: str | None = None) -> MagicMock:
    pg = MagicMock()
    if existing_hash:
        pg.find_by_file_hash.return_value = DocumentRecord(
            id=existing_hash, filename="doc.txt", file_hash=existing_hash, chunk_count=3,
        )
    else:
        pg.find_by_file_hash.return_value = None
    pg.get_chunks_by_document.return_value = []
    pg.find_documents_by_metadata.return_value = []
    return pg


def _mock_opensearch() -> MagicMock:
    os = MagicMock()
    os.index_chunks.return_value = 5
    return os


def _mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embed_chunks.side_effect = lambda chunks: chunks
    return e


def _mock_s3() -> MagicMock:
    s3 = MagicMock()
    s3.upload_file.return_value = "s3://bucket/key"
    return s3


class TestIncrementalIngestor:
    def test_skip_unchanged_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")

        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())

        # Patch compute_file_hash to return a known hash, and make find_by_file_hash return a match
        with patch.object(ContentHashTracker, "compute_file_hash", return_value="abc123"):
            pg.find_by_file_hash.return_value = DocumentRecord(id="abc123", filename="doc.txt")
            job = ingestor.ingest_incremental(str(f), "test-index")

        assert job.status == IngestionStatus.PENDING
        assert job.error_message == "unchanged"

    def test_ingest_new_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Some content for testing. " * 20)

        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())

        with patch.object(ContentHashTracker, "compute_file_hash", return_value="newhash"):
            pg.find_by_file_hash.return_value = None
            job = ingestor.ingest_incremental(str(f), "test-index")

        assert job.status == IngestionStatus.INDEXED
        assert job.chunks_indexed > 0

    def test_batch_incremental(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A " * 20)
        f2.write_text("content B " * 20)

        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())

        with patch.object(ContentHashTracker, "compute_file_hash", return_value="hash"):
            pg.find_by_file_hash.return_value = None
            jobs = ingestor.batch_incremental([str(f1), str(f2)], "test-index")

        assert len(jobs) == 2
        assert all(j.status == IngestionStatus.INDEXED for j in jobs)

    def test_get_status_returns_job(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content " * 20)

        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())

        with patch.object(ContentHashTracker, "compute_file_hash", return_value="h"):
            pg.find_by_file_hash.return_value = None
            job = ingestor.ingest_incremental(str(f), "idx")

        retrieved = ingestor.get_status(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_status_unknown_returns_none(self):
        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())
        assert ingestor.get_status("nope") is None

    def test_error_on_nonexistent_file(self):
        pg = _mock_postgres()
        ingestor = IncrementalIngestor(pg, _mock_opensearch(), _mock_embedder(), _mock_s3())
        job = ingestor.ingest_incremental("/no/such/file.txt", "idx")
        assert job.status == IngestionStatus.ERROR
        assert "Cannot read file" in job.error_message

    def test_delete_document(self):
        pg = _mock_postgres()
        pg.delete_document.return_value = True
        os = _mock_opensearch()
        s3 = _mock_s3()
        ingestor = IncrementalIngestor(pg, os, _mock_embedder(), s3)

        result = ingestor.delete_document("doc123")
        assert result is True
        pg.delete_document.assert_called_once_with("doc123")


# ================================================================== #
#  ReindexManager (mocked)
# ================================================================== #


class TestReindexManager:
    def test_reindex_document_not_found(self):
        pg = _mock_postgres()
        pg.get_document.return_value = None
        os = _mock_opensearch()
        embedder = _mock_embedder()
        manager = ReindexManager(pg, os, embedder)

        job = manager.reindex_document("missing", "test-index")
        assert job.status == IngestionStatus.ERROR
        assert "not found" in job.error_message

    def test_reindex_document_no_chunks(self):
        pg = _mock_postgres()
        pg.get_document.return_value = DocumentRecord(id="d1", filename="f.txt")
        pg.get_chunks_by_document.return_value = []
        os = _mock_opensearch()
        manager = ReindexManager(pg, os, _mock_embedder())

        job = manager.reindex_document("d1", "test-index")
        assert job.status == IngestionStatus.INDEXED
        assert job.chunks_indexed == 0

    def test_reindex_document_success(self):
        pg = _mock_postgres()
        pg.get_document.return_value = DocumentRecord(id="d1", filename="f.txt")
        pg.get_chunks_by_document.return_value = [
            ChunkRecord(id="c1", document_id="d1", chunk_index=0, content="hello world", token_count=2),
            ChunkRecord(id="c2", document_id="d1", chunk_index=1, content="more content", token_count=2),
        ]
        os = _mock_opensearch()
        os.index_exists.return_value = True
        manager = ReindexManager(pg, os, _mock_embedder())

        job = manager.reindex_document("d1", "test-index")
        assert job.status == IngestionStatus.INDEXED
        assert job.chunks_indexed == 2

        # Verify alias_swap was called
        assert os.alias_swap.call_count >= 1

    def test_reindex_all(self):
        pg = _mock_postgres()
        pg.list_documents.return_value = [
            DocumentRecord(id="d1", filename="a.txt"),
            DocumentRecord(id="d2", filename="b.txt"),
        ]
        pg.get_document.side_effect = lambda did: DocumentRecord(id=did, filename=f"{did}.txt")
        pg.get_chunks_by_document.return_value = []
        os = _mock_opensearch()
        manager = ReindexManager(pg, os, _mock_embedder())

        jobs = manager.reindex_all("test-index")
        assert len(jobs) == 2

    def test_reindex_get_status(self):
        pg = _mock_postgres()
        pg.get_document.return_value = None
        os = _mock_opensearch()
        manager = ReindexManager(pg, os, _mock_embedder())

        job = manager.reindex_document("d1", "idx")
        assert manager.get_status(job.job_id) is not None
        assert manager.get_status("nonexistent") is None
