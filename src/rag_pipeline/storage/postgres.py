"""PostgreSQL document and chunk registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Schema
# ------------------------------------------------------------------ #

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_url      TEXT,
    mime_type       TEXT,
    file_hash       TEXT,
    size_bytes      INTEGER,
    chunk_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    indexed         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
"""


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #


@dataclass
class DocumentRecord:
    """A registered document in PostgreSQL."""

    id: str
    filename: str
    source_url: str | None = None
    mime_type: str | None = None
    file_hash: str | None = None
    size_bytes: int | None = None
    chunk_count: int = 0
    created_at: datetime | None = None


@dataclass
class ChunkRecord:
    """A registered chunk in PostgreSQL."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int | None = None
    indexed: bool = False
    created_at: datetime | None = None


# ------------------------------------------------------------------ #
#  Client
# ------------------------------------------------------------------ #


class PostgresClient:
    """PostgreSQL client for document/chunk registry."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "rag_pipeline",
        user: str = "rag",
        password: str = "rag_dev_password",
        pool_size: int = 5,
    ) -> None:
        self._dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self._pool_size = pool_size
        self._conn: psycopg2.extensions.connection | None = None

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
        return self._conn

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        logger.info("PostgreSQL schema initialized")

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning("PostgreSQL health check failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    #  Documents
    # ------------------------------------------------------------------ #

    def insert_document(self, doc: DocumentRecord) -> None:
        """Insert a document record."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, filename, source_url, mime_type, file_hash, size_bytes, chunk_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    source_url = EXCLUDED.source_url,
                    mime_type = EXCLUDED.mime_type,
                    file_hash = EXCLUDED.file_hash,
                    size_bytes = EXCLUDED.size_bytes,
                    chunk_count = EXCLUDED.chunk_count
                """,
                (
                    doc.id,
                    doc.filename,
                    doc.source_url,
                    doc.mime_type,
                    doc.file_hash,
                    doc.size_bytes,
                    doc.chunk_count,
                ),
            )
        logger.info("Inserted document: %s (%s)", doc.id, doc.filename)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Get a document by ID."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return DocumentRecord(**row)

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        """List documents with pagination."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            return [DocumentRecord(**row) for row in cur.fetchall()]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks (cascade). Returns True if deleted."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            return cur.rowcount > 0

    def count_documents(self) -> int:
        """Total document count."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def find_by_file_hash(self, file_hash: str) -> DocumentRecord | None:
        """Find a document by its content hash. Returns None if not found."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
            if row is None:
                return None
            return DocumentRecord(**row)

    def find_documents_by_metadata(self, **filters) -> list[DocumentRecord]:
        """Find documents matching metadata filters (e.g. source_url containing a pattern)."""
        if not filters:
            return self.list_documents()
        conn = self._get_conn()
        conditions = []
        values = []
        for key, value in filters.items():
            if key in ("source_url", "filename", "mime_type"):
                conditions.append(f"{key} LIKE %s")
                values.append(f"%{value}%")
            elif key == "crawl_group":
                conditions.append("source_url LIKE %s")
                values.append(f"%{value}%")
        if not conditions:
            return self.list_documents()
        where = " AND ".join(conditions)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM documents WHERE {where}", values)
            return [DocumentRecord(**row) for row in cur.fetchall()]

    def get_document_ids_by_metadata(self, **filters) -> list[str]:
        """Get document IDs matching metadata filters."""
        docs = self.find_documents_by_metadata(**filters)
        return [d.id for d in docs]

    def update_chunk_count(self, doc_id: str, count: int) -> None:
        """Update the chunk count for a document."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET chunk_count = %s WHERE id = %s",
                (count, doc_id),
            )

    # ------------------------------------------------------------------ #
    #  Chunks
    # ------------------------------------------------------------------ #

    def insert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Bulk insert chunk records."""
        if not chunks:
            return
        conn = self._get_conn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO chunks (id, document_id, chunk_index, content, token_count, indexed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    indexed = EXCLUDED.indexed
                """,
                [
                    (c.id, c.document_id, c.chunk_index, c.content, c.token_count, c.indexed)
                    for c in chunks
                ],
                page_size=500,
            )
        logger.info("Inserted %d chunks", len(chunks))

    def get_chunks_by_document(self, doc_id: str) -> list[ChunkRecord]:
        """Get all chunks for a document, ordered by index."""
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                (doc_id,),
            )
            return [ChunkRecord(**row) for row in cur.fetchall()]

    def mark_chunks_indexed(self, chunk_ids: list[str]) -> None:
        """Mark chunks as indexed in OpenSearch."""
        if not chunk_ids:
            return
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET indexed = TRUE WHERE id = ANY(%s)",
                (chunk_ids,),
            )

    def count_chunks(self, doc_id: str | None = None) -> int:
        """Count chunks, optionally for a specific document."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            if doc_id:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (doc_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]

    def delete_chunks_by_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            return cur.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("PostgreSQL connection closed")
