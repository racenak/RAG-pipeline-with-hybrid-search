"""Shared test fixtures for RAG pipeline tests."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_txt_path():
    """Path to sample.txt fixture."""
    return FIXTURES_DIR / "sample.txt"


@pytest.fixture
def sample_md_path():
    """Path to sample.md fixture."""
    return FIXTURES_DIR / "sample.md"


@pytest.fixture
def sample_html_path():
    """Path to sample.html fixture."""
    return FIXTURES_DIR / "sample.html"


@pytest.fixture
def sample_csv_path():
    """Path to sample.csv fixture."""
    return FIXTURES_DIR / "sample.csv"


@pytest.fixture
def sample_pdf_path():
    """Path to sample.pdf fixture."""
    return FIXTURES_DIR / "sample.pdf"


@pytest.fixture
def sample_docx_path():
    """Path to sample.docx fixture."""
    return FIXTURES_DIR / "sample.docx"


@pytest.fixture
def mock_opensearch():
    """Mock OpenSearch client."""
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "doc1",
                    "_score": 0.9,
                    "_source": {"content": "Test content 1", "document_id": "doc1"},
                },
                {
                    "_id": "doc2",
                    "_score": 0.8,
                    "_source": {"content": "Test content 2", "document_id": "doc2"},
                },
            ],
            "total": {"value": 2},
        }
    }
    client.indices.exists.return_value = True
    client.info.return_value = {"version": {"number": "3.7.0"}}
    return client


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL connection."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.setex.return_value = True
    client.ping.return_value = True
    client.keys.return_value = []
    return client


@pytest.fixture
def mock_llm_backend():
    """Mock LLM backend that returns deterministic responses."""
    backend = AsyncMock()
    backend.generate.return_value = "This is a mock LLM response about the query."
    return backend


@pytest.fixture
def sample_chunks():
    """Sample chunk data for testing."""
    from rag_pipeline.data.chunking import Chunk

    return [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="The embedding dimension is 1024 for BGE-large-en-v1.5 model.",
            index=0,
            token_count=12,
            metadata={"source": "test"},
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            content="OpenSearch supports kNN vector search with HNSW algorithm.",
            index=1,
            token_count=10,
            metadata={"source": "test"},
        ),
        Chunk(
            id="chunk-3",
            document_id="doc-2",
            content="Redis provides TTL-based caching for query results.",
            index=0,
            token_count=9,
            metadata={"source": "test"},
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from rag_pipeline.retrieval.vector import SearchResult

    return [
        SearchResult(id="chunk-1", score=0.95, content="Embedding dimension is 1024", metadata={}),
        SearchResult(id="chunk-2", score=0.85, content="OpenSearch kNN search", metadata={}),
        SearchResult(id="chunk-3", score=0.75, content="Redis caching layer", metadata={}),
    ]


@pytest.fixture
def sample_eval_cases():
    """Sample evaluation cases for testing."""
    return [
        {
            "id": "eval-1",
            "query": "What is the embedding dimension?",
            "expected_answer": "1024",
            "expected_documents": ["doc-1"],
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "id": "eval-2",
            "query": "Compare vector search vs BM25",
            "expected_answer": "Vector search uses semantic similarity while BM25 uses keyword matching.",
            "expected_documents": ["doc-1", "doc-2"],
            "category": "comparison",
            "difficulty": "medium",
        },
    ]
