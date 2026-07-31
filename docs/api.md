# API Documentation — RAG Pipeline with Hybrid Search

Version: `v1`
Base URL: `http://localhost:8000/api/v1`

---

## Table of Contents

1. [API Overview](#1-api-overview)
2. [Document Management Endpoints](#2-document-management-endpoints)
3. [Query / Retrieval Endpoints](#3-query--retrieval-endpoints)
4. [Search Endpoints](#4-search-endpoints)
5. [Pipeline Management Endpoints](#5-pipeline-management-endpoints)
6. [Data Models](#6-data-models)
7. [Configuration](#7-configuration)

---

## 1. API Overview

### 1.1 Framework

The API is built with **FastAPI** (Python 3.11+). All request/response payloads use JSON (`application/json`). File uploads use `multipart/form-data`.

OpenAPI schema is auto-generated and available at `/docs` (Swagger UI) and `/openapi.json`.

### 1.2 RESTful Design

| Principle | Convention |
|---|---|
| Resource naming | Plural nouns: `/documents`, `/pipeline` |
| Actions via HTTP methods | `GET` read, `POST` create/action, `PUT` replace, `PATCH` update, `DELETE` remove |
| Versioning | URL prefix: `/api/v1/` |
| Idempotency | `DELETE` and `POST /reindex` are idempotent |
| Pagination | Cursor-based with `limit` and `cursor` query parameters |
| Filtering | Query parameters for simple filters; request body for complex filters |
| Error format | Consistent JSON error envelope across all endpoints |

### 1.3 Authentication and Authorization

The API supports optional authentication configured via environment variables:

- **Disabled by default** for local development (`AUTH_ENABLED=false`).
- When enabled, uses **JWT Bearer tokens** (`Authorization: Bearer <token>`).
- Tokens are validated against a configurable JWKS endpoint.
- A lightweight role model controls access:

| Role | Capabilities |
|---|---|
| `reader` | Query, search, read document metadata |
| `writer` | `reader` + upload, delete, reindex documents |
| `admin` | `writer` + pipeline management, evaluation, metrics |

Endpoints that require authentication return `401 Unauthorized` when no valid token is present, and `403 Forbidden` when the token lacks the required role.

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

### 1.4 Rate Limiting

Rate limiting is enforced per API key or per authenticated user:

| Tier | Requests / minute | Burst |
|---|---|---|
| Anonymous | 30 | 10 |
| Reader | 120 | 30 |
| Writer | 60 | 20 |
| Admin | 200 | 50 |

Rate limit headers are included in every response:

```http
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 114
X-RateLimit-Reset: 1690000000
```

When exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header.

---

## 2. Document Management Endpoints

### 2.1 POST `/api/v1/documents/ingest`

Upload one or more documents for ingestion into the pipeline. Accepts file uploads or references to documents already in object storage.

**Request**

- Content-Type: `multipart/form-data` (file upload) or `application/json` (reference)

**Form fields (multipart):**

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | file[] | yes | One or more files to upload |
| `collection` | string | no | Target collection name (default: `default`) |
| `metadata` | string (JSON) | no | JSON string of key-value metadata applied to all files |
| `chunking_strategy` | string | no | `recursive`, `semantic`, `fixed` (default: `recursive`) |

**JSON body (reference):**

```json
{
  "sources": [
    {
      "storage_key": "s3://bucket/path/to/doc.pdf",
      "mime_type": "application/pdf"
    }
  ],
  "collection": "engineering-docs",
  "metadata": {
    "team": "platform",
    "project": "rag-pipeline"
  },
  "chunking_strategy": "recursive"
}
```

**Response — `202 Accepted`**

```json
{
  "job_id": "ingest_a1b2c3d4e5",
  "status": "queued",
  "documents": [
    {
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "collection": "engineering-docs",
      "status": "queued"
    }
  ],
  "created_at": "2026-07-27T10:00:00Z"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Invalid file type, missing required fields, malformed metadata JSON |
| `401` | Missing or invalid authentication |
| `403` | User lacks `writer` role |
| `413` | File exceeds maximum size (configurable, default 50 MB) |
| `422` | Validation error in request body |
| `429` | Rate limit exceeded |

```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "File type 'application/exe' is not supported",
    "details": {
      "allowed_types": ["application/pdf", "text/plain", "text/markdown"]
    }
  }
}
```

---

### 2.2 GET `/api/v1/documents/`

List ingested documents with optional filtering and pagination.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collection` | string | — | Filter by collection name |
| `status` | string | — | Filter by status: `queued`, `processing`, `indexed`, `failed` |
| `limit` | integer | 20 | Max results per page (1–100) |
| `cursor` | string | — | Pagination cursor from previous response |
| `sort_by` | string | `created_at` | Sort field: `created_at`, `updated_at`, `filename` |
| `sort_order` | string | `desc` | `asc` or `desc` |

**Response — `200 OK`**

```json
{
  "documents": [
    {
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "collection": "engineering-docs",
      "status": "indexed",
      "chunk_count": 47,
      "file_size_bytes": 1048576,
      "mime_type": "application/pdf",
      "metadata": {
        "team": "platform"
      },
      "created_at": "2026-07-27T10:00:00Z",
      "updated_at": "2026-07-27T10:02:30Z"
    }
  ],
  "pagination": {
    "next_cursor": "eyJpZCI6ImRvY18xMjM0In0=",
    "has_more": true,
    "total_count": 156
  }
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `401` | Authentication required but not provided |
| `403` | User lacks `reader` role |
| `422` | Invalid query parameters |

---

### 2.3 GET `/api/v1/documents/{doc_id}`

Retrieve full details for a single document, including its chunks and processing state.

**Path parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | string | Document identifier |

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_chunks` | boolean | false | Include chunk summaries in response |

**Response — `200 OK`**

```json
{
  "doc_id": "doc_x1y2z3",
  "filename": "architecture.pdf",
  "collection": "engineering-docs",
  "status": "indexed",
  "mime_type": "application/pdf",
  "file_size_bytes": 1048576,
  "storage_key": "s3://bucket/raw/architecture.pdf",
  "chunk_count": 47,
  "metadata": {
    "team": "platform",
    "project": "rag-pipeline"
  },
  "processing": {
    "chunking_strategy": "recursive",
    "embedding_model": "text-embedding-3-small",
    "indexed_at": "2026-07-27T10:02:30Z",
    "duration_ms": 4320
  },
  "chunks": [
    {
      "chunk_id": "chk_abc123",
      "index": 0,
      "content_preview": "The RAG pipeline architecture consists of three main layers...",
      "token_count": 384,
      "start_page": 1,
      "end_page": 2
    }
  ],
  "created_at": "2026-07-27T10:00:00Z",
  "updated_at": "2026-07-27T10:02:30Z"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `401` | Authentication required but not provided |
| `403` | User lacks `reader` role |
| `404` | Document not found |

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "No document found with id 'doc_nonexistent'"
  }
}
```

---

### 2.4 DELETE `/api/v1/documents/{doc_id}`

Remove a document and all its chunks from the pipeline. This deletes data from OpenSearch, PostgreSQL, and object storage.

**Path parameters:**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | string | Document identifier |

**Response — `200 OK`**

```json
{
  "doc_id": "doc_x1y2z3",
  "status": "deleted",
  "deleted_at": "2026-07-27T12:00:00Z"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `401` | Authentication required but not provided |
| `403` | User lacks `writer` role |
| `404` | Document not found |

---

### 2.5 POST `/api/v1/documents/reindex`

Trigger re-indexing for one or more documents. Useful after chunking strategy changes or embedding model upgrades.

**Request body:**

```json
{
  "doc_ids": ["doc_x1y2z3", "doc_a1b2c3"],
  "chunking_strategy": "semantic",
  "embedding_model": "text-embedding-3-small"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `doc_ids` | string[] | yes | Document IDs to reindex (max 50) |
| `chunking_strategy` | string | no | Override chunking strategy |
| `embedding_model` | string | no | Override embedding model |

**Response — `202 Accepted`**

```json
{
  "job_id": "reindex_f6g7h8",
  "status": "queued",
  "document_count": 2,
  "created_at": "2026-07-27T12:00:00Z"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Empty `doc_ids` array or exceeds 50 |
| `401` | Authentication required but not provided |
| `403` | User lacks `writer` role |
| `404` | One or more document IDs not found |
| `409` | Reindex already in progress for these documents |

```json
{
  "error": {
    "code": "REINDEX_IN_PROGRESS",
    "message": "Reindex job 'reindex_f6g7h8' is already running for the requested documents",
    "details": {
      "existing_job_id": "reindex_f6g7h8"
    }
  }
}
```

---

## 3. Query / Retrieval Endpoints

### 3.1 POST `/api/v1/query`

The primary RAG endpoint. Takes a user query, retrieves relevant chunks, optionally reranks them, and generates an answer with citations.

**Request body:**

```json
{
  "query": "How does the hybrid search pipeline combine BM25 and vector results?",
  "collection": "engineering-docs",
  "retrieval_config": {
    "top_k": 10,
    "rerank": true,
    "rerank_top_k": 20,
    "mode": "hybrid",
    "filters": {
      "metadata.team": "platform"
    },
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "rrf_k": 60
  },
  "generation_config": {
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 2048,
    "system_prompt": "You are a helpful assistant that answers questions about the RAG pipeline architecture.",
    "stream": false
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | User query text (1–10,000 characters) |
| `collection` | string | no | Target collection (default: `default`) |
| `conversation_id` | string | no | ID for multi-turn conversations |
| `retrieval_config` | RetrievalConfig | no | Overrides for retrieval behavior |
| `generation_config` | GenerationConfig | no | Overrides for LLM generation |

**Response — `200 OK` (non-streaming)**

```json
{
  "answer": "The hybrid search pipeline combines BM25 and vector retrieval using Reciprocal Rank Fusion (RRF). Each retrieval method produces an ordered list of results, and RRF merges them by computing a fused score based on each result's rank position across both lists. The `rrf_k` parameter (default 60) controls the steepness of the ranking function...",
  "sources": [
    {
      "chunk_id": "chk_abc123",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "The pipeline uses Reciprocal Rank Fusion to merge BM25 and vector results...",
      "score": 0.89,
      "rerank_score": 0.94,
      "metadata": {
        "team": "platform",
        "section": "retrieval"
      },
      "citation_index": 1
    },
    {
      "chunk_id": "chk_def456",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "RRF computes a fused score: score(d) = sum(1 / (k + rank_i(d)))...",
      "score": 0.82,
      "rerank_score": 0.91,
      "metadata": {
        "team": "platform",
        "section": "retrieval"
      },
      "citation_index": 2
    }
  ],
  "conversation_id": "conv_9z8y7x",
  "metadata": {
    "model": "gpt-4o",
    "retrieval_mode": "hybrid",
    "chunks_retrieved": 10,
    "chunks_reranked": 10,
    "latency_ms": 1847,
    "latency_breakdown": {
      "embedding": 120,
      "bm25": 45,
      "vector": 68,
      "fusion": 12,
      "reranking": 310,
      "generation": 1290
    }
  }
}
```

**Streaming response (SSE) when `generation_config.stream = true`:**

```
event: metadata
data: {"conversation_id":"conv_9z8y7x","model":"gpt-4o","retrieval_latency_ms":555}

event: sources
data: [{"chunk_id":"chk_abc123","doc_id":"doc_x1y2z3","citation_index":1,"score":0.89}]

event: token
data: {"text":"The"}

event: token
data: {"text":" hybrid"}

event: token
data: {"text":" search"}

...

event: done
data: {"total_tokens":512,"latency_ms":1847}
```

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Query empty or exceeds length limit |
| `401` | Authentication required but not provided |
| `403` | User lacks `reader` role |
| `408` | Generation timed out (configurable, default 30s) |
| `422` | Invalid retrieval or generation config |
| `429` | Rate limit exceeded |
| `500` | LLM provider failure or internal error |

```json
{
  "error": {
    "code": "GENERATION_TIMEOUT",
    "message": "LLM generation did not complete within 30000ms",
    "details": {
      "timeout_ms": 30000,
      "retrieval_completed": true
    }
  }
}
```

---

### 3.2 POST `/api/v1/retrieve`

Retrieval only — returns ranked chunks without LLM generation. Useful for debugging retrieval quality or building custom generation pipelines.

**Request body:**

```json
{
  "query": "How does the hybrid search pipeline combine BM25 and vector results?",
  "collection": "engineering-docs",
  "top_k": 10,
  "mode": "hybrid",
  "rerank": true,
  "rerank_top_k": 20,
  "filters": {
    "metadata.team": "platform"
  },
  "vector_weight": 0.6,
  "bm25_weight": 0.4,
  "rrf_k": 60
}
```

**Response — `200 OK`**

```json
{
  "results": [
    {
      "chunk_id": "chk_abc123",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "The pipeline uses Reciprocal Rank Fusion to merge BM25 and vector results...",
      "score": 0.89,
      "rerank_score": 0.94,
      "vector_score": 0.91,
      "bm25_score": 0.78,
      "rank": 1,
      "metadata": {
        "team": "platform",
        "section": "retrieval"
      }
    }
  ],
  "metadata": {
    "mode": "hybrid",
    "total_results": 10,
    "latency_ms": 555,
    "latency_breakdown": {
      "embedding": 120,
      "bm25": 45,
      "vector": 68,
      "fusion": 12,
      "reranking": 310
    }
  }
}
```

---

## 4. Search Endpoints

Low-level search endpoints for direct access to individual retrieval strategies.

### 4.1 POST `/api/v1/search/vector`

Pure semantic vector search using embeddings.

**Request body:**

```json
{
  "query": "architecture design patterns",
  "collection": "engineering-docs",
  "top_k": 10,
  "filters": {
    "metadata.team": "platform",
    "created_at": {
      "$gte": "2026-01-01"
    }
  },
  "embedding_model": "text-embedding-3-small"
}
```

**Response — `200 OK`**

```json
{
  "results": [
    {
      "chunk_id": "chk_abc123",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "The RAG pipeline architecture follows a layered design pattern...",
      "score": 0.91,
      "metadata": {}
    }
  ],
  "metadata": {
    "mode": "vector",
    "embedding_model": "text-embedding-3-small",
    "latency_ms": 68
  }
}
```

---

### 4.2 POST `/api/v1/search/bm25`

Pure lexical BM25 search.

**Request body:**

```json
{
  "query": "reciprocal rank fusion algorithm",
  "collection": "engineering-docs",
  "top_k": 10,
  "filters": {
    "metadata.team": "platform"
  },
  "analyzer": "standard"
}
```

**Response — `200 OK`**

```json
{
  "results": [
    {
      "chunk_id": "chk_def456",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "RRF computes a fused score: score(d) = sum(1 / (k + rank_i(d)))...",
      "score": 0.78,
      "metadata": {}
    }
  ],
  "metadata": {
    "mode": "bm25",
    "analyzer": "standard",
    "latency_ms": 45
  }
}
```

---

### 4.3 POST `/api/v1/search/hybrid`

Hybrid search combining BM25 and vector retrieval with Reciprocal Rank Fusion (RRF).

**Request body:**

```json
{
  "query": "how does reranking work in the pipeline",
  "collection": "engineering-docs",
  "top_k": 10,
  "filters": {
    "metadata.team": "platform"
  },
  "vector_weight": 0.6,
  "bm25_weight": 0.4,
  "rrf_k": 60,
  "rerank": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Search query (required) |
| `collection` | string | `default` | Target collection |
| `top_k` | integer | 10 | Number of results (1–100) |
| `filters` | object | — | Metadata filter conditions |
| `vector_weight` | float | 0.5 | Weight for vector scores (0.0–1.0) |
| `bm25_weight` | float | 0.5 | Weight for BM25 scores (0.0–1.0) |
| `rrf_k` | integer | 60 | RRF parameter controlling rank falloff |
| `rerank` | boolean | false | Apply cross-encoder reranking |

**Response — `200 OK`**

```json
{
  "results": [
    {
      "chunk_id": "chk_abc123",
      "doc_id": "doc_x1y2z3",
      "filename": "architecture.pdf",
      "content": "The reranking stage applies a cross-encoder model to rescore top candidates...",
      "score": 0.89,
      "vector_score": 0.91,
      "bm25_score": 0.78,
      "rank": 1,
      "metadata": {}
    }
  ],
  "metadata": {
    "mode": "hybrid",
    "rrf_k": 60,
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "latency_ms": 125
  }
}
```

---

## 5. Pipeline Management Endpoints

### 5.1 GET `/api/v1/pipeline/status`

Returns health and status of all pipeline components.

**Response — `200 OK`**

```json
{
  "status": "healthy",
  "components": {
    "opensearch": {
      "status": "healthy",
      "latency_ms": 3,
      "index_count": 4
    },
    "postgresql": {
      "status": "healthy",
      "latency_ms": 2,
      "active_connections": 8
    },
    "minio": {
      "status": "healthy",
      "latency_ms": 5,
      "bucket": "rag-documents"
    },
    "embedding_service": {
      "status": "healthy",
      "model": "text-embedding-3-small",
      "dimension": 1536
    },
    "reranker": {
      "status": "healthy",
      "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "llm": {
      "status": "healthy",
      "provider": "openai",
      "model": "gpt-4o"
    }
  },
  "checked_at": "2026-07-27T12:00:00Z"
}
```

---

### 5.2 POST `/api/v1/pipeline/evaluate`

Trigger an evaluation run against a predefined test dataset.

**Request body:**

```json
{
  "dataset": "rag-eval-v1",
  "sample_size": 50,
  "metrics": ["precision@5", "recall@10", "mrr", "answer_faithfulness"],
  "retrieval_config": {
    "mode": "hybrid",
    "top_k": 10
  }
}
```

**Response — `202 Accepted`**

```json
{
  "eval_job_id": "eval_m1n2o3",
  "status": "queued",
  "dataset": "rag-eval-v1",
  "sample_size": 50,
  "created_at": "2026-07-27T12:00:00Z"
}
```

---

### 5.3 GET `/api/v1/pipeline/metrics`

Returns pipeline performance metrics collected via OpenTelemetry.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window` | string | `1h` | Time window: `1h`, `24h`, `7d`, `30d` |
| `metric` | string | — | Filter to specific metric name |

**Response — `200 OK`**

```json
{
  "window": "24h",
  "metrics": {
    "queries_total": 1284,
    "queries_per_minute": 0.89,
    "avg_latency_ms": 1847,
    "p50_latency_ms": 1520,
    "p95_latency_ms": 3200,
    "p99_latency_ms": 5100,
    "retrieval": {
      "avg_latency_ms": 555,
      "avg_chunks_retrieved": 10.2,
      "rerank_rate": 0.73
    },
    "generation": {
      "avg_latency_ms": 1290,
      "avg_tokens_generated": 412,
      "timeout_rate": 0.003
    },
    "documents": {
      "total": 342,
      "total_chunks": 15420,
      "indexed": 338,
      "failed": 4
    },
    "errors": {
      "total": 12,
      "by_type": {
        "GENERATION_TIMEOUT": 4,
        "RETRIEVAL_EMPTY": 6,
        "LLM_ERROR": 2
      }
    }
  },
  "collected_at": "2026-07-27T12:00:00Z"
}
```

---

### 5.4 GET `/api/v1/health`

Simple liveness probe for load balancers and container orchestrators.

**Response — `200 OK`**

```json
{
  "status": "ok"
}
```

No authentication required. No dependency checks.

---

## 6. Data Models

All schemas below use Pydantic v2. These models are shared between request validation and response serialization.

### 6.1 Core Models

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# --- Enums ---

class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED = "fixed"


# --- Document Models ---

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    collection: str
    status: DocumentStatus
    mime_type: str
    file_size_bytes: int
    chunk_count: int
    metadata: dict = {}
    created_at: str
    updated_at: str

class ChunkSummary(BaseModel):
    chunk_id: str
    index: int
    content_preview: str
    token_count: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None

class DocumentDetail(DocumentMetadata):
    storage_key: Optional[str] = None
    processing: Optional[dict] = None
    chunks: list[ChunkSummary] = []


# --- Chunk Models ---

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    score: float
    rerank_score: Optional[float] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rank: Optional[int] = None
    citation_index: Optional[int] = None
    metadata: dict = {}


# --- Configuration Models ---

class RetrievalConfig(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    rerank_top_k: int = Field(default=20, ge=1, le=200)
    filters: dict = {}
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

class GenerationConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=16384)
    system_prompt: Optional[str] = None
    stream: bool = False

class LatencyBreakdown(BaseModel):
    embedding: Optional[int] = None
    bm25: Optional[int] = None
    vector: Optional[int] = None
    fusion: Optional[int] = None
    reranking: Optional[int] = None
    generation: Optional[int] = None
```

### 6.2 Request Models

```python
class IngestRequest(BaseModel):
    sources: list[dict] = []
    collection: str = "default"
    metadata: dict = {}
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

class ReindexRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1, max_length=50)
    chunking_strategy: Optional[ChunkingStrategy] = None
    embedding_model: Optional[str] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    collection: str = "default"
    conversation_id: Optional[str] = None
    retrieval_config: RetrievalConfig = RetrievalConfig()
    generation_config: GenerationConfig = GenerationConfig()

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    collection: str = "default"
    top_k: int = Field(default=10, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    rerank_top_k: int = Field(default=20, ge=1, le=200)
    filters: dict = {}
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    collection: str = "default"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict = {}

class HybridSearchRequest(SearchRequest):
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rerank: bool = False

class EvaluateRequest(BaseModel):
    dataset: str
    sample_size: int = Field(default=50, ge=1, le=1000)
    metrics: list[str] = ["precision@5", "recall@10", "mrr"]
    retrieval_config: RetrievalConfig = RetrievalConfig()
```

### 6.3 Response Models

```python
class ErrorResponse(BaseModel):
    error: ErrorDetail

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None

class PaginationMeta(BaseModel):
    next_cursor: Optional[str] = None
    has_more: bool = False
    total_count: Optional[int] = None

class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    pagination: PaginationMeta

class IngestResponse(BaseModel):
    job_id: str
    status: str
    documents: list[dict]
    created_at: str

class ReindexResponse(BaseModel):
    job_id: str
    status: str
    document_count: int
    created_at: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Chunk]
    conversation_id: Optional[str] = None
    metadata: dict

class RetrieveResponse(BaseModel):
    results: list[Chunk]
    metadata: dict

class SearchResponse(BaseModel):
    results: list[Chunk]
    metadata: dict

class PipelineStatusResponse(BaseModel):
    status: str
    components: dict
    checked_at: str

class PipelineMetricsResponse(BaseModel):
    window: str
    metrics: dict
    collected_at: str
```

---

## 7. Configuration

### 7.1 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `API_WORKERS` | `4` | Uvicorn worker count |
| `API_LOG_LEVEL` | `info` | Logging level |
| `API_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `API_PREFIX` | `/api/v1` | URL prefix for all routes |
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `AUTH_JWKS_URL` | — | JWKS endpoint for token validation |
| `AUTH_AUDIENCE` | — | Expected token audience |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_DEFAULT` | `30` | Default requests/minute for anonymous |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum file upload size |
| `DEFAULT_COLLECTION` | `default` | Default document collection |
| `OPENSEARCH_URL` | `http://localhost:9200` | OpenSearch endpoint |
| `OPENSEARCH_INDEX_PREFIX` | `rag` | Index name prefix |
| `POSTGRES_URL` | `postgresql://localhost:5432/rag` | PostgreSQL connection string |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `MINIO_BUCKET` | `rag-documents` | Object storage bucket |
| `MINIO_ACCESS_KEY` | — | MinIO access key |
| `MINIO_SECRET_KEY` | — | MinIO secret key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model |
| `EMBEDDING_DIMENSION` | `1536` | Embedding vector dimension |
| `EMBEDDING_API_KEY` | — | API key for embedding provider |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `LLM_PROVIDER` | `openai` | LLM provider |
| `LLM_MODEL` | `gpt-4o` | Default LLM model |
| `LLM_API_KEY` | — | LLM provider API key |
| `LLM_TIMEOUT_MS` | `30000` | LLM generation timeout |
| `OTEL_EXPORTER_ENDPOINT` | `http://localhost:4317` | OpenTelemetry collector |
| `OTEL_SERVICE_NAME` | `rag-api` | OpenTelemetry service name |

### 7.2 Configuration File Loading

The API supports YAML configuration files as an alternative to environment variables. Loading priority:

1. Environment variables (highest priority)
2. `config/api.yaml` in the working directory
3. `/etc/rag-api/config.yaml` (system-wide)
4. Built-in defaults (lowest priority)

Example `config/api.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins:
    - "http://localhost:3000"

auth:
  enabled: false

opensearch:
  url: "http://localhost:9200"
  index_prefix: "rag"

embedding:
  model: "text-embedding-3-small"
  dimension: 1536

llm:
  provider: "openai"
  model: "gpt-4o"
  timeout_ms: 30000
```

### 7.3 API Configuration Options

| Setting | Type | Default | Description |
|---|---|---|---|
| `request_id_header` | string | `X-Request-ID` | Header name for request tracing |
| `response_compression` | boolean | `true` | Enable gzip response compression |
| `docs_enabled` | boolean | `true` | Enable Swagger UI at `/docs` |
| `openapi_path` | string | `/openapi.json` | OpenAPI schema path |
| `default_page_size` | integer | `20` | Default pagination size |
| `max_page_size` | integer | `100` | Maximum pagination size |
| `timeout_seconds` | integer | `60` | Global request timeout |

---

## Appendix: HTTP Status Code Summary

| Code | Meaning | When Used |
|---|---|---|
| `200` | OK | Successful read/update |
| `202` | Accepted | Async job queued (ingest, reindex, evaluate) |
| `400` | Bad Request | Malformed request, invalid file type |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient role permissions |
| `404` | Not Found | Document or resource not found |
| `408` | Request Timeout | LLM generation timed out |
| `409` | Conflict | Duplicate operation (e.g., reindex in progress) |
| `413` | Payload Too Large | File upload exceeds size limit |
| `422` | Unprocessable Entity | Pydantic validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Unhandled exception |
| `502` | Bad Gateway | Upstream provider (LLM, embedding) failure |
| `503` | Service Unavailable | Dependency unhealthy (OpenSearch, Postgres) |
