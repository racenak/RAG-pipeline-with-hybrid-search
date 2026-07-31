# System Architecture

This document provides a high-level overview of the RAG pipeline with hybrid search. It describes the system components, their interactions, technology choices, and the rationale behind each decision.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Technology Stack](#5-technology-stack)
6. [Data Flow](#6-data-flow)
7. [Configuration Management](#7-configuration-management)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Cross-Cutting Concerns](#9-cross-cutting-concerns)
10. [Detailed Documentation Index](#10-detailed-documentation-index)

---

## 1. System Overview

### 1.1 Purpose

This system implements a production-grade Retrieval-Augmented Generation (RAG) pipeline that combines **dense vector retrieval** with **sparse BM25 retrieval** using **Reciprocal Rank Fusion (RRF)** and **cross-encoder reranking** to achieve high-quality document retrieval and answer generation.

### 1.2 Key Capabilities

| Capability | Description |
|---|---|
| **Multi-format ingestion** | PDF, DOCX, TXT, Markdown, HTML, CSV |
| **Semantic chunking** | Paragraph/heading-aware splitting with configurable overlap |
| **Dual retrieval** | Dense (embedding) + sparse (BM25) in parallel |
| **Rank fusion** | Reciprocal Rank Fusion merges results from both engines |
| **Cross-encoder reranking** | Re-ranks fused results for maximum precision |
| **Streaming generation** | SSE streaming for low time-to-first-token |
| **Full observability** | Structured logging, OpenTelemetry tracing, Prometheus metrics |
| **Automated evaluation** | Retrieval metrics, answer quality, regression detection |

### 1.3 Design Goals

1. **Quality**: Hybrid retrieval + reranking yields better results than any single method.
2. **Modularity**: Each pipeline stage is independently configurable and replaceable.
3. **Observability**: Every stage emits structured logs, traces, and metrics.
4. **Testability**: Evaluation framework catches regressions automatically.
5. **Operability**: Health checks, status endpoints, and dashboards for production use.

---

## 2. Architecture Principles

| Principle | Rationale |
|---|---|
| **Separation of concerns** | Each module handles one stage of the pipeline (ingestion, retrieval, generation, evaluation). |
| **Configuration over code** | Behavior is driven by YAML config files, not hardcoded values. |
| **Fail gracefully** | Each stage has error handling, retries, and fallbacks. |
| **Measure everything** | Latency, throughput, and quality metrics at every stage. |
| **Lean abstractions** | Interfaces are minimal. No over-engineering for hypothetical future needs. |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          API Layer (FastAPI)                        │
│  /api/v1/documents/*    /api/v1/query    /api/v1/search/*          │
└──────────┬────────────────────────────┬────────────────────────────┘
           │                            │
           v                            v
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│   Data Pipeline      │  │          Query Pipeline                  │
│   (Ingestion)        │  │                                          │
│                      │  │  ┌─────────────┐                         │
│  1. Parse            │  │  │ 1. Query    │  (optional)             │
│  2. Chunk            │  │  │    Rewrite  │                         │
│  3. Extract Metadata │  │  └──────┬──────┘                         │
│  4. Generate Embeds  │  │         v                                │
│  5. Index            │  │  ┌──────┴──────────────────────┐         │
│                      │  │  │ 2. Parallel Retrieval       │         │
│                      │  │  │    ├── Vector (dense)       │         │
│                      │  │  │    └── BM25  (sparse)       │         │
│                      │  │  └──────┬──────────────────────┘         │
│                      │  │         v                                │
│                      │  │  ┌──────┴──────┐                         │
│                      │  │  │ 3. RRF      │  Reciprocal Rank Fusion │
│                      │  │  └──────┬──────┘                         │
│                      │  │         v                                │
│                      │  │  ┌──────┴──────┐                         │
│                      │  │  │ 4. Rerank   │  Cross-encoder          │
│                      │  │  └──────┬──────┘                         │
│                      │  │         v                                │
│                      │  │  ┌──────┴──────┐                         │
│                      │  │  │ 5. Context  │  Assemble prompt        │
│                      │  │  │    Build    │                         │
│                      │  │  └──────┬──────┘                         │
│                      │  │         v                                │
│                      │  │  ┌──────┴──────┐                         │
│                      │  │  │ 6. LLM      │  Streaming generation   │
│                      │  │  │    Generate │                         │
│                      │  │  └─────────────┘                         │
└──────────┬───────────┘  └──────────┬──────────────────────────────┘
           │                          │
           v                          v
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Raw Files    │  │  Vector +    │  │  Document/Chunk Metadata │  │
│  │  (SeaweedFS)  │  │  BM25 Index  │  │  (PostgreSQL)            │  │
│  │               │  │  (OpenSearch)│  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                          │
           v                          v
┌─────────────────────────────────────────────────────────────────────┐
│                     Observability Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Structured  │  │  OpenTelemetry│  │  Prometheus Metrics      │  │
│  │  Logs (JSON) │  │  Traces       │  │  + Grafana Dashboards    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 API Layer

**Responsibility**: Expose REST endpoints for document management, querying, search, and pipeline operations.

- **Framework**: FastAPI (async, auto-generated OpenAPI docs)
- **Auth**: JWT Bearer tokens with role-based access (reader/writer/admin)
- **Rate limiting**: Tiered rate limits per API key
- **Streaming**: Server-Sent Events (SSE) for query responses

See [api.md](./api.md) for endpoint specifications.

### 4.2 Data Pipeline (Ingestion)

**Responsibility**: Take raw documents from ingestion through to indexed, searchable chunks.

| Stage | Description |
|---|---|
| **Parsing** | Format-specific parsers extract text and structure |
| **Chunking** | Semantic splitting by paragraph/heading boundaries |
| **Metadata** | Auto-extract document ID, section headings, page numbers |
| **Embedding** | Generate dense vectors via sentence-transformers or OpenAI |
| **Indexing** | Write to both vector store and BM25 index |

See [data-pipeline.md](./data-pipeline.md) for full details.

### 4.3 Query Pipeline (Retrieval + Generation)

**Responsibility**: Take a user query through retrieval, fusion, reranking, context assembly, and LLM generation.

| Stage | Description |
|---|---|
| **Query Rewrite** | Optional: expand, rephrase, or generate HyDE queries |
| **Parallel Retrieval** | Run vector search and BM25 concurrently |
| **RRF Fusion** | Merge ranked lists using Reciprocal Rank Fusion |
| **Reranking** | Cross-encoder scores final candidates |
| **Context Build** | Assemble chunks into LLM prompt within token budget |
| **LLM Generate** | Streaming response with source citations |

See [retrieval.md](./retrieval.md) for full details.

### 4.4 Storage Layer

| Store | Purpose | Options |
|---|---|---|
| **Vector Store** | Dense embedding similarity search | FAISS, Chroma, Qdrant, pgvector |
| **BM25 Index** | Sparse keyword-based retrieval | Whoosh, rank_bm25, Elasticsearch |
| **Metadata Store** | Document/chunk tracking, lineage | PostgreSQL, SQLite |
| **Embedding Cache** | Avoid re-computing embeddings for unchanged content | File-based, Redis |

### 4.5 Observability Layer

| Component | Purpose |
|---|---|
| **Structured Logs** | JSON logs with correlation IDs for debugging |
| **OpenTelemetry Traces** | Distributed tracing across pipeline stages |
| **Prometheus Metrics** | Latency histograms, counters, gauges |
| **Grafana Dashboards** | Real-time monitoring and alerting |

---

## 5. Technology Stack

### 5.1 Core

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Rich ML/NLP ecosystem, async support |
| **Web Framework** | FastAPI | Auto-generated OpenAPI, async, Pydantic validation |
| **Config** | YAML files + pydantic-settings | Type-safe, mergeable configuration |
| **HTTP Client** | httpx | Async HTTP for external API calls |

### 5.2 Embedding & Generation

| Component | Technology | Rationale |
|---|---|---|
| **Embeddings** | sentence-transformers, OpenAI API | Local for speed, API for flexibility |
| **Reranker** | cross-encoder/ms-marco, bge-reranker | Proven cross-encoder models |
| **LLM** | OpenAI GPT-4 / local models | Production quality, with local fallback |

### 5.3 Storage

| Component | Technology | Rationale |
|---|---|---|
| **File Storage** | SeaweedFS | Raw file storage (PDF, DOCX, etc.) — distributed, scalable |
| **Vector + BM25** | OpenSearch | kNN vector search + BM25 text search in one service |
| **BM25** | Whoosh, rank_bm25 | Lightweight, no external dependencies |
| **Metadata DB** | PostgreSQL (prod), SQLite (dev) | Reliable, well-understood |

### 5.4 Observability

| Component | Technology | Rationale |
|---|---|---|
| **Tracing** | OpenTelemetry | Vendor-neutral, industry standard |
| **Metrics** | Prometheus | Pull-based, widely supported |
| **Dashboards** | Grafana | Rich visualization, alerting |
| **Logging** | structlog / stdlib JSON | Structured, searchable |

---

## 6. Data Flow

### 6.1 Ingestion Flow

```
Document Upload
      │
      v
┌─────┴─────┐
│  Validate  │  Check file type, size, hash for dedup
└─────┬─────┘
      │
      v
┌─────┴─────┐
│   Parse    │  Format-specific text extraction
└─────┬─────┘
      │
      v
┌─────┴─────┐
│   Chunk    │  Semantic splitting (paragraphs/headings)
└─────┬─────┘
      │
      v
┌─────┴──────┐
│  Extract   │  Document ID, heading path, page numbers
│  Metadata  │
└─────┬──────┘
      │
      v
┌─────┴──────┐
│  Embed     │  Batch embedding generation (cache-aware)
└─────┬──────┘
      │
      v
┌─────┴──────┐
│  Index     │  Write to vector store + BM25 index + metadata DB
└────────────┘
```

### 6.2 Query Flow

```
User Query
      │
      v
┌─────┴──────┐
│  Rewrite   │  Optional: expand, HyDE, multi-query
└─────┬──────┘
      │
      ├──────────────────┐
      v                  v
┌─────┴──────┐    ┌─────┴──────┐
│  Vector    │    │  BM25      │   Runs in parallel
│  Search    │    │  Search    │
└─────┬──────┘    └─────┬──────┘
      │                  │
      └──────┬───────────┘
             v
      ┌──────┴──────┐
      │    RRF      │   Reciprocal Rank Fusion (k=60)
      └──────┬──────┘
             v
      ┌──────┴──────┐
      │   Rerank    │   Cross-encoder scoring
      └──────┬──────┘
             v
      ┌──────┴──────┐
      │  Context    │   Token budget management
      │  Build      │   + source attribution
      └──────┬──────┘
             v
      ┌──────┴──────┐
      │  LLM Gen    │   Streaming response with citations
      └─────────────┘
```

---

## 7. Configuration Management

Configuration is hierarchical and environment-driven:

```
config/
├── defaults.yaml          # System-wide defaults
├── retrieval.yaml         # Retrieval pipeline settings
├── embedding.yaml         # Embedding model config
├── evaluation.yaml        # Evaluation framework settings
└── logging.yaml           # Observability configuration
```

**Override priority**: Environment variables > Local config > Defaults

Example configuration hierarchy:

```yaml
# defaults.yaml
retrieval:
  vector:
    top_k: 20
    similarity_threshold: 0.7
  bm25:
    top_k: 20
    k1: 1.5
    b: 0.75
  rrf:
    k: 60
  rerank:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 10

# Environment override
# RETRIEVAL_VECTOR_TOP_K=30
```

See individual documentation files for complete configuration schemas.

---

## 8. Deployment Architecture

### 8.1 Development

```
┌─────────────────────────────┐
│  Single Process             │
│  FastAPI + Uvicorn          │
│  SQLite + FAISS (in-memory) │
│  Local embeddings           │
└─────────────────────────────┘
```

### 8.2 Production

```
┌──────────────────────────────────────────┐
│  Load Balancer (nginx / ALB)             │
└──────────┬───────────────────────────────┘
           │
    ┌──────┴──────┐
    │  FastAPI    │  x N (horizontal scaling)
    │  Workers    │
    └──────┬──────┘
           │
    ┌──────┴──────────────────────────┐
    │  Shared Storage                 │
    │  ├── PostgreSQL (metadata)      │
    │  ├── Qdrant (vector + BM25)     │
    │  └── Redis (caching)            │
    └──────┬──────────────────────────┘
           │
    ┌──────┴──────────────────────────┐
    │  Observability                  │
    │  ├── OpenTelemetry Collector    │
    │  ├── Prometheus                 │
    │  └── Grafana                    │
    └─────────────────────────────────┘
```

---

## 9. Cross-Cutting Concerns

### 9.1 Error Handling

Each pipeline stage implements:
- **Retries** with exponential backoff for transient failures
- **Graceful degradation** (e.g., skip reranking if reranker is unavailable)
- **Structured error responses** with error codes and context

### 9.2 Security

- **Auth**: JWT Bearer tokens with role-based access control
- **Input validation**: Pydantic models with field constraints
- **Rate limiting**: Per-key rate limits with tiered quotas
- **Secrets**: Environment variables, never in code or config files

### 9.3 Performance

- **Async I/O**: FastAPI async handlers, async embedding calls
- **Batch processing**: Batch embedding and indexing
- **Parallel retrieval**: Vector + BM25 run concurrently
- **Caching**: Embedding cache avoids redundant computation
- **Connection pooling**: Database and HTTP connection pools

### 9.4 Evaluation

- **Golden dataset**: Curated query/answer/document sets
- **Automated metrics**: Retrieval quality (MRR, NDCG), answer quality (faithfulness, relevance)
- **Regression detection**: Compare runs over time with configurable thresholds
- **CI/CD integration**: Evaluation runs on every pipeline change

See [evaluation.md](./evaluation.md) for full details.

---

## 10. Detailed Documentation Index

| Document | Scope | Lines |
|---|---|---|
| [data-pipeline.md](./data-pipeline.md) | Ingestion, parsing, chunking, metadata, embeddings, indexing | ~1400 |
| [retrieval.md](./retrieval.md) | Vector search, BM25, hybrid, RRF, reranking, context construction | ~1450 |
| [evaluation.md](./evaluation.md) | Metrics, benchmarks, observability, CI/CD | ~1250 |
| [api.md](./api.md) | REST endpoints, request/response schemas, configuration | ~1200 |

---

## Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Hybrid over vector-only | Vector + BM25 + RRF | BM25 handles keyword matching that embeddings miss; RRF is robust to score scale differences |
| RRF over score-based fusion | RRF | No normalization needed, works across different score scales, well-studied |
| Cross-encoder reranking | Yes | 10-15% improvement in retrieval quality at acceptable latency cost |
| Semantic chunking | Paragraph/heading-aware | Preserves document structure and context boundaries |
| FastAPI over Flask/Django | FastAPI | Native async, auto OpenAPI, Pydantic validation |
| OpenTelemetry | Yes | Vendor-neutral tracing, widely adopted |
