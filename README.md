# RAG Pipeline with Hybrid Search

> Production-grade Retrieval-Augmented Generation system with query processing, hybrid search, reranking, and multi-layer caching — built from scratch in Python.

## Features

- **Hybrid Search** — BM25 (sparse) + kNN (dense) with Reciprocal Rank Fusion
- **Query Processing** — normalization, validation, type detection, rewriting (HyDE, Multi-Query, Step-Back)
- **Cross-Encoder Reranking** — MiniLM-L-6-v2 for result quality
- **RAG Generation** — context assembly, prompt templates, OpenRouter/OpenAI/Ollama with streaming
- **Citations** — automatic extraction, mapping, validation
- **Incremental Ingestion** — SHA-256 dedup, hash-based change detection
- **Evaluation** — 62 golden test cases, retrieval metrics (precision, recall, MRR, NDCG), generation metrics (ROUGE, BLEU)
- **Observability** — OTel Collector, Loki, Tempo, Prometheus, Grafana
- **Reliability** — circuit breaker, retry with backoff, graceful degradation

## Quick Start

### Prerequisites

- Python 3.13+
- Podman (or Docker)
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone and install

```bash
git clone https://github.com/vu/RAG-pipeline-with-hybrid-search.git
cd RAG-pipeline-with-hybrid-search
python -m venv .venv
source .venv/bin/activate
uv sync
```

### 2. Start infrastructure

```bash
podman compose up -d
```

This starts: OpenSearch, PostgreSQL, Redis, SeaweedFS, Prometheus, Grafana, OTel Collector, Loki, Tempo.

### 3. Configure LLM (optional)

Add your OpenRouter API key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 4. Ingest documents

```bash
# Ingest a file
python scripts/ingest.py file --path document.pdf

# Ingest a URL
python scripts/ingest.py url --url https://docs.example.com

# Crawl a website
python scripts/crawl.py --url https://docs.example.com --limit 50
```

### 5. Search and generate

```bash
# Search
python scripts/search.py "What is the embedding dimension?"

# Generate answer with citations
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the embedding dimension?", "mode": "hybrid"}'
```

### 6. Run tests

```bash
make test           # All 805 tests
make coverage       # Coverage report
```

## Installation

### From source

```bash
git clone https://github.com/vu/RAG-pipeline-with-hybrid-search.git
cd RAG-pipeline-with-hybrid-search
python -m venv .venv
source .venv/bin/activate
uv sync
```

### Docker/Podman

```bash
podman build -t rag-pipeline .
podman run -p 8000:8000 rag-pipeline
```

### Dependencies

Core dependencies (installed automatically via `uv sync`):
- **Web**: FastAPI, Uvicorn, Pydantic v2
- **Embeddings**: Sentence-Transformers (BAAI/bge-large-en-v1.5)
- **Search**: OpenSearch, FAISS, rank-bm25
- **Storage**: PostgreSQL, Redis, SeaweedFS (S3)
- **LLM**: OpenAI SDK (OpenRouter compatible)
- **Observability**: OpenTelemetry, Prometheus

## Configuration

All settings are in `config/defaults.yaml` with environment variable overrides.

### Key configuration

```yaml
embedding:
  model: "BAAI/bge-large-en-v1.5"
  dimension: 1024
  batch_size: 64

retrieval:
  vector_top_k: 20
  bm25_top_k: 20
  rrf_k: 60
  rerank_enabled: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

generation:
  provider: "openrouter"
  model: "inclusionai/ling-3.0-flash:free"
  temperature: 0.7
  max_tokens: 1024
```

### Environment variables

Override any setting with environment variables:

```bash
# Storage
OPENSEARCH_HOST=localhost
POSTGRES_HOST=localhost
REDIS_HOST=localhost

# LLM
GENERATION_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...

# Security
API_KEYS=your-production-key-here
```

See [docs/environment-variables.md](docs/environment-variables.md) for the full reference.

## API Usage

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "mode": "hybrid", "top_k": 10}'
```

### Generate (RAG)

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain hybrid search", "mode": "hybrid"}'
```

### Stream generation

```bash
curl -X POST http://localhost:8000/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "mode": "hybrid"}'
```

### Ingest file

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@document.pdf"
```

### List documents

```bash
curl http://localhost:8000/api/v1/documents
```

### Health check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Full API docs: http://localhost:8000/docs

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Application                                      │
│  POST /search  POST /generate  POST /ingest  GET /documents  DELETE /documents/{id}  │
└──────────────┬──────────────────────────────────────────────────┬────────────────────┘
               │                                                  │
    ┌──────────▼──────────────────────────────┐   ┌───────────────▼──────────────────┐
    │         SEARCH PIPELINE                  │   │      INGESTION PIPELINE          │
    │                                          │   │                                  │
    │  0. Query Processing                     │   │  Fetch (Firecrawl / File)        │
    │     WhitespaceNormalizer                 │   │  ↓                               │
    │     SpecialCharacterHandler              │   │  Clean (8 cleaners)              │
    │     QueryValidator                       │   │  ↓                               │
    │     QueryTypeDetector                    │   │  Incremental (hash check)        │
    │     QueryExpansion / HyDE                │   │  ↓                               │
    │     MultiQuery / StepBack                │   │  Chunk (semantic, 512 tok)       │
    │  ↓                                       │   │  ↓                               │
    │  1. Redis Query Cache → HIT? DONE        │   │  Embed (BGE-Large, 1024-dim)     │
    │  ↓                                       │   │  ↓                               │
    │  2. Embed Query (Redis Emb Cache)        │   │  Index (OpenSearch kNN+BM25)     │
    │  ↓                                       │   │  Store (SeaweedFS S3)            │
    │  3. Search OpenSearch                    │   │  Register (PostgreSQL)           │
    │     ├─ BM25 (sparse, keyword)            │   └──────────────────────────────────┘
    │     ├─ kNN (dense, semantic)             │
    │     └─ RRF Fusion (combine)              │
    │  ↓                                       │
    │  4. Rerank (Cross-Encoder)               │
    │  ↓                                       │
    │  5. PostgreSQL Enrichment                 │
    │  ↓                                       │
    │  6. RAG Generation                       │
    │     Context Assembly (token budget)      │
    │     Prompt Construction (configurable)   │
    │     LLM Backend (OpenAI / Ollama)        │
    │     Streaming (SSE)                      │
    │  ↓                                       │
    │  7. Citations                            │
    │     Extract [1], [2] markers             │
    │     Map to source chunks                 │
    │     Validate against context             │
    │  ↓                                       │
    │  8. Redis Cache Write                    │
    │  ↓                                       │
    │  Response (answer + citations + sources) │
    └──────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Pydantic v2 | REST endpoints, request validation, OpenAPI docs |
| **Query Processing** | Custom pipeline | Preprocessing, validation, type detection, rewriting |
| **Embeddings** | Sentence-Transformers (BAAI/bge-large-en-v1.5) | 1024-dim dense vectors |
| **Vector Search** | OpenSearch kNN (HNSW + FAISS) | Approximate nearest neighbor retrieval |
| **BM25** | OpenSearch match queries | Keyword-based sparse retrieval |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines ranked lists from vector + BM25 |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | Re-score results for precision |
| **Metadata DB** | PostgreSQL 16 | Document registry, dedup, enrichment |
| **Cache** | Redis 7 | Query result cache (5min) + embedding cache (1h) |
| **Object Storage** | SeaweedFS (S3-compatible) | Raw file storage |
| **Ingestion** | Firecrawl API | Website crawling + markdown extraction |
| **Observability** | Prometheus + Grafana | Metrics, dashboards |
| **Containerization** | Podman Compose | 8-service local deployment |

## Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                           │
│  • File upload (PDF, DOCX, MD, HTML, CSV, TXT)                 │
│  • URL (single page via Firecrawl scrape)                       │
│  • Website crawl (full site via Firecrawl crawl)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. VALIDATE                                                     │
│  • Check file exists, compute SHA-256 hash                      │
│  • Dedup check: PostgreSQL find_by_file_hash()                  │
│  → Skip if already ingested                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. PARSE                                                        │
│  • PDF → pdfplumber (extract text + tables)                     │
│  • DOCX → python-docx (headings + paragraphs)                   │
│  • Markdown → regex (strip syntax, preserve structure)           │
│  • HTML → regex (extract text, remove tags)                     │
│  • CSV → pandas (rows → text)                                   │
│  • TXT → raw read                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CLEAN (8-step pipeline)                                      │
│  • Fix encoding (ftfy)                                          │
│  • Normalize unicode (NFKC)                                     │
│  • Decode HTML entities (&amp; → &)                             │
│  • Remove control characters                                    │
│  • Normalize whitespace (collapse spaces, tabs)                 │
│  • Collapse blank lines (max 1)                                 │
│  • Clean PDF artifacts (headers, footers, page numbers)         │
│  • Strip residual HTML tags                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. STORE (S3)                                                   │
│  • Upload raw file to SeaweedFS                                 │
│  • Key: crawl/{group}/{slug}.md (crawl)                         │
│  • Key: uploads/{hash}.{ext} (file upload)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. CHUNK (semantic, token-aware)                                │
│  • Target: 512 tokens per chunk                                 │
│  • Max: 1024 tokens                                             │
│  • Overlap: 50 tokens                                           │
│  • Split on paragraph boundaries                                │
│  • Each chunk gets: id, document_id, index, metadata            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. EMBED (BGE-Large)                                            │
│  • Model: BAAI/bge-large-en-v1.5                                │
│  • Dimension: 1024                                              │
│  • Batch processing (64 chunks/batch)                           │
│  • L2 normalize vectors                                         │
│  • Cache: SHA-256 content-addressed (.npy files)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. INDEX (OpenSearch)                                           │
│  • Bulk index via _bulk API                                     │
│  • Fields: embedding (knn_vector), content, metadata            │
│  • HNSW index (faiss engine, ef_construction=128, m=16)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. REGISTER (PostgreSQL)                                        │
│  • INSERT document (id, filename, source_url, file_hash, ...)   │
│  • INSERT chunks (id, document_id, chunk_index, content, ...)   │
│  • Enables dedup + metadata queries                             │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Formats

| Format | Parser | Extracts |
|--------|--------|----------|
| PDF | pdfplumber | Text, tables, page metadata |
| DOCX | python-docx | Headings, paragraphs, tables |
| Markdown | Custom regex | Text, headings, links |
| HTML | Custom regex | Text, stripped tags |
| CSV | pandas | Rows as text |
| TXT | Built-in | Raw text |
| URL | Firecrawl API | Markdown from web page |
| Website | Firecrawl crawl | Markdown from all pages |

## Search Pipeline

```
Query: "How do you monitor flow runs?"
  │
  ▼
┌─ Step 0: Query Processing ─────────────────────────────────────┐
│  WhitespaceNormalizer → SpecialCharacterHandler → Validator    │
│  QueryTypeDetector → HOW_TO                                     │
│  Rewriting: QueryExpansion, HyDE, MultiQuery, StepBack         │
│  → 5 query variants for better recall                          │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 1: Redis Cache Check ────────────────────────────────────┐
│  SHA256(query:top_k) → GET                                     │
│  HIT? Return cached results (0ms)                              │
│  MISS? Continue...                                              │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 2: Embed Query ──────────────────────────────────────────┐
│  Redis EmbeddingCache → HIT? Use cached vector                 │
│  MISS: sentence-transformers → 1024-dim vector                 │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 3: Hybrid Search ────────────────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ BM25 (sparse)   │    │ kNN (dense)     │                   │
│  │ keyword match    │    │ cosine sim      │                   │
│  └────────┬────────┘    └────────┬────────┘                   │
│           └──────────┬───────────┘                             │
│                      ▼                                         │
│           Reciprocal Rank Fusion (RRF, k=60)                  │
│           × 5 query variants → merge best scores              │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 4: Reranking ────────────────────────────────────────────┐
│  CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)               │
│  Score each (query, chunk) pair → sort by cross-encoder score │
│  Fallback: if model unavailable → return original order        │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 5: PostgreSQL Enrichment ─────────────────────────────────┐
│  Lookup document metadata (filename, chunk_count, created_at)  │
│  Add to result metadata                                         │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ Step 6: Cache & Return ───────────────────────────────────────┐
│  Redis SET (TTL: 5min) — instant for next identical query      │
│  → JSON response                                                │
└────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/rag_pipeline/
├── api/                         # FastAPI application
│   ├── routes/
│   │   ├── documents.py         # Document management (list/get/delete)
│   │   ├── ingest.py            # File/URL/directory ingestion
│   │   └── search.py            # Search with caching + enrichment
│   └── schemas/                 # Pydantic request/response models
├── data/                        # Document processing
│   ├── chunking.py              # Semantic chunker (token-aware)
│   ├── cleaning.py              # 8-step text cleaning pipeline
│   ├── fetchers.py              # Firecrawl scrape + crawl
│   ├── ingestion.py             # File/URL ingestion orchestration
│   ├── models.py                # Document, ParsedChunk dataclasses
│   ├── parsers.py               # 6 format parsers
│   ├── storage.py               # S3-compatible storage (SeaweedFS)
│   └── validation.py            # File validation + hashing
├── embeddings/                  # Embedding generation
│   ├── backends/
│   │   └── sentence_transformers.py
│   ├── cache.py                 # File-based embedding cache
│   └── generator.py             # Batch embedding orchestration
├── retrieval/                   # Search backends
│   ├── bm25.py                  # Pure Python BM25 (Okapi scoring)
│   ├── bm25_index.py            # OpenSearch BM25 (production)
│   ├── hybrid.py                # RRF + score fusion
│   ├── query.py                 # Query processing + rewriting
│   ├── reranking.py             # Cross-encoder reranking
│   ├── search.py                # Unified search interface
│   └── vector.py                # OpenSearch kNN + FAISS backends
├── generation/                  # RAG generation
│   ├── context.py               # Context assembly (token budget, dedup, ordering)
│   ├── prompt.py                # Prompt templates (system + user)
│   ├── llm.py                   # LLM backends (OpenAI, Ollama) with streaming
│   ├── generator.py             # RAG generator orchestrator
│   └── citations.py             # Citation extraction, mapping, validation
├── storage/                     # Data stores
│   ├── opensearch.py            # Index management, bulk indexing, kNN search
│   ├── postgres.py              # Document/chunk registry, dedup, enrichment
│   └── redis_cache.py           # Query cache + embedding cache
├── config.py                    # Pydantic Settings + YAML config
└── pipeline.py                  # Full ingestion pipeline orchestrator
```

## RAG Generation

The generation pipeline retrieves context, builds prompts, and calls an LLM:

1. **Context Assembly** — token budget management, chunk deduplication, XML/markdown formatting
2. **Prompt Construction** — configurable system/user templates with citation instructions
3. **LLM Backend** — OpenRouter, OpenAI, and Ollama backends with async streaming
4. **Streaming** — Server-Sent Events (SSE) for token-by-token response delivery

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Search** | | |
| `POST` | `/api/v1/search` | Hybrid search with reranking |
| **Generation** | | |
| `POST` | `/api/v1/generate` | RAG generation with citations |
| `POST` | `/api/v1/generate/stream` | SSE streaming generation |
| **Ingestion** | | |
| `POST` | `/api/v1/ingest/file` | Upload and ingest a file |
| `POST` | `/api/v1/ingest/url` | Ingest from URL |
| `POST` | `/api/v1/ingest/directory` | Ingest entire directory |
| **Documents** | | |
| `GET` | `/api/v1/documents` | List documents (paginated) |
| `GET` | `/api/v1/documents/{doc_id}` | Get document details |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete document from all stores |
| `POST` | `/api/v1/documents/incremental` | Incremental ingestion (hash-based dedup) |
| `POST` | `/api/v1/documents/batch` | Batch incremental ingestion |
| `GET` | `/api/v1/documents/status/{job_id}` | Check ingestion job status |
| `POST` | `/api/v1/documents/reindex` | Trigger full reindex |
| **Evaluation** | | |
| `POST` | `/api/v1/evaluation/run` | Run evaluation suite |
| `GET` | `/api/v1/evaluation/datasets` | List golden datasets |
| `GET` | `/api/v1/evaluation/datasets/{name}` | Get dataset info |
| `GET` | `/api/v1/evaluation/metrics` | List available metrics |
| **Pipeline** | | |
| `GET` | `/api/v1/pipeline/status` | Component health check |
| `GET` | `/api/v1/pipeline/metrics` | Pipeline metrics |
| **Operations** | | |
| `GET` | `/` | Root endpoint |
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness check |
| `GET` | `/docs` | OpenAPI documentation |

## Incremental Ingestion

- **Content hash tracking** — SHA-256 per document, skip unchanged files on re-ingestion
- **Modified detection** — hash mismatch triggers full re-process (remove old chunks, re-index)
- **Batch processing** — process multiple files with progress tracking
- **Reindex operations** — full/partial reindex with zero-downtime alias swap
- **Status tracking** — per-document status (pending, processing, indexed, error)

## Evaluation

Comprehensive evaluation framework for measuring retrieval and generation quality.

### Available Metrics

| Category | Metrics |
|----------|---------|
| **Retrieval** | precision@k, recall@k, MRR, NDCG@k, hit rate, MAP |
| **Generation** | ROUGE-1, ROUGE-L, BLEU, word overlap, faithfulness, relevance, completeness |
| **Latency** | total_ms, retrieval_ms, generation_ms, ttft_ms, queries_per_second |

### Golden Dataset

62 curated test cases across 5 categories:
- **Factual** — direct lookup queries
- **Multi-hop** — require multiple documents
- **Summarization** — synthesize information across chunks
- **Comparison** — compare entities or concepts
- **Edge cases** — empty context, ambiguous queries

### Running Evaluation

```bash
# CLI evaluation
python -m evaluation.run_eval --dataset evaluation/golden_dataset.json --mode hybrid

# Filter by category
python -m evaluation.run_eval --category factual --difficulty easy

# API endpoint
curl -X POST http://localhost:8000/api/v1/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "retrieval", "top_k": 10}'

# List available datasets
curl http://localhost:8000/api/v1/evaluation/datasets

# Get dataset info
curl http://localhost:8000/api/v1/evaluation/datasets/golden_dataset.json
```

## Citations

Automatic citation extraction from LLM responses:

- **Marker extraction** — regex-based `[1]`, `[2]` detection from generated text
- **Source mapping** — maps citation markers to retrieved chunks with metadata
- **Validation** — flags hallucinated citations and missing references
- **Formats** — inline markers, footnote style, source block with snippets

## Advanced Retrieval

LL-powered query strategies and intelligent routing:

| Strategy | Description | Fallback |
|----------|-------------|----------|
| **LLM Query Expansion** | Generate related terms via LLM | Word splitting |
| **LLM HyDE** | Generate hypothetical documents via LLM | Template prefix |
| **LLM Multi-Query** | Generate paraphrased query variations via LLM | Template paraphrases |
| **LLM Step-Back** | Generate broader context queries via LLM | Key term extraction |
| **Metadata Filtering** | Date range, document type, custom filters | No filtering |
| **Query Classification** | Detect type/complexity/specificity, route to optimal strategy | Default hybrid |

### Query Classification → Strategy Routing

| Query Type | Strategy | Top-K | Rerank |
|------------|----------|-------|--------|
| Factual | Hybrid | 5 | Yes |
| Summarization | Multi-Query | 15 | No |
| Comparison | Hybrid + Rerank | 10 | Yes |
| How-To | Hybrid | 10 | Yes |
| General | Hybrid | 10 | Yes |

## Performance

### Connection Pooling

Singleton clients with connection pooling — no per-request connection overhead:

```python
from rag_pipeline.storage.clients import get_opensearch_client, get_redis_client

client = get_opensearch_client()  # reused across requests
redis = get_redis_client()        # reused across requests
```

### LLM Response Cache

Redis-based content-addressed cache for LLM calls:

```python
from rag_pipeline.generation.llm import get_llm_backend

# With caching enabled
backend = get_llm_backend(use_cache=True)
```

### Benchmarking

```bash
python scripts/benchmark.py --iterations 100 --query "What is RAG?"
python scripts/benchmark.py --output results.json
```

## Experiment Tracking

Track, compare, and report on evaluation experiments:

- **Experiment lifecycle** — start, log metrics, complete/fail
- **Configuration tracking** — model, retrieval mode, parameters
- **Metric logging** — retrieval, generation, latency, cost
- **Comparison** — side-by-side metric comparison with regression detection
- **Reporting** — JSON, Markdown, and trend visualization

### CLI Usage

```bash
# Run experiment
python scripts/run_experiment.py --name "hybrid-baseline" --mode hybrid --top-k 10

# Compare against baseline
python scripts/run_experiment.py --name "hybrid-v2" --mode hybrid --baseline "hybrid-baseline"

# Add tags and save report
python scripts/run_experiment.py --name "bm25-test" --mode bm25 --tag test --output-report report.md
```

### Storage

Experiments are saved as JSON files in the `experiments/` directory (auto-created).

## Observability

Full observability stack with structured logging, distributed tracing, and metrics.

### Components

| Component | Purpose | Port |
|-----------|---------|------|
| **OTel Collector** | Receives/processes/exports telemetry | 4317 (gRPC), 4318 (HTTP) |
| **Loki** | Log aggregation | 3100 |
| **Tempo** | Distributed tracing | 3200 |
| **Prometheus** | Metrics collection | 9090 |
| **Grafana** | Dashboards (auto-provisions data sources) | 3000 |

### Telemetry Pipeline

```
App → OTel Collector → Tempo (traces)
                   → Prometheus (metrics)
                   → Loki (logs)

Grafana ← Prometheus (metrics)
       ← Loki (logs)
       ← Tempo (traces)
```

### Code Modules

- **`observability/logging.py`** — structured JSON logging with correlation IDs
- **`observability/tracing.py`** — OpenTelemetry spans with OTLP export
- **`observability/metrics.py`** — counters, histograms, gauges with RAG convenience functions

### Metrics Endpoint

```
GET /metrics — returns all collected metrics (counters, histograms, gauges)
```

## Testing

```bash
make test                    # Run all 805 tests
pytest tests/unit/ -v        # Unit tests only
pytest tests/ -k "reranking" # Reranking tests
pytest tests/ -k "query"     # Query processing tests
pytest tests/ -k "hybrid"    # Hybrid search tests
```

## Infrastructure Services

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI | 8000 | REST API |
| OpenSearch | 9200 | Vector + BM25 search engine |
| OpenSearch Dashboards | 5601 | Index inspection UI |
| PostgreSQL | 5432 | Document/chunk metadata, dedup |
| Redis | 6379 | Query cache (5min) + embedding cache (1h) |
| SeaweedFS Master | 9333 | S3 storage master |
| SeaweedFS Volume | 8080 | S3 storage volumes |
| SeaweedFS Filer | 8333 | S3 API endpoint |
| OTel Collector | 4317 | OpenTelemetry (traces, metrics, logs) |
| Loki | 3100 | Log aggregation |
| Tempo | 3200 | Distributed tracing |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboards |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System architecture and design decisions |
| [Data Pipeline](docs/data-pipeline.md) | Ingestion, parsing, cleaning, chunking |
| [Retrieval](docs/retrieval.md) | Search, BM25, hybrid, reranking |
| [Evaluation](docs/evaluation.md) | Metrics, golden dataset, experiment tracking |
| [API Reference](docs/api.md) | Complete API documentation |
| [Deployment](docs/deployment.md) | Podman deployment guide |
| [Environment Variables](docs/environment-variables.md) | Configuration reference |
| [Runbook](docs/runbook.md) | Troubleshooting and operations |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Development](DEVELOPMENT.md) | Local development setup |

## License

MIT
