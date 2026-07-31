# Development Guide

## Prerequisites

- Python 3.13+
- Podman (or Docker)
- uv (package manager)

## Quick Start

```bash
# Install dependencies
uv sync

# Start infrastructure
podman compose up -d

# Run the API
uvicorn rag_pipeline.api.app:app --reload

# Run tests
make test
```

## Architecture

See `docs/architecture.md` for full architecture documentation.

## Project Structure

```
src/rag_pipeline/
├── api/            # FastAPI endpoints and middleware
├── data/           # Ingestion, parsing, cleaning, chunking
├── embeddings/     # Embedding generation and caching
├── evaluation/     # Metrics, experiment tracking
├── generation/     # LLM backends, context, prompts, citations
├── observability/  # Logging, tracing, metrics
├── reliability/    # Circuit breaker, retry
├── retrieval/      # Search, BM25, hybrid, reranking
└── storage/        # OpenSearch, PostgreSQL, Redis, S3
```

## Configuration

Settings are loaded from:
1. `config/defaults.yaml` — base defaults
2. `.env` — environment overrides
3. Environment variables — highest priority

## Running Specific Tests

```bash
# Tests for a specific module
pytest tests/unit/test_bm25.py -v

# Tests matching a pattern
pytest tests/ -k "reranking"

# Integration tests only
make test-integration
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
