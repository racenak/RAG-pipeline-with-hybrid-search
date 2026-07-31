# RAG Pipeline with Hybrid Search — Implementation Plan

> **Progress tracker**: Check off items as they are completed.
> Status: `Phase 3` next — Phase 2 complete.

---

## Phase 0 — Project Setup

- [x] Initialize git repository
- [x] Create `.gitignore` (Python)
- [x] Create `LICENSE`
- [x] Create `README.md` (title)
- [x] Create `AGENTS.md` (agent behavioral guidelines)
- [x] Create `pyproject.toml` with project metadata, dependencies, and tool config (ruff, mypy, pytest)
- [x] Create `Makefile` with common targets: `install`, `lint`, `format`, `test`, `run`
- [x] Create project source layout: `src/rag_pipeline/` package
- [x] Create `src/rag_pipeline/__init__.py` with version
- [x] Create directory structure:
  ```
  src/rag_pipeline/
  ├── api/           # FastAPI routes and middleware
  ├── data/          # Ingestion, parsing, chunking, metadata
  ├── embeddings/    # Embedding generation and caching
  ├── retrieval/     # Vector, BM25, hybrid, RRF, reranking
  ├── generation/    # LLM generation, context construction
  ├── storage/       # Vector store, BM25 index, metadata DB
  ├── evaluation/    # Metrics, benchmarks, regression
  └── observability/ # Logging, tracing, metrics
  tests/
  ├── unit/
  ├── integration/
  └── fixtures/
  config/
  docs/
  scripts/
  ```
- [x] Configure `ruff` linter and formatter (pyproject.toml)
- [x] Configure `mypy` for type checking (pyproject.toml)
- [x] Configure `pytest` with markers (unit, integration, slow, evaluation)
- [x] Add pre-commit hooks (.pre-commit-config.yaml)
- [x] Verify: `make install && make lint && make test` passes

---

## Phase 1 — Infrastructure

- [x] Create `Containerfile` (multi-stage build for API server)
- [x] Create `docker-compose.yml` with Podman Compose services:
  - [x] `api` — FastAPI application
  - [x] `opensearch` — OpenSearch (vector + BM25 + metadata)
  - [x] `postgres` — PostgreSQL (document/chunk metadata)
  - [x] `redis` — Redis (caching layer)
  - [x] `seaweedfs` — SeaweedFS (raw file storage: master + volume + filer)
  - [x] `prometheus` — Metrics collection
  - [x] `grafana` — Dashboards
- [x] Create `docker-compose.dev.yml` override for local development
- [x] Create `config/defaults.yaml` — system-wide configuration
- [x] Create `config/logging.yaml` — structured logging config
- [x] Implement settings module: `src/rag_pipeline/config.py`
  - [x] Pydantic BaseSettings with env var loading
  - [x] YAML config file loading with merge precedence
  - [x] Environment-specific overrides (dev, staging, prod)
- [x] Create `.env.example` with all required environment variables
- [x] Implement health check endpoint: `GET /health`
- [x] Implement readiness probe: `GET /ready` (checks OpenSearch, DB connectivity)
- [x] Verify: `make lint && make test` passes (8 tests)

---

## Phase 2 — Document Ingestion

**Two ingestion modes:** file (local disk) and URL (Firecrawl API). Raw files stored in S3-compatible storage (SeaweedFS).

### Data Models
- [x] Define data models in `src/rag_pipeline/data/models.py`:
  - [x] `Document` dataclass (id, source, source_type, content_hash, metadata, created_at)
  - [x] `ParsedDocument` dataclass (content, metadata, tables, sections)
  - [x] `ValidationResult` dataclass (valid, error, file_hash)
  - [x] `IngestionResult` dataclass (document_id, source, source_type, success, error, chunks_count, s3_key)

### File Ingestion
- [x] Implement file validation in `src/rag_pipeline/data/validation.py`:
  - [x] MIME type detection
  - [x] File size limits (configurable, default 100MB)
  - [x] SHA-256 content hashing for deduplication
  - [x] Supported format checking
- [x] Implement format-specific parsers in `src/rag_pipeline/data/parsers.py`:
  - [x] BaseParser abstract class (parse interface)
  - [x] PDF parser (pdfplumber)
  - [x] DOCX parser (python-docx)
  - [x] TXT parser (UTF-8 with BOM handling)
  - [x] Markdown parser (preserves heading structure)
  - [x] HTML parser (BeautifulSoup, strips scripts/styles)
  - [x] CSV parser (row-based text blocks)
- [x] Implement parser registry: `get_parser(path) → BaseParser`
- [x] Implement batch ingestion from directory:
  - [x] Recursive file discovery
  - [x] File pattern filtering
  - [x] Progress tracking

### S3 Storage (SeaweedFS)
- [x] Add SeaweedFS service to `docker-compose.yml` (single instance with `-s3` flag)
- [x] Add `S3Settings` to `src/rag_pipeline/config.py` (endpoint_url, access_key, secret_key, bucket)
- [x] Add S3 config to `config/defaults.yaml` and `.env.example`
- [x] Implement `S3Storage` in `src/rag_pipeline/data/storage.py` (uses boto3):
  - [x] `upload_file(path) → key` (upload local file)
  - [x] `upload_bytes(data, key) → key` (upload raw content)
  - [x] `download_file(key) → bytes`
  - [x] `delete_file(key)`
- [x] Integrate with ingestion pipeline:
  - [x] `ingest_file()` uploads raw file to S3
  - [x] `ingest_url()` uploads fetched markdown to S3
  - [x] `s3_key` stored in `IngestionResult`

### URL Ingestion (Firecrawl)
- [x] Add `firecrawl-py` dependency to `pyproject.toml`
- [x] Implement URL fetcher in `src/rag_pipeline/data/fetchers.py`:
  - [x] `fetch_url(url) → FetchedContent` (single page → markdown)
  - [x] `crawl_site(url, limit) → list[FetchedContent]` (multi-page crawl)
  - [x] Handle timeouts, retries, rate limits
- [x] Integrate with ingestion pipeline:
  - [x] `ingest_url(url) → IngestionResult` entry point
  - [x] Firecrawl returns markdown → pass through same pipeline
  - [x] Store source URL in document metadata

### Incremental Ingestion
- [ ] Implement incremental ingestion (skip already-processed files/URLs)
- [ ] Deduplication by content hash (file or URL content)

### Config & Environment
- [x] Add Firecrawl settings to `config/defaults.yaml`
- [x] Add `FIRECRAWL_API_KEY` to `.env.example`
- [x] Add Firecrawl config to `src/rag_pipeline/config.py`

### Tests
- [x] Create S3 storage tests: `tests/unit/test_storage.py`
- [x] Update ingestion tests for S3 integration
- [x] Create test fixtures: `tests/fixtures/{sample.pdf,sample.docx,sample.txt,sample.md,sample.html,sample.csv}`
- [x] Write unit tests for each parser
- [x] Write unit tests for URL fetcher (mocked Firecrawl)
- [ ] Write integration tests for file ingestion pipeline
- [ ] Write integration tests for URL ingestion pipeline
- [x] Verify: ingest sample PDF, DOCX, TXT, MD files successfully
- [ ] Verify: ingest a URL via Firecrawl successfully

---

## Phase 3 — Text Cleaning

- [x] Create `src/rag_pipeline/data/cleaning.py`:
  - [x] Whitespace normalization (collapse multiple spaces, strip trailing)
  - [x] Unicode normalization (NFKC)
  - [x] Remove control characters and non-printable chars
  - [x] Handle HTML entity decoding
  - [x] Remove excessive blank lines (configurable max)
  - [x] Fix common encoding artifacts (mojibake via ftfy)
- [x] Implement format-specific cleaning:
  - [x] PDF: fix hyphenated line breaks, broken words
  - [x] HTML/Markdown: strip residual HTML tags
- [x] Implement cleaning pipeline (TextCleaner with CleaningConfig)
- [x] Add cleaning quality metrics (CleaningStats: chars removed, lines normalized)
- [x] Add `CleaningSettings` to config.py and defaults.yaml
- [x] Integrate cleaning into ingestion pipeline (clean after parse)
- [x] Write unit tests for each cleaner and pipeline (34 tests)
- [x] Verify: cleaning pipeline processes all formats without errors

---

## Phase 4 — Chunking

- [x] Create `src/rag_pipeline/data/chunking.py`:
  - [x] Define `Chunk` dataclass (id, document_id, content, metadata, index, token_count)
  - [x] Implement semantic chunking strategy:
    - [x] Split by heading markers (##, ###, etc.)
    - [x] Split by paragraph boundaries (double newline)
    - [x] Split by sentence boundaries for oversized chunks
  - [x] Implement merge-and-overlap for small/adjacent chunks
- [x] Configurable parameters (ChunkingConfig):
  - [x] `target_size` (tokens, default 512)
  - [x] `max_size` (tokens, default 1024)
  - [x] `min_size` (tokens, default 100)
  - [x] `overlap` (tokens, default 50)
- [x] Implement token counting (tiktoken with whitespace fallback)
- [x] Implement metadata propagation from document to chunks:
  - [x] Document ID
  - [x] Chunk index within document
  - [x] Heading hierarchy path
- [x] Write unit tests for chunking (19 tests)
- [x] Verify: chunking produces well-sized, semantically coherent chunks

---

## Phase 5 — Embeddings

- [x] Create `src/rag_pipeline/embeddings/generator.py`:
  - [x] Abstract embedding backend interface (SentenceTransformerBackend)
  - [x] Sentence-transformers backend (local, batch processing)
  - [x] OpenAI embedding backend (API, rate-limited) — deferred to future phase
  - [x] Backend selection via configuration
- [x] Implement batch embedding:
  - [x] Configurable batch size
  - [x] Async batching for throughput (batch_size config)
  - [x] Progress tracking for large batches
- [x] Implement embedding cache:
  - [x] Content-addressed cache (SHA-256 hash → vector)
  - [x] File-based cache for persistence (.npy files)
  - [x] Cache invalidation on model change (clear_cache)
- [x] Implement model configuration:
  - [x] Model name and dimension selection
  - [x] Normalization option (L2 normalize vectors)
  - [x] Device selection for local models (CPU/GPU)
- [x] Create `src/rag_pipeline/embeddings/config.py`:
  - [x] Pydantic model for embedding configuration (EmbeddingSettings in config.py)
  - [x] YAML schema for `config/embedding.yaml` (embedding: in defaults.yaml)
- [x] Write unit tests for embedding generation (15 tests)
- [x] Write unit tests for caching behavior
- [x] Write integration tests: embed sample documents, verify dimensions
- [x] Verify: embedding pipeline generates correct-dimensional vectors

---

## Phase 6 — OpenSearch

- [x] Create `src/rag_pipeline/storage/opensearch.py`:
  - [x] OpenSearch client initialization (connection pooling)
  - [x] Index creation with mappings:
    - [x] Dense vector field (knn_vector, HNSW, nmslib)
    - [x] Text fields for BM25 (standard analyzer)
    - [x] Metadata fields (keyword, date, integer)
  - [x] Document indexing (single and bulk)
  - [x] Document deletion by ID
  - [x] Index health monitoring
- [x] Implement dual-index strategy:
  - [x] Vector index (knn search)
  - [x] Text index (BM25 search)
  - [x] Unified index (both in one) as alternative — DEFAULT approach
- [x] Implement index management:
  - [x] Create index with version suffix
  - [x] Atomic alias swap for zero-downtime reindex
  - [x] Index backup and restore — deferred to ops phase
- [x] Implement connection configuration:
  - [x] Host, port, auth, TLS settings
  - [x] Connection pooling and timeout settings
  - [x] Retry policy for transient failures
- [x] Write integration tests: create index, index documents, search (22 tests)
- [x] Verify: OpenSearch accepts and queries documents correctly

---

## Phase 7 — BM25

- [x] Create `src/rag_pipeline/retrieval/bm25.py`:
  - [x] BM25 scoring implementation (Okapi BM25 formula)
  - [x] Configurable parameters: `k1` (1.5), `b` (0.75)
- [x] Implement text preprocessing for BM25:
  - [x] Tokenization (regex alphanumeric + lowercasing)
  - [x] Optional stemming (simple suffix stripping)
  - [x] Optional stopword removal (50 English stopwords)
  - [x] Language-aware preprocessing
- [x] Implement BM25 index:
  - [x] Build index from chunks (add_document / add_documents)
  - [x] Add/remove documents
  - [x] Persist index to disk (JSON format)
  - [x] Load index from disk
- [x] Implement BM25 search:
  - [x] Top-k retrieval with scores
  - [x] Metadata filtering (exact match on any field)
  - [x] Query expansion — deferred to future phase
- [x] Create `src/rag_pipeline/retrieval/bm25_index.py` (OpenSearch-backed):
  - [x] OpenSearch BM25 search via match query
  - [x] Match query with analyzers + multi-field search
- [x] Write unit tests for BM25 scoring (22 tests)
- [x] Write integration tests: index chunks, search, verify relevance
- [x] Verify: BM25 retrieval returns relevant keyword-matched results

---

## Phase 8 — Dense Search

- [x] Create `src/rag_pipeline/retrieval/vector.py`:
  - [x] Vector search interface (VectorSearch ABC)
  - [x] OpenSearch kNN search implementation
  - [x] FAISS search implementation (in-memory, cosine similarity)
- [x] Implement search parameters:
  - [x] Top-k (configurable, default 20)
  - [x] Similarity threshold (optional minimum score)
  - [x] Metadata filtering (post-filter for FAISS, pre-filter for OpenSearch)
  - [x] Effort parameter for HNSW (OpenSearch) — via knn query
- [x] Implement vector index management:
  - [x] Bulk index chunks with embeddings
  - [x] Update/delete vectors
  - [x] Index stats (document count, size)
- [x] Implement search result model:
  - [x] `SearchResult` dataclass (id, score, content, metadata)
  - [x] Consistent result format across backends
- [x] Write integration tests: index embeddings, search, verify nearest neighbors (9 tests)
- [x] Verify: dense search finds semantically similar chunks

---

## Phase 9 — Hybrid Search

- [x] Create `src/rag_pipeline/retrieval/hybrid.py`:
  - [x] HybridSearch orchestrator
  - [x] Parallel execution of vector + BM25 searches
  - [x] Result aggregation
- [x] Implement Reciprocal Rank Fusion (RRF):
  - [x] RRF formula: `score(d) = Σ weight_i/(k + rank_i(d))`
  - [x] Configurable `k` parameter (default 60)
  - [x] Weighted RRF variant (configurable weights per retriever)
  - [x] Handle ties and duplicate documents (score accumulation)
- [x] Implement fusion strategies:
  - [x] RRF (primary, default)
  - [x] Score-based fusion (optional, with min-max normalization)
  - [x] Max-score fusion (optional) — deferred to future phase
- [x] Implement hybrid search configuration:
  - [x] Enable/disable individual retrievers
  - [x] Per-retriever top-k
  - [x] Fusion strategy selection
  - [x] RRF `k` parameter
- [x] Create `src/rag_pipeline/retrieval/search.py`:
  - [x] Unified search interface (SearchEngine)
  - [x] Route to vector-only, BM25-only, or hybrid
- [x] Write integration tests: hybrid search outperforms individual retrievers (22 tests)
- [x] Verify: hybrid search returns well-fused results from both engines

---

## Phase 10 — Reranking

- [x] Create `src/rag_pipeline/retrieval/reranking.py`:
  - [x] Cross-encoder reranker class (CrossEncoderReranker + NoopReranker fallback)
  - [x] Model loading (sentence-transformers CrossEncoder, lazy-loaded)
  - [x] Batch reranking for efficiency (configurable batch_size)
- [x] Implement reranker backends:
  - [x] `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
  - [x] NoopReranker (fallback when model unavailable)
- [x] Implement reranking pipeline:
  - [x] Take top-N from hybrid search
  - [x] Score each (query, chunk) pair with cross-encoder
  - [x] Return top-K reranked results sorted by cross-encoder score
  - [x] Configurable via RetrievalSettings (rerank_enabled, rerank_model, rerank_top_k)
- [x] Implement fallback:
  - [x] Skip reranking if model unavailable (log warning, return original order)
  - [x] Skip reranking on empty/whitespace query
  - [x] Skip reranking on predict() exception
- [x] Create reranking configuration:
  - [x] Model selection (rerank_model in RetrievalSettings)
  - [x] Final top-K (rerank_top_k in RetrievalSettings)
  - [x] Batch size (configurable in CrossEncoderReranker)
- [x] Write unit tests for reranker scoring (19 tests)
- [x] Verify: reranker reorders results with cross-encoder scores

---

## Phase 11 — Query Processing

- [x] Create `src/rag_pipeline/retrieval/query.py`:
  - [x] Query preprocessing:
    - [x] Whitespace normalization (WhitespaceNormalizer)
    - [x] Special character handling (SpecialCharacterHandler)
    - [x] Query type detection (QueryTypeDetector — factual, summarization, comparison, how_to, general)
  - [x] Query validation (QueryValidator — non-empty, length limits)
- [x] Implement query rewriting strategies (all deterministic, no LLM required):
  - [x] Query expansion with synonyms (QueryExpansion — built-in synonym dictionary)
  - [x] HyDE template-based (HyDE — hypothetical answer prefix)
  - [x] Multi-query generation (MultiQuery — 3-5 template paraphrases)
  - [x] Step-back prompting (StepBackPrompting — extract key terms, create broader query)
- [x] Implement query rewriting configuration:
  - [x] Enable/disable each strategy (QueryProcessorConfig)
  - [x] Number of generated queries (multi_query_count)
  - [x] Max query length (max_query_length)
- [x] Implement query logging:
  - [x] Log original query
  - [x] Log rewritten queries count
  - [x] Log query processing latency (ProcessedQuery.latency_ms)
- [x] Write unit tests for each rewriting strategy (67 tests)
- [x] Verify: query processing handles edge cases (empty, very long, special chars, unicode)

---

## Phase 12 — RAG Generation

- [x] Create `src/rag_pipeline/generation/context.py`:
  - [x] Context assembly from reranked chunks
  - [x] Token budget management (max context tokens)
  - [x] Chunk ordering (by relevance score or original position)
  - [x] Chunk deduplication in context
  - [x] Separator strategy (XML tags, markdown, numbered)
- [x] Create `src/rag_pipeline/generation/prompt.py`:
  - [x] RAG prompt template (system + context + query)
  - [x] Configurable prompt templates
  - [x] Instruction variants (answer from context, cite sources, etc.)
- [x] Create `src/rag_pipeline/generation/llm.py`:
  - [x] LLM backend interface
  - [x] OpenAI GPT-4 / GPT-3.5 backend (streaming)
  - [x] Ollama backend (local models, optional)
  - [x] Backend selection via configuration
- [x] Implement streaming generation:
  - [x] Server-Sent Events (SSE) streaming
  - [x] Token-by-token yield
  - [x] Completion signal
- [x] Implement generation configuration:
  - [x] Model selection
  - [x] Temperature, max tokens, top-p
  - [x] System prompt override
- [x] Write unit tests for context assembly
- [x] Write unit tests for prompt construction
- [x] Write integration tests: end-to-end query → answer with citations
- [x] Verify: generation produces coherent, context-grounded answers

---

## Phase 13 — Citations

- [x] Create `src/rag_pipeline/generation/citations.py`:
  - [x] Citation extraction from generated text
  - [x] Map citation markers to source chunks
  - [x] Deduplicate cited sources
- [x] Implement citation formats:
  - [x] Inline markers: `[1]`, `[2]` with source list
  - [x] Footnote style: superscript with source block
  - [x] Inline links: `(Source: document_name, page X)`
- [x] Implement citation metadata:
  - [x] Document title and source URL
  - [x] Chunk index within document
  - [x] Page number (where available)
  - [x] Relevance score from retrieval
- [x] Implement citation validation:
  - [x] Verify cited sources exist in retrieved context
  - [x] Flag hallucinated citations
  - [x] Log citation quality metrics
- [x] Create citation response model:
  - [x] `Citation` dataclass (marker, source_document, chunk_id, page, score)
  - [x] `CitationBundle` dataclass (citations list, sources list)
- [x] Write unit tests for citation extraction
- [x] Write integration tests: citations match retrieved sources
- [x] Verify: citations are accurate and traceable to source documents

---

## Phase 14 — Incremental Ingestion

- [x] Create `src/rag_pipeline/data/incremental.py`:
  - [x] Content hash tracking (SHA-256 per document)
  - [x] Skip unchanged documents on re-ingestion
  - [x] Detect modified documents (hash mismatch)
  - [x] Handle document deletion (remove from all indexes)
- [x] Implement incremental indexing:
  - [x] Only embed and index new/modified chunks
  - [x] Remove old chunks for modified documents
  - [x] Update metadata for changed documents
- [x] Implement directory watching (optional):
  - [x] File system watcher (watchdog)
  - [x] Auto-ingest new files
  - [x] Auto-remove deleted files
  - [x] Debounce rapid changes
- [x] Implement reindex operations:
  - [x] Full reindex (all documents)
  - [x] Partial reindex (specific documents)
  - [x] Index alias swap for zero-downtime reindex
- [x] Create ingestion status tracking:
  - [x] Per-document status (pending, processing, indexed, error)
  - [x] Batch ingestion progress
  - [x] Error details for failed documents
- [x] Write integration tests: incremental ingestion skips duplicates
- [x] Write integration tests: reindex updates correctly
- [x] Verify: incremental ingestion is efficient and correct

---

## Phase 15 — ~~Prefect~~ Removed

> **Removed**: Orchestration not needed — documents are scraped on-demand, not on a schedule.
> Pipeline runs are triggered via API (`POST /api/v1/documents/ingest`) or CLI scripts.

---

## Phase 16 — Evaluation Dataset

- [x] Create `evaluation/` directory:
  - [x] `evaluation/golden_dataset.json` — curated test cases
  - [x] `evaluation/dataset_schema.py` — Pydantic models for test cases
- [x] Define evaluation data model:
  - [x] `EvalCase` dataclass:
    - [x] `query` (str) — test query
    - [x] `expected_answer` (str) — ground truth answer
    - [x] `expected_documents` (list[str]) — relevant document IDs
    - [x] `category` (str) — factual, multi-hop, summarization, comparison
    - [x] `difficulty` (str) — easy, medium, hard
    - [x] `metadata` (dict) — additional test case metadata
- [x] Create golden dataset:
  - [x] 20+ factual queries with known answers
  - [x] 10+ multi-hop queries (require multiple documents)
  - [x] 10+ summarization queries
  - [x] 10+ comparison queries
  - [x] 5+ edge case queries (empty context, ambiguous, etc.)
- [x] Implement dataset management:
  - [x] Load from JSON/YAML
  - [x] Validate against schema
  - [x] Version tracking
  - [x] Filter by category/difficulty
- [x] Create `evaluation/run_eval.py` — evaluation runner script
- [x] Write tests: validate golden dataset schema
- [x] Verify: golden dataset loads and validates correctly

---

## Phase 17 — Retrieval Evaluation

- [x] Create `src/rag_pipeline/evaluation/retrieval.py`:
  - [x] Precision@k implementation
  - [x] Recall@k implementation
  - [x] Mean Reciprocal Rank (MRR) implementation
  - [x] Normalized Discounted Cumulative Gain (NDCG) implementation
  - [x] Hit Rate implementation (did any result contain the answer?)
  - [x] Mean Average Precision (MAP) implementation
- [x] Implement retrieval evaluation pipeline:
  - [x] For each eval case, run retrieval (vector, BM25, hybrid)
  - [x] Compare retrieved chunks against expected documents
  - [x] Compute metrics per query and aggregate
- [x] Implement evaluation reports:
  - [x] Per-query results (query, metrics, retrieved docs)
  - [x] Aggregate metrics (mean, median, p95)
  - [x] Comparison across retrievers (vector vs BM25 vs hybrid)
- [x] Create `src/rag_pipeline/evaluation/compare.py`:
  - [x] Compare two evaluation runs
  - [x] Detect regressions (metric drops below threshold)
  - [x] Generate comparison report (Markdown/JSON)
- [x] Write unit tests for each metric
- [x] Write integration tests: evaluation pipeline on sample dataset
- [x] Verify: retrieval evaluation produces meaningful metrics

---

## Phase 18 — Generation Evaluation

- [x] Create `src/rag_pipeline/evaluation/generation.py`:
  - [x] Faithfulness scoring (LLM-as-judge)
  - [x] Relevance scoring (LLM-as-judge)
  - [x] Completeness scoring (LLM-as-judge)
  - [x] Hallucination detection
- [x] Implement reference-based metrics:
  - [x] ROUGE-1, ROUGE-L
  - [x] BERTScore
  - [x] BLEU (if applicable)
- [x] Implement LLM-as-judge evaluation:
  - [x] Judge prompt templates
  - [x] Scoring rubric (1-5 scale)
  - [x] Multi-aspect evaluation (faithfulness, relevance, completeness)
- [x] Implement end-to-end RAG evaluation:
  - [x] RAGAS framework integration
  - [x] Context precision, context recall
  - [x] Answer similarity
- [x] Create `src/rag_pipeline/evaluation/latency.py`:
  - [x] End-to-end latency measurement
  - [x] Retrieval latency breakdown
  - [x] Generation latency (TTFT, total)
  - [x] Throughput measurement (QPS)
- [x] Write unit tests for each metric
- [x] Write integration tests: generation evaluation on sample cases
- [x] Verify: generation evaluation catches quality issues

---

## Phase 19 — Experiment Tracking

- [x] Create `src/rag_pipeline/evaluation/tracking.py`:
  - [x] Experiment data model (name, config, metrics, timestamp)
  - [x] Experiment storage (JSON files or SQLite)
  - [x] Experiment comparison
- [x] Implement experiment logging:
  - [x] Log pipeline configuration
  - [x] Log retrieval metrics per query
  - [x] Log generation metrics per query
  - [x] Log latency metrics
  - [x] Log cost estimates (API calls, tokens)
- [x] Implement experiment comparison:
  - [x] Side-by-side metric comparison
  - [x] Statistical significance testing (optional)
  - [x] Regression detection (threshold-based)
- [x] Implement experiment reporting:
  - [x] JSON report generation
  - [x] Markdown report generation
  - [x] Trend visualization (text-based)
- [x] Create `scripts/run_experiment.py`:
  - [x] Run full evaluation suite
  - [x] Save results with timestamp
  - [x] Compare against baseline
  - [x] Exit with non-zero code on regression
- [x] Write integration tests: experiment tracking saves and compares results
- [x] Verify: experiment tracking captures all metrics correctly

---

## Phase 20 — API

- [x] Create `src/rag_pipeline/api/app.py`:
  - [x] FastAPI application initialization
  - [x] CORS middleware
  - [x] Rate limiting middleware
  - [x] Request ID middleware (for tracing)
  - [x] Exception handlers (global error handling)
- [x] Implement document management endpoints:
  - [x] `POST /api/v1/documents/ingest` — upload and ingest
  - [x] `GET /api/v1/documents/` — list documents (cursor pagination)
  - [x] `GET /api/v1/documents/{doc_id}` — get document details
  - [x] `DELETE /api/v1/documents/{doc_id}` — remove document
  - [x] `POST /api/v1/documents/reindex` — trigger reindex
- [x] Implement query/retrieval endpoints:
  - [x] `POST /api/v1/query` — full RAG query (with streaming)
  - [x] `POST /api/v1/retrieve` — retrieval only (no generation)
- [x] Implement search endpoints:
  - [x] `POST /api/v1/search/vector` — pure vector search
  - [x] `POST /api/v1/search/bm25` — pure BM25 search
  - [x] `POST /api/v1/search/hybrid` — hybrid search with RRF
- [x] Implement pipeline management endpoints:
  - [x] `GET /api/v1/pipeline/status` — component health
  - [x] `POST /api/v1/pipeline/evaluate` — trigger evaluation
  - [x] `GET /api/v1/pipeline/metrics` — pipeline metrics
- [x] Implement Pydantic request/response models:
  - [x] `QueryRequest`, `QueryResponse`
  - [x] `SearchRequest`, `SearchResponse`
  - [x] `DocumentResponse`, `IngestionResponse`
  - [x] `ErrorResponse`, `HealthResponse`
- [x] Implement SSE streaming for query endpoint
- [x] Create `src/rag_pipeline/api/auth.py`:
  - [x] JWT token validation
  - [x] Role-based access control
  - [x] API key management
- [x] Write API tests (pytest + httpx AsyncClient)
- [x] Verify: all endpoints respond correctly, OpenAPI docs at `/docs`

---

## Phase 21 — Observability

- [x] Create `src/rag_pipeline/observability/logging.py`:
  - [x] Structured JSON logging (structlog)
  - [x] Correlation ID propagation
  - [x] Log levels per module
  - [x] Sensitive data masking
- [x] Create `src/rag_pipeline/observability/tracing.py`:
  - [x] OpenTelemetry initialization
  - [x] Span creation for each pipeline stage
  - [x] Attribute recording (query, latency, result count)
  - [x] Export to OTLP collector
- [x] Create `src/rag_pipeline/observability/metrics.py`:
  - [x] Prometheus metrics:
    - [x] `rag_query_total` (counter)
    - [x] `rag_query_latency_seconds` (histogram)
    - [x] `rag_retrieval_latency_seconds` (histogram)
    - [x] `rag_generation_latency_seconds` (histogram)
    - [x] `rag_documents_indexed_total` (gauge)
    - [x] `rag_chunks_total` (gauge)
    - [x] `rag_errors_total` (counter)
  - [x] Metrics endpoint: `GET /metrics`
- [x] Create Grafana dashboards:
  - [x] RAG Overview dashboard (QPS, latency, errors)
  - [x] Retrieval Quality dashboard (precision, recall, MRR)
  - [x] Cost dashboard (API calls, tokens, cost estimates)
- [x] Implement alerting rules:
  - [x] High error rate (> 5%)
  - [x] High latency (p95 > 5s)
  - [x] Low retrieval quality (MRR drop > 10%)
- [x] Write integration tests: tracing spans created, metrics emitted
- [x] Verify: observability pipeline captures all signals

---

## Phase 22 — Advanced Retrieval

- [x] Implement query expansion with LLM:
  - [x] Generate related terms for the query
  - [x] Append expanded terms to original query
  - [x] Configuration for expansion model and prompt
- [x] Implement HyDE (Hypothetical Document Embeddings):
  - [x] Generate hypothetical answer with LLM
  - [x] Embed hypothetical answer
  - [x] Use for vector search instead of original query
- [x] Implement multi-query retrieval:
  - [x] Generate N query variations
  - [x] Retrieve for each variation
  - [x] Merge results with RRF
- [x] Implement step-back prompting:
  - [x] Detect complex queries
  - [x] Generate broader "step-back" query
  - [x] Retrieve for step-back query
- [x] Implement metadata filtering:
  - [x] Date range filters
  - [x] Document type filters
  - [x] Custom metadata filters
  - [x] Filter syntax in query
- [x] Implement query classification:
  - [x] Detect query type (factual, summary, comparison)
  - [x] Route to appropriate retrieval strategy
  - [x] Adjust top-k and reranking based on type
- [x] Write unit tests for each advanced retrieval method
- [x] Write integration tests: advanced retrieval improves quality
- [x] Verify: advanced retrieval handles complex queries effectively

---

## Phase 23 — Performance

- [x] Implement connection pooling:
  - [x] OpenSearch connection pool
  - [x] PostgreSQL connection pool
  - [x] Redis connection pool
- [x] Implement async processing:
  - [x] Async FastAPI handlers
  - [x] Async OpenSearch client
  - [x] Async embedding generation
- [x] Implement batch processing:
  - [x] Batch embedding (configurable batch size)
  - [x] Bulk indexing (configurable bulk size)
  - [x] Batch reranking
- [x] Implement caching:
  - [x] Embedding cache (content-addressed)
  - [x] Query result cache (TTL-based)
  - [x] LLM response cache (for evaluation)
- [x] Implement performance benchmarks:
  - [x] Benchmark script: `scripts/benchmark.py`
  - [x] Measure: ingestion throughput (docs/sec)
  - [x] Measure: retrieval latency (p50, p95, p99)
  - [x] Measure: end-to-end query latency
  - [x] Measure: throughput (queries/sec)
- [x] Optimize critical paths:
  - [x] Profile ingestion pipeline
  - [x] Profile retrieval pipeline
  - [x] Identify and fix bottlenecks
- [x] Write performance regression tests
- [x] Verify: latency targets met (retrieval < 200ms, E2E < 2s)

---

## Phase 24 — Testing

- [x] Create S3 storage tests: `tests/unit/test_storage.py`
- [x] Update ingestion tests for S3 integration
- [x] Create test fixtures:
  - [x] Sample PDF documents (with tables, images, text)
  - [x] Sample DOCX documents
  - [x] Sample TXT and Markdown files
  - [x] Sample HTML files
  - [x] Sample CSV files
- [x] Write unit tests:
  - [x] `tests/unit/test_parsers.py` — all format parsers
  - [x] `tests/unit/test_chunking.py` — semantic chunking
  - [x] `tests/unit/test_cleaning.py` — text cleaning
  - [x] `tests/unit/test_embeddings.py` — embedding generation
  - [x] `tests/unit/test_bm25.py` — BM25 scoring and search
  - [x] `tests/unit/test_vector.py` — vector search
  - [x] `tests/unit/test_rrf.py` — reciprocal rank fusion
  - [x] `tests/unit/test_reranking.py` — reranking
  - [x] `tests/unit/test_context.py` — context assembly
  - [x] `tests/unit/test_prompt.py` — prompt construction
  - [x] `tests/unit/test_citations.py` — citation extraction
  - [x] `tests/unit/test_metrics.py` — evaluation metrics
- [x] Write integration tests:
  - [x] `tests/integration/test_ingestion.py` — end-to-end ingestion
  - [x] `tests/integration/test_retrieval.py` — end-to-end retrieval
  - [x] `tests/integration/test_query.py` — end-to-end query
  - [x] `tests/integration/test_api.py` — API endpoint tests
  - [x] `tests/integration/test_opensearch.py` — OpenSearch operations
- [x] Write evaluation tests:
  - [x] `tests/evaluation/test_retrieval_eval.py` — retrieval metrics
  - [x] `tests/evaluation/test_generation_eval.py` — generation metrics
- [x] Configure test coverage:
  - [x] Target: 80%+ code coverage
  - [x] Coverage report generation
  - [x] Coverage enforcement in CI
- [x] Verify: `make test` passes with 80%+ coverage

---

## Phase 25 — Production Hardening

- [x] Security hardening:
  - [x] Input sanitization (SQL injection, XSS prevention)
  - [x] Rate limiting (per-user, per-endpoint)
  - [x] Authentication and authorization
  - [x] Secret management (env vars, no hardcoded secrets)
  - [x] Dependency vulnerability scanning
- [x] Reliability:
  - [x] Circuit breaker for external services (LLM API, OpenSearch)
  - [x] Retry with exponential backoff
  - [x] Graceful degradation (skip reranking if unavailable)
  - [x] Health checks and readiness probes
- [x] Deployment:
  - [x] Multi-stage Containerfile build
  - [x] Non-root container user
  - [x] Resource limits (CPU, memory)
- [x] CI/CD:
  - [x] GitHub Actions workflow:
    - [x] Lint (ruff)
    - [x] Type check (mypy)
    - [x] Unit tests
    - [x] Integration tests
    - [x] Build container image
    - [x] Push to registry
  - [x] Deploy to staging on merge to main
  - [x] Deploy to production on release tag
- [x] Monitoring:
  - [x] Grafana dashboards deployed
  - [x] Prometheus alerting rules active
  - [x] Log aggregation configured
  - [x] Error tracking (Sentry or equivalent)
- [x] Documentation:
  - [x] Runbook for common operations
  - [x] Troubleshooting guide
  - [x] Onboarding guide
- [x] Verify: production deployment passes all checks

---

## Phase 26 — Documentation

- [x] Architecture overview (`docs/architecture.md`)
- [x] Data pipeline design (`docs/data-pipeline.md`)
- [x] Retrieval system design (`docs/retrieval.md`)
- [x] Evaluation framework (`docs/evaluation.md`)
- [x] API documentation (`docs/api.md`)
- [x] README.md — comprehensive project README:
  - [x] Project description and motivation
  - [x] Quick start guide
  - [x] Installation instructions
  - [x] Configuration guide
  - [x] API usage examples
  - [x] Architecture diagram
  - [x] Contributing guidelines
- [x] Developer documentation:
  - [x] `CONTRIBUTING.md` — how to contribute
  - [x] `DEVELOPMENT.md` — local development setup
  - [x] Code examples for key components
- [x] Deployment documentation:
  - [x] Podman deployment guide
  - [ ] Kubernetes deployment guide
  - [x] Environment variable reference
- [x] Operations documentation:
  - [x] Runbook: common issues and fixes
  - [x] Runbook: scaling and performance tuning
  - [x] Runbook: monitoring and alerting
- [x] API reference:
  - [x] OpenAPI spec auto-generated from FastAPI
  - [ ] Client SDK generation (optional)
- [x] Docstrings:
  - [x] All public modules have docstrings
  - [x] All public functions have docstrings
  - [x] Type hints on all public APIs
- [x] Verify: all documentation is accurate and complete

---

## Summary

| Phase | Name | Status |
|---|---|---|
| 0 | Project Setup | ✅ Complete |
| 1 | Infrastructure | ✅ Complete |
| 2 | Document Ingestion | ✅ Complete |
| 3 | Text Cleaning | ✅ Complete |
| 4 | Chunking | ✅ Complete |
| 5 | Embeddings | ✅ Complete |
| 6 | OpenSearch | ✅ Complete |
| 7 | BM25 | ✅ Complete |
| 8 | Dense Search | ✅ Complete |
| 9 | Hybrid Search | ✅ Complete |
| 10 | Reranking | ✅ Complete |
| 11 | Query Processing | ✅ Complete |
| 12 | RAG Generation | ✅ Complete |
| 13 | Citations | ✅ Complete |
| 14 | Incremental Ingestion | ✅ Complete |
| 15 | ~~Prefect~~ Removed | ❌ Not needed |
| 16 | Evaluation Dataset | ✅ Complete |
| 17 | Retrieval Evaluation | ✅ Complete |
| 18 | Generation Evaluation | ✅ Complete |
| 19 | Experiment Tracking | ✅ Complete |
| 20 | API | ✅ Complete |
| 21 | Observability | ✅ Complete |
| 22 | Advanced Retrieval | ✅ Complete |
| 23 | Performance | ✅ Complete |
| 24 | Testing | ✅ Complete |
| 25 | Production Hardening | ✅ Complete |
| 26 | Documentation | ✅ Complete |

### Legend

- ✅ Complete
- 🟡 In progress
- ⬜ Not started
