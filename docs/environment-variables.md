# Environment Variables

All configuration is loaded from `config/defaults.yaml` with environment variable overrides. Set any variable in `.env` or export it in your shell.

Loading priority: **Environment variables > Local config > Defaults**

## Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment name (`development`, `staging`, `production`) |
| `APP_DEBUG` | `false` | Enable debug mode |
| `APP_NAME` | `rag-pipeline` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `LOG_FORMAT` | `json` | Log format (`json`, `text`) |

## OpenSearch

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENSEARCH_HOST` | `localhost` | OpenSearch host |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `OPENSEARCH_SCHEME` | `http` | Protocol (`http` or `https`) |
| `OPENSEARCH_INDEX_PREFIX` | `rag` | Index name prefix |
| `OPENSEARCH_TIMEOUT` | `30` | Request timeout in seconds |
| `OPENSEARCH_USERNAME` | `` | Basic auth username |
| `OPENSEARCH_PASSWORD` | `` | Basic auth password |
| `OPENSEARCH_MAX_RETRIES` | `3` | Max retries on failure |
| `OPENSEARCH_RETRY_ON_TIMEOUT` | `true` | Retry on timeout |

## PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `rag_pipeline` | Database name |
| `POSTGRES_USER` | `rag` | Database user |
| `POSTGRES_PASSWORD` | `rag_dev_password` | Database password |
| `POSTGRES_POOL_SIZE` | `5` | Connection pool size |
| `POSTGRES_MAX_OVERFLOW` | `10` | Max overflow connections |

The DSN is constructed as:
`postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}`

## Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_TTL_SECONDS` | `3600` | Default cache TTL |

## Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BACKEND` | `sentence-transformers` | Embedding backend |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_DIMENSION` | `1024` | Vector dimension |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size for embedding |
| `EMBEDDING_NORMALIZE` | `true` | L2 normalize vectors |
| `EMBEDDING_DEVICE` | `cpu` | Compute device (`cpu`, `cuda`) |
| `EMBEDDING_CACHE_ENABLED` | `true` | Enable embedding cache |
| `EMBEDDING_CACHE_DIR` | `.cache/embeddings` | Cache directory |

## Generation (LLM)

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_PROVIDER` | `openrouter` | LLM provider (`openrouter`, `openai`, `ollama`) |
| `GENERATION_MODEL` | `inclusionai/ling-3.0-flash:free` | Model name |
| `GENERATION_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `GENERATION_TEMPERATURE` | `0.7` | Sampling temperature |
| `GENERATION_MAX_TOKENS` | `1024` | Max tokens in response |
| `GENERATION_STREAMING` | `true` | Enable streaming responses |
| `GENERATION_SYSTEM_PROMPT` | `` | Custom system prompt |
| `OPENROUTER_API_KEY` | `` | OpenRouter API key |

## Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKING_STRATEGY` | `semantic` | Chunking strategy |
| `CHUNKING_TARGET_SIZE` | `512` | Target tokens per chunk |
| `CHUNKING_MAX_SIZE` | `1024` | Maximum tokens per chunk |
| `CHUNKING_MIN_SIZE` | `100` | Minimum tokens per chunk |
| `CHUNKING_OVERLAP` | `50` | Token overlap between chunks |

## S3-Compatible Storage (SeaweedFS)

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_ENDPOINT_URL` | `http://localhost:8333` | S3 endpoint URL |
| `S3_ACCESS_KEY` | `anything` | S3 access key |
| `S3_SECRET_KEY` | `anything` | S3 secret key |
| `S3_BUCKET` | `rag-documents` | S3 bucket name |

## Firecrawl (URL Ingestion)

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRECRAWL_API_KEY` | `` | Firecrawl API key (required for URL ingestion) |
| `FIRECRAWL_TIMEOUT` | `30` | Request timeout in seconds |
| `FIRECRAWL_CRAWL_LIMIT` | `100` | Max pages per crawl |

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVABILITY_LOG_LEVEL` | `INFO` | Log level for observability |
| `OBSERVABILITY_LOG_FORMAT` | `json` | Log format |
| `OBSERVABILITY_TRACING_ENABLED` | `true` | Enable distributed tracing |
| `OBSERVABILITY_TRACING_EXPORTER` | `otlp` | Tracing exporter |
| `OBSERVABILITY_TRACING_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |
| `OBSERVABILITY_METRICS_ENABLED` | `true` | Enable metrics collection |
| `OTEL_EXPORTER_ENDPOINT` | `http://localhost:4317` | OTel collector endpoint |
| `OTEL_SERVICE_NAME` | `rag-pipeline` | Service name for traces |

## API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API listen host |
| `API_PORT` | `8000` | API listen port |
| `API_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Rate limit per minute |
| `API_AUTH_ENABLED` | `false` | Enable API key authentication |
| `API_KEYS` | `` | Comma-separated API keys |

## Incremental Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `INCREMENTAL_ENABLED` | `true` | Enable incremental ingestion |
| `INCREMENTAL_HASH_ALGORITHM` | `sha256` | Hash algorithm for dedup |
| `INCREMENTAL_REINDEX_BATCH_SIZE` | `100` | Batch size for reindexing |

## Grafana (Docker)

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFANA_USER` | `admin` | Grafana admin username |
| `GRAFANA_PASSWORD` | `admin` | Grafana admin password |
