# Runbook

Troubleshooting guide and operational procedures.

## Common Issues

### OpenSearch won't start

**Symptoms**: Container exits immediately, health check fails, `connection refused` on port 9200.

**Fix**:

```bash
# Check memory — OpenSearch needs at least 512MB
podman stats

# Increase VM map count (Linux, required for OpenSearch)
sudo sysctl -w vm.max_map_count=262144

# Make persistent
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf

# Check logs
podman logs rag-opensearch --tail 50
```

### PostgreSQL connection refused

**Symptoms**: `connection refused` errors, API returns 500 on ingest.

**Fix**:

```bash
# Check if running
podman ps | grep postgres

# Check logs
podman logs rag-postgres --tail 20

# Verify pg_hba.conf exists
ls -la config/pg_hba.conf

# Reset data (loses all documents)
podman compose down -v
podman compose up -d postgres

# Wait for health check
podman compose up -d
```

### Redis connection timeout

**Symptoms**: `ConnectionError` in API logs, search returns stale results.

**Fix**:

```bash
# Check Redis is running
podman exec rag-redis redis-cli ping
# Should return: PONG

# Check memory usage
podman exec rag-redis redis-cli info memory

# Clear cache (non-destructive)
podman exec rag-redis redis-cli FLUSHDB

# Check connectivity from API
podman exec rag-api python -c "import redis; r = redis.Redis(host='redis'); r.ping()"
```

### Embedding model download slow

**First run** downloads ~1.3GB model. Subsequent runs use the local cache.

```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Cache location
ls -la .cache/
```

### LLM API errors

**Symptoms**: Generation returns empty responses or 401/403 errors.

**Fix**:

```bash
# Check API key is set
echo $OPENROUTER_API_KEY

# Test with curl
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Check the model name is valid
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "inclusionai/ling-3.0-flash:free", "messages": [{"role": "user", "content": "hi"}]}'
```

### Ingestion hangs or fails

**Symptoms**: File upload returns timeout, ingestion status stuck at `processing`.

**Fix**:

```bash
# Check SeaweedFS is running
curl http://localhost:9333/

# Check disk space
df -h

# Check logs
podman logs rag-api --tail 50

# Reset stuck jobs
curl -X POST http://localhost:8000/api/v1/documents/reindex
```

### Health check fails

**Symptoms**: `GET /ready` returns 503.

```bash
# Check each dependency
curl http://localhost:9200/_cluster/health  # OpenSearch
curl http://localhost:5432                    # PostgreSQL (expect connection error = running)
curl http://localhost:6379                    # Redis (expect NOAUTH = running)

# Full check
curl -s http://localhost:8000/ready | python -m json.tool
```

## Scaling

### Increase concurrent requests

1. Increase Uvicorn workers:

```bash
uvicorn rag_pipeline.api.app:app --workers 4
```

2. Increase PostgreSQL connection pool:

```bash
POSTGRES_POOL_SIZE=10 POSTGRES_MAX_OVERFLOW=20
```

3. Increase Redis connection pool in config.

### Increase ingestion throughput

1. Increase embedding batch size:

```bash
EMBEDDING_BATCH_SIZE=128
```

2. Use multiple API instances behind a load balancer.

### Increase search performance

1. Adjust vector search parameters:

```yaml
retrieval:
  vector_top_k: 30        # retrieve more candidates
  rerank_top_k: 10        # keep final results low
```

2. Tune RRF parameter:

```yaml
retrieval:
  rrf_k: 60              # lower = more weight to top results
```

## Monitoring

### Check metrics

```bash
curl http://localhost:8000/metrics
```

### Check logs

```bash
podman logs rag-api --tail 100

# Follow logs
podman logs -f rag-api
```

### Check traces

1. Open Grafana: http://localhost:3000
2. Navigate to **Explore** → Select **Tempo** data source
3. Search by service name: `rag-pipeline`
4. Filter by operation name (e.g., `search`, `generate`, `ingest`)

### Prometheus queries

```promql
# Request rate
rate(http_requests_total[5m])

# Latency p95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Active ingestion jobs
ingestion_active_jobs
```

## Backup and Restore

### Backup PostgreSQL

```bash
podman exec rag-postgres pg_dump -U rag rag_pipeline > backup_$(date +%Y%m%d).sql
```

### Restore PostgreSQL

```bash
cat backup_20250101.sql | podman exec -i rag-postgres psql -U rag -d rag_pipeline
```

### Backup OpenSearch indices

```bash
# Snapshot API
curl -X PUT "localhost:9200/_snapshot/backup/snapshot_$(date +%Y%m%d)?wait_for_completion=true"
```

### Export Grafana dashboards

```bash
# Export via API
curl -s http://localhost:3000/api/dashboards/db/rag-pipeline > grafana-dashboard.json
```

## Log Reference

### API log fields (JSON)

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "level": "INFO",
  "logger": "rag_pipeline.api.routes.search",
  "message": "Search completed",
  "correlation_id": "abc-123",
  "query": "What is RAG?",
  "mode": "hybrid",
  "results_count": 10,
  "latency_ms": 245
}
```

### Common log patterns

| Pattern | Meaning |
|---------|---------|
| `Connection refused` | Service not running |
| `TimeoutError` | Service overloaded or network issue |
| `OpenSearch` + `bulk` | Indexing operation |
| `Cache HIT` | Query served from Redis cache |
| `Cache MISS` | Query executed against search engine |
