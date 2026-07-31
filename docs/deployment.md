# Deployment Guide

## Local Development

```bash
podman compose up -d
```

This starts all infrastructure services: OpenSearch, PostgreSQL, Redis, SeaweedFS, Prometheus, Grafana, OTel Collector, Loki, Tempo.

Verify services are running:

```bash
podman compose ps
curl http://localhost:8000/health
```

## Production Deployment

### 1. Build the image

```bash
podman build -t rag-pipeline:latest .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with production values
```

Key production settings:

```bash
# Use a strong Postgres password
POSTGRES_PASSWORD=<strong-random-password>

# Set your LLM API key
OPENROUTER_API_KEY=sk-or-v1-...

# Enable auth for the API
AUTH_ENABLED=true
API_KEYS=<production-api-key>

# Production logging
LOG_LEVEL=WARNING
LOG_FORMAT=json
```

### 3. Run standalone API

```bash
podman run -d \
  --name rag-api \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  rag-pipeline:latest
```

### 4. Run with infrastructure

```bash
podman compose up -d
```

To run the API container alongside infrastructure, uncomment the `api` service in `docker-compose.yml` and set environment overrides:

```bash
podman compose up -d
```

### 5. Verify deployment

```bash
# Health check
curl http://localhost:8000/health

# Readiness check (verifies OpenSearch, PostgreSQL, Redis)
curl http://localhost:8000/ready

# API docs
curl http://localhost:8000/docs
```

## Resource Requirements

| Component | CPU | Memory |
|-----------|-----|--------|
| API | 0.5 | 512MB |
| OpenSearch | 1.0 | 1GB |
| PostgreSQL | 0.5 | 512MB |
| Redis | 0.25 | 256MB |
| SeaweedFS | 0.25 | 256MB |
| OTel Collector | 0.25 | 256MB |
| Prometheus | 0.25 | 256MB |
| Grafana | 0.5 | 512MB |
| Loki | 0.5 | 512MB |
| Tempo | 0.5 | 512MB |

**Minimum for full stack**: 4 CPU, 8GB RAM
**Minimum for API only**: 0.5 CPU, 512MB RAM (requires external services)

## Health Checks

- **Liveness**: `GET /health` — returns 200 if the API process is running
- **Readiness**: `GET /ready` — returns 200 only if OpenSearch, PostgreSQL, and Redis are reachable

Use these for container orchestration health probes:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"]
  interval: 30s
  timeout: 5s
  start_period: 10s
  retries: 3
```

## Monitoring

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Dashboards, logs, traces |
| Prometheus | http://localhost:9090 | Metrics, alerts |
| OpenSearch Dashboards | http://localhost:5601 | Index inspection |
| API Docs | http://localhost:8000/docs | OpenAPI/Swagger UI |
| Metrics | http://localhost:8000/metrics | Prometheus metrics endpoint |

## Volumes

Named volumes persist data across restarts:

```bash
# List volumes
podman volume ls | grep rag

# Backup PostgreSQL
podman exec rag-postgres pg_dump -U rag rag_pipeline > backup.sql

# Backup OpenSearch indices
curl -X GET "localhost:9200/_snapshot/my_backup/_all" | jq .
```

## Stopping

```bash
# Stop all services (preserves data)
podman compose down

# Stop and remove volumes (deletes all data)
podman compose down -v
```
