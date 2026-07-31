"""FastAPI application — entry point for the RAG pipeline API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_pipeline.api.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    register_exception_handlers,
)
from rag_pipeline.api.routes.documents import router as documents_router
from rag_pipeline.api.routes.evaluation import router as evaluation_router
from rag_pipeline.api.routes.generation import router as generation_router
from rag_pipeline.api.routes.ingest import router as ingest_router
from rag_pipeline.api.routes.search import router as search_router
from rag_pipeline.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle."""
    from rag_pipeline.observability.logging import setup_logging
    from rag_pipeline.observability.metrics import init_metrics
    from rag_pipeline.observability.tracing import init_tracing

    settings = get_settings()
    app.state.settings = settings
    app.state.start_time = time.time()

    setup_logging(settings.observability.log_level, settings.observability.log_format)
    init_tracing("rag-pipeline", settings.observability.tracing_endpoint)
    init_metrics(settings.observability.metrics_enabled)

    yield

    from rag_pipeline.storage.clients import close_all_clients
    close_all_clients()


app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade RAG pipeline with hybrid search",
    version="0.1.0",
    lifespan=lifespan,
)

# ---- Middleware ---------------------------------------------------------------

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

# ---- Exception Handlers ------------------------------------------------------

register_exception_handlers(app)


# ---- Health & Readiness ------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """Liveness probe — confirms the process is running."""
    return JSONResponse({"status": "healthy", "service": "rag-pipeline"})


@app.get("/ready", tags=["ops"])
async def readiness() -> JSONResponse:
    """Readiness probe — checks downstream service connectivity."""
    settings = get_settings()
    checks: dict[str, str] = {}

    # OpenSearch
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            url = (
                f"{settings.storage.opensearch_scheme}://"
                f"{settings.storage.opensearch_host}:{settings.storage.opensearch_port}"
                f"/_cluster/health"
            )
            r = await client.get(url)
            checks["opensearch"] = "healthy" if r.status_code == 200 else "unhealthy"
    except Exception:
        checks["opensearch"] = "unreachable"

    # PostgreSQL
    try:
        import psycopg2

        conn = psycopg2.connect(settings.storage.postgres_dsn, connect_timeout=5)
        conn.close()
        checks["postgres"] = "healthy"
    except Exception:
        checks["postgres"] = "unreachable"

    # Redis
    try:
        import redis

        r = redis.Redis(
            host=settings.storage.redis_host,
            port=settings.storage.redis_port,
            socket_timeout=5,
        )
        r.ping()
        r.close()
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unreachable"

    all_healthy = all(v == "healthy" for v in checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        {
            "status": "ready" if all_healthy else "degraded",
            "checks": checks,
        },
        status_code=status_code,
    )


# ---- Routes ------------------------------------------------------------------

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(generation_router)
app.include_router(evaluation_router)


# ---- Root --------------------------------------------------------------------


@app.get("/metrics", tags=["ops"])
async def metrics() -> dict:
    """Prometheus-compatible metrics endpoint."""
    from rag_pipeline.observability.metrics import get_metrics

    return get_metrics()


@app.get("/", tags=["ops"])
async def root() -> dict[str, str]:
    return {"service": "rag-pipeline", "docs": "/docs"}


def main() -> None:
    """Entry point for `python -m rag_pipeline.api.app`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_pipeline.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
