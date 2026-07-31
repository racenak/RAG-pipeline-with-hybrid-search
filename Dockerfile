# =============================================================================
# Multi-stage Containerfile for RAG Pipeline API
# =============================================================================

# ---- Base stage -------------------------------------------------------------
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for psycopg2, lxml, faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies stage -----------------------------------------------------
FROM base AS deps

# Copy only dependency files (not source, not tool configs)
COPY pyproject.toml uv.lock ./

# Install uv, then install ONLY production deps (no dev)
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project \
    && rm -rf /root/.cache/uv

# ---- Runtime stage ----------------------------------------------------------
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Non-root user
RUN groupadd -r rag && useradd -r -g rag -d /app -s /sbin/nologin rag \
    && chown -R rag:rag /app
USER rag

# Copy only what the API needs
COPY src/ src/
COPY config/defaults.yaml config/defaults.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "rag_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
