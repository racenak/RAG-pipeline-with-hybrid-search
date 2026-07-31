"""Configuration management — Pydantic BaseSettings + YAML config loading.

Loading priority: Environment variables > Local config > Defaults
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class EmbeddingSettings(BaseSettings):
    backend: str = "sentence-transformers"
    model: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"
    cache_enabled: bool = True
    cache_dir: str = ".cache/embeddings"


class RetrievalSettings(BaseSettings):
    vector_top_k: int = 20
    vector_similarity_threshold: float = 0.7
    bm25_top_k: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10


class GenerationSettings(BaseSettings):
    provider: str = "openrouter"
    model: str = "inclusionai/ling-3.0-flash:free"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = True
    system_prompt: str = ""


class ChunkingSettings(BaseSettings):
    strategy: str = "semantic"
    target_size: int = 512
    max_size: int = 1024
    min_size: int = 100
    overlap: int = 50



class S3Settings(BaseSettings):
    endpoint_url: str = "http://localhost:8333"
    access_key: str = "anything"
    secret_key: str = "anything"
    bucket: str = "rag-documents"


class CleaningSettings(BaseSettings):
    fix_encoding: bool = True
    normalize_unicode: bool = True
    decode_html_entities: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True
    max_blank_lines: int = 1
    clean_pdf_artifacts: bool = True
    strip_residual_html: bool = True


class FirecrawlSettings(BaseSettings):
    api_key: str = ""
    timeout: int = 30
    crawl_limit: int = 100

    model_config = {"env_prefix": "FIRECRAWL_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class StorageSettings(BaseSettings):
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_scheme: str = "http"
    opensearch_index_prefix: str = "rag"
    opensearch_timeout: int = 30
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_max_retries: int = 3
    opensearch_retry_on_timeout: bool = True
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "rag_pipeline"
    postgres_user: str = "rag"
    postgres_password: str = "rag_dev_password"
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )


class ObservabilitySettings(BaseSettings):
    log_level: str = "INFO"
    log_format: str = "json"
    tracing_enabled: bool = True
    tracing_exporter: str = "otlp"
    tracing_endpoint: str = "http://localhost:4317"
    metrics_enabled: bool = True


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 60
    auth_enabled: bool = False


class Settings(BaseSettings):
    """Root settings — composes all sub-settings."""

    app_name: str = "rag-pipeline"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    openrouter_api_key: str = ""

    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    s3: S3Settings = S3Settings()
    cleaning: CleaningSettings = CleaningSettings()
    firecrawl: FirecrawlSettings = FirecrawlSettings()
    storage: StorageSettings = StorageSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    api: APISettings = APISettings()

    model_config = {"env_prefix": "APP_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings(config_path: str | Path | None = None) -> Settings:
    """Return cached Settings instance.

    Loads defaults.yaml, merges with optional local config, then applies
    environment variable overrides.
    """
    # Load .env into os.environ so sub-settings pick up env vars
    from dotenv import load_dotenv
    load_dotenv(CONFIG_DIR.parent / ".env", override=False)

    defaults = _load_yaml(CONFIG_DIR / "defaults.yaml")

    if config_path:
        local = _load_yaml(Path(config_path))
        defaults = _deep_merge(defaults, local)

    return Settings(**defaults)
