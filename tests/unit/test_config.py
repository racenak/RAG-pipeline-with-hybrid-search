"""Tests for configuration module."""

from rag_pipeline.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings()
    assert settings.app_name == "rag-pipeline"
    assert settings.environment == "development"
    assert settings.storage.opensearch_port == 9200
    assert settings.storage.postgres_database == "rag_pipeline"


def test_postgres_dsn():
    settings = Settings()
    assert "postgresql://" in settings.storage.postgres_dsn
    assert "rag_pipeline" in settings.storage.postgres_dsn


def test_get_settings_returns_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
