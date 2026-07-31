"""API authentication — API key validation."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from rag_pipeline.config import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> set[str]:
    """Load API keys from environment variable (comma-separated)."""
    keys_str = os.environ.get("API_KEYS", "")
    if keys_str:
        return {k.strip() for k in keys_str.split(",") if k.strip()}
    return set()


async def validate_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str | None:
    """Validate API key. Returns key if valid, None if auth disabled."""
    settings = get_settings()

    if not settings.api.auth_enabled:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "detail": "API key required"},
        )

    valid_keys = _load_api_keys()
    if not valid_keys:
        # No keys configured — allow in dev mode only
        if settings.environment == "development":
            return api_key
        raise HTTPException(
            status_code=500,
            detail={"error": "server_error", "detail": "No API keys configured"},
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden", "detail": "Invalid API key"},
        )

    return api_key


def require_api_key() -> Security:
    """Dependency that requires a valid API key."""
    return Security(validate_api_key)
