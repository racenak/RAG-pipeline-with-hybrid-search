"""Tests for API middleware — request ID, rate limiting, error handlers."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_pipeline.api.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    register_exception_handlers,
)


def _make_app() -> FastAPI:
    """Build a minimal app with middleware for testing."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)
    register_exception_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"msg": "ok"}

    @app.get("/value-error")
    async def raises_value_error():
        raise ValueError("bad input")

    @app.get("/unhandled")
    async def raises_unhandled():
        raise RuntimeError("something broke")

    return app


@pytest.fixture
def client():
    return TestClient(_make_app(), raise_server_exceptions=False)


# ---- RequestIDMiddleware tests ----


def test_request_id_added_when_missing(client):
    """Middleware generates X-Request-ID when none is sent."""
    r = client.get("/ok")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) == 36  # UUID4 format


def test_request_id_preserved_when_sent(client):
    """Middleware preserves an existing X-Request-ID header."""
    custom_id = "test-id-12345"
    r = client.get("/ok", headers={"X-Request-ID": custom_id})
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == custom_id


def test_request_id_unique_per_request(client):
    """Each request without an ID gets a unique one."""
    r1 = client.get("/ok")
    r2 = client.get("/ok")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


# ---- RateLimitMiddleware tests ----


def test_requests_under_limit_allowed(client):
    """Requests under the limit succeed."""
    for _ in range(3):
        r = client.get("/ok")
        assert r.status_code == 200


def test_requests_over_limit_blocked(client):
    """Requests over the limit return 429."""
    for _ in range(3):
        client.get("/ok")
    r = client.get("/ok")
    assert r.status_code == 429
    data = r.json()
    assert data["error"] == "rate_limit_exceeded"
    assert "Retry-After" in r.headers


def test_rate_limit_returns_safe_message(client):
    """429 response does not leak internal details."""
    for _ in range(4):
        client.get("/ok")
    r = client.get("/ok")
    assert r.status_code == 429
    assert "detail" in r.json()
    assert "Too many requests" in r.json()["detail"]


# ---- Exception handler tests ----


def test_global_exception_handler_returns_500(client):
    """Unhandled exceptions return 500 with safe message."""
    r = client.get("/unhandled")
    assert r.status_code == 500
    data = r.json()
    assert data["error"] == "internal_server_error"
    assert "unexpected" in data["detail"].lower()


def test_value_error_handler_returns_400(client):
    """ValueError returns 400 with the error message."""
    r = client.get("/value-error")
    assert r.status_code == 400
    data = r.json()
    assert data["error"] == "bad_request"
    assert data["detail"] == "bad input"


def test_exception_handler_no_stack_trace_leak(client):
    """Error responses never include Python tracebacks."""
    r = client.get("/unhandled")
    body = r.text
    assert "Traceback" not in body
    assert "File " not in body
