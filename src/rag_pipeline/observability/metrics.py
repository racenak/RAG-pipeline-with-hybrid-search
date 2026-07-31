"""Prometheus metrics — counters, histograms, gauges for RAG pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

# Metrics registry (initialized lazily)
_metrics_enabled = False
_counter_registry: dict[str, int] = {}
_histogram_registry: dict[str, list[float]] = {}
_gauge_registry: dict[str, float] = {}


def init_metrics(enabled: bool = True) -> None:
    """Initialize Prometheus metrics collection."""
    global _metrics_enabled
    _metrics_enabled = enabled
    if enabled:
        logger.info("Metrics collection enabled")


def increment_counter(name: str, value: float = 1.0, labels: dict | None = None) -> None:
    """Increment a counter metric."""
    if not _metrics_enabled:
        return
    key = _make_key(name, labels)
    _counter_registry[key] = _counter_registry.get(key, 0) + value


def observe_histogram(name: str, value: float, labels: dict | None = None) -> None:
    """Record a histogram observation."""
    if not _metrics_enabled:
        return
    key = _make_key(name, labels)
    if key not in _histogram_registry:
        _histogram_registry[key] = []
    _histogram_registry[key].append(value)


def set_gauge(name: str, value: float, labels: dict | None = None) -> None:
    """Set a gauge value."""
    if not _metrics_enabled:
        return
    key = _make_key(name, labels)
    _gauge_registry[key] = value


def get_metrics() -> dict:
    """Return all collected metrics."""
    return {
        "counters": dict(_counter_registry),
        "histograms": {
            k: {
                "count": len(v),
                "sum": sum(v),
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
                "avg": sum(v) / len(v) if v else 0,
            }
            for k, v in _histogram_registry.items()
        },
        "gauges": dict(_gauge_registry),
    }


def reset_metrics() -> None:
    """Reset all metrics (for testing)."""
    _counter_registry.clear()
    _histogram_registry.clear()
    _gauge_registry.clear()


@contextmanager
def track_latency(name: str, labels: dict | None = None) -> Generator[None, None, None]:
    """Context manager to track operation latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        observe_histogram(f"{name}_latency_ms", elapsed_ms, labels)


def _make_key(name: str, labels: dict | None) -> str:
    """Create a metric key from name and labels."""
    if not labels:
        return name
    label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


# Pre-defined RAG metrics (convenience functions)


def record_query(mode: str, latency_ms: float) -> None:
    """Record a RAG query metric."""
    increment_counter("rag_query_total", labels={"mode": mode})
    observe_histogram("rag_query_latency_ms", latency_ms, labels={"mode": mode})


def record_retrieval(mode: str, latency_ms: float, result_count: int) -> None:
    """Record retrieval metrics."""
    observe_histogram("rag_retrieval_latency_ms", latency_ms, labels={"mode": mode})
    set_gauge("rag_retrieval_results", result_count, labels={"mode": mode})


def record_generation(model: str, latency_ms: float, tokens: int) -> None:
    """Record generation metrics."""
    observe_histogram("rag_generation_latency_ms", latency_ms, labels={"model": model})
    increment_counter("rag_generation_tokens_total", tokens, labels={"model": model})


def record_indexing(operation: str, count: int) -> None:
    """Record indexing metrics."""
    increment_counter("rag_documents_indexed_total", count, labels={"operation": operation})


def record_error(error_type: str) -> None:
    """Record an error."""
    increment_counter("rag_errors_total", labels={"error_type": error_type})
