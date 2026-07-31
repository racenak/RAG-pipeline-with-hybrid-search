"""Observability — structured logging, distributed tracing, metrics collection."""

from rag_pipeline.observability.logging import get_logger, mask_sensitive, setup_logging
from rag_pipeline.observability.metrics import (
    get_metrics,
    init_metrics,
    record_error,
    record_generation,
    record_indexing,
    record_query,
    record_retrieval,
    reset_metrics,
    track_latency,
)
from rag_pipeline.observability.tracing import get_tracer, init_tracing, trace_span

__all__ = [
    "setup_logging",
    "get_logger",
    "mask_sensitive",
    "init_tracing",
    "trace_span",
    "get_tracer",
    "init_metrics",
    "record_query",
    "record_retrieval",
    "record_generation",
    "record_indexing",
    "record_error",
    "get_metrics",
    "reset_metrics",
    "track_latency",
]
