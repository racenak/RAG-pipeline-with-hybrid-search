"""Tests for observability modules — logging, tracing, metrics."""

import json
import logging
import time
from unittest.mock import patch

from rag_pipeline.observability.logging import (
    JSONFormatter,
    correlation_id_var,
    get_logger,
    mask_sensitive,
    setup_logging,
)
from rag_pipeline.observability.metrics import (
    _counter_registry,
    _gauge_registry,
    _histogram_registry,
    get_metrics,
    increment_counter,
    init_metrics,
    observe_histogram,
    record_error,
    record_generation,
    record_indexing,
    record_query,
    record_retrieval,
    reset_metrics,
    set_gauge,
    track_latency,
)
from rag_pipeline.observability.tracing import (
    _StubSpan,
    _StubTracer,
    get_tracer,
    trace_span,
)


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


def test_json_formatter_produces_valid_json():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello"
    assert parsed["logger"] == "test"


def test_json_formatter_includes_correlation_id():
    formatter = JSONFormatter()
    token = correlation_id_var.set("abc-123")
    try:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["correlation_id"] == "abc-123"
    finally:
        correlation_id_var.reset(token)


def test_json_formatter_no_correlation_id():
    formatter = JSONFormatter()
    correlation_id_var.set(None)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=1,
        msg="msg", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "correlation_id" not in parsed


def test_json_formatter_exception_info():
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=1,
        msg="error happened", args=(), exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_json_formatter_extra_data():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=1,
        msg="msg", args=(), exc_info=None,
    )
    record.extra_data = {"key": "val"}
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["extra"] == {"key": "val"}


def test_setup_logging_configures_root_logger():
    setup_logging("DEBUG", "json")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_text_format():
    setup_logging("WARNING", "text")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert not isinstance(root.handlers[0].formatter, JSONFormatter)


def test_get_logger_returns_named_logger():
    logger = get_logger("my.module")
    assert logger.name == "my.module"


def test_mask_sensitive_masks_api_keys():
    data = {"api_key": "sk-1234", "password": "secret123", "host": "localhost"}
    masked = mask_sensitive(data)
    assert masked["api_key"] == "***MASKED***"
    assert masked["password"] == "***MASKED***"
    assert masked["host"] == "localhost"


def test_mask_sensitive_preserves_non_sensitive():
    data = {"name": "test", "count": 42, "nested": {"token": "tok", "port": 8080}}
    masked = mask_sensitive(data)
    assert masked["name"] == "test"
    assert masked["count"] == 42
    assert masked["nested"]["token"] == "***MASKED***"
    assert masked["nested"]["port"] == 8080


def test_mask_sensitive_custom_keys():
    data = {"my_secret": "val", "other": "keep"}
    masked = mask_sensitive(data, keys=["secret"])
    assert masked["my_secret"] == "***MASKED***"
    assert masked["other"] == "keep"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_init_metrics_sets_enabled_flag():
    reset_metrics()
    init_metrics(True)
    increment_counter("test_counter")
    metrics = get_metrics()
    assert metrics["counters"]["test_counter"] == 1.0
    reset_metrics()


def test_increment_counter_accumulates():
    reset_metrics()
    init_metrics(True)
    increment_counter("req")
    increment_counter("req")
    increment_counter("req", value=3)
    metrics = get_metrics()
    assert metrics["counters"]["req"] == 5.0
    reset_metrics()


def test_observe_histogram_stores_observations():
    reset_metrics()
    init_metrics(True)
    observe_histogram("latency", 10.0)
    observe_histogram("latency", 20.0)
    observe_histogram("latency", 30.0)
    metrics = get_metrics()
    h = metrics["histograms"]["latency"]
    assert h["count"] == 3
    assert h["sum"] == 60.0
    assert h["min"] == 10.0
    assert h["max"] == 30.0
    assert h["avg"] == 20.0
    reset_metrics()


def test_set_gauge_stores_value():
    reset_metrics()
    init_metrics(True)
    set_gauge("queue_depth", 42)
    set_gauge("queue_depth", 10)
    metrics = get_metrics()
    assert metrics["gauges"]["queue_depth"] == 10.0
    reset_metrics()


def test_get_metrics_returns_all_types():
    reset_metrics()
    init_metrics(True)
    increment_counter("c")
    observe_histogram("h", 1.0)
    set_gauge("g", 5.0)
    metrics = get_metrics()
    assert "counters" in metrics
    assert "histograms" in metrics
    assert "gauges" in metrics
    reset_metrics()


def test_reset_metrics_clears_everything():
    reset_metrics()
    init_metrics(True)
    increment_counter("c")
    observe_histogram("h", 1.0)
    set_gauge("g", 5.0)
    reset_metrics()
    metrics = get_metrics()
    assert metrics["counters"] == {}
    assert metrics["histograms"] == {}
    assert metrics["gauges"] == {}


def test_metrics_disabled_does_nothing():
    reset_metrics()
    init_metrics(False)
    increment_counter("c")
    observe_histogram("h", 1.0)
    set_gauge("g", 5.0)
    metrics = get_metrics()
    assert metrics["counters"] == {}
    assert metrics["histograms"] == {}
    assert metrics["gauges"] == {}
    reset_metrics()


def test_track_latency_records_elapsed():
    reset_metrics()
    init_metrics(True)
    with track_latency("op"):
        time.sleep(0.01)
    metrics = get_metrics()
    assert "op_latency_ms" in metrics["histograms"]
    assert metrics["histograms"]["op_latency_ms"]["count"] == 1
    assert metrics["histograms"]["op_latency_ms"]["min"] > 0
    reset_metrics()


def test_record_query_convenience():
    reset_metrics()
    init_metrics(True)
    record_query("vector", 55.0)
    metrics = get_metrics()
    assert "rag_query_total{mode=vector}" in metrics["counters"]
    assert "rag_query_latency_ms{mode=vector}" in metrics["histograms"]
    reset_metrics()


def test_record_retrieval_convenience():
    reset_metrics()
    init_metrics(True)
    record_retrieval("hybrid", 30.0, 5)
    metrics = get_metrics()
    assert "rag_retrieval_latency_ms{mode=hybrid}" in metrics["histograms"]
    assert metrics["gauges"]["rag_retrieval_results{mode=hybrid}"] == 5.0
    reset_metrics()


def test_record_generation_convenience():
    reset_metrics()
    init_metrics(True)
    record_generation("gpt-4", 200.0, 512)
    metrics = get_metrics()
    assert "rag_generation_latency_ms{model=gpt-4}" in metrics["histograms"]
    assert metrics["counters"]["rag_generation_tokens_total{model=gpt-4}"] == 512.0
    reset_metrics()


def test_record_indexing_convenience():
    reset_metrics()
    init_metrics(True)
    record_indexing("bulk", 100)
    metrics = get_metrics()
    assert metrics["counters"]["rag_documents_indexed_total{operation=bulk}"] == 100.0
    reset_metrics()


def test_record_error_convenience():
    reset_metrics()
    init_metrics(True)
    record_error("timeout")
    metrics = get_metrics()
    assert metrics["counters"]["rag_errors_total{error_type=timeout}"] == 1.0
    reset_metrics()


def test_make_key_with_labels():
    from rag_pipeline.observability.metrics import _make_key

    result = _make_key("name", {"b": "2", "a": "1"})
    assert result == "name{a=1,b=2}"


def test_make_key_without_labels():
    from rag_pipeline.observability.metrics import _make_key

    result = _make_key("name", None)
    assert result == "name"


# ---------------------------------------------------------------------------
# Tracing tests
# ---------------------------------------------------------------------------


def test_trace_span_with_stub():
    with trace_span("test-span") as span:
        assert isinstance(span, _StubSpan)


def test_trace_span_sets_attributes():
    with trace_span("test-span", attributes={"key": "value"}) as span:
        assert isinstance(span, _StubSpan)
        # Should not raise
        span.set_attribute("key", "value")


def test_stub_tracer_no_op():
    tracer = _StubTracer()
    with tracer.start_as_current_span("span") as span:
        assert isinstance(span, _StubSpan)
        span.set_attribute("k", "v")
        span.set_status()
        span.end()


def test_stub_span_no_op():
    span = _StubSpan()
    span.set_attribute("k", "v")
    span.set_status()
    span.end()


def test_get_tracer_returns_stub_by_default():
    tracer = get_tracer()
    assert isinstance(tracer, _StubTracer)
