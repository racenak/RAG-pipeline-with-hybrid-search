"""OpenTelemetry tracing — span creation and export."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Global tracer (initialized lazily)
_tracer = None


def init_tracing(service_name: str = "rag-pipeline", endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter."""
    global _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint)
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        logger.info("Tracing initialized", extra={"extra_data": {"endpoint": endpoint}})
    except ImportError:
        logger.warning("opentelemetry packages not installed, tracing disabled")
        _tracer = _StubTracer()


def get_tracer():
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = _StubTracer()
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for creating a trace span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


class _StubTracer:
    """No-op tracer when OpenTelemetry is not available."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        yield _StubSpan()


class _StubSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args, **kwargs) -> None:
        pass

    def end(self) -> None:
        pass
