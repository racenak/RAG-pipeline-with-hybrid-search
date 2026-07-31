"""Structured logging — JSON format with correlation IDs."""

import json
import logging
import sys
from contextvars import ContextVar

# Correlation ID propagated through context
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        corr_id = correlation_id_var.get()
        if corr_id:
            log_entry["correlation_id"] = corr_id

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", format_type: str = "json") -> None:
    """Configure structured logging for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)


def mask_sensitive(data: dict, keys: list[str] | None = None) -> dict:
    """Mask sensitive fields in a dict (e.g. api_key, password)."""
    if keys is None:
        keys = ["api_key", "password", "secret", "token", "authorization"]

    masked = {}
    for k, v in data.items():
        if any(sensitive in k.lower() for sensitive in keys):
            masked[k] = "***MASKED***" if isinstance(v, str) else v
        elif isinstance(v, dict):
            masked[k] = mask_sensitive(v, keys)
        else:
            masked[k] = v
    return masked
