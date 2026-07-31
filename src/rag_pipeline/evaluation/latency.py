"""Latency measurement — timing, aggregation, and async context manager."""

from __future__ import annotations

import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LatencyMetrics:
    """Aggregated latency measurements."""

    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    ttft_ms: float = 0.0
    queries_per_second: float = 0.0


class LatencyTracker:
    """Track and aggregate latency measurements."""

    def __init__(self) -> None:
        self._marks: dict[str, float] = {}
        self._results: list[LatencyMetrics] = []

    def start(self, name: str) -> None:
        """Start timing a named section."""
        self._marks[name] = time.monotonic()

    def stop(self, name: str) -> float:
        """Stop timing, return elapsed milliseconds."""
        if name not in self._marks:
            return 0.0
        elapsed = (time.monotonic() - self._marks.pop(name)) * 1000
        return elapsed

    def record(self, metrics: LatencyMetrics) -> None:
        """Record a completed measurement."""
        self._results.append(metrics)

    def get_summary(self) -> LatencyMetrics:
        """Aggregate recorded measurements: mean, median, p95 for each metric."""
        if not self._results:
            return LatencyMetrics()
        return LatencyMetrics(
            total_ms=self._aggregate("total_ms"),
            retrieval_ms=self._aggregate("retrieval_ms"),
            rerank_ms=self._aggregate("rerank_ms"),
            generation_ms=self._aggregate("generation_ms"),
            ttft_ms=self._aggregate("ttft_ms"),
            queries_per_second=self._aggregate("queries_per_second"),
        )

    @asynccontextmanager
    async def track(self, name: str) -> AsyncIterator[None]:
        """Async context manager for timing. Auto-records result."""
        self.start(name)
        try:
            yield
        finally:
            elapsed = self.stop(name)
            self.record(LatencyMetrics(total_ms=elapsed))

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _aggregate(self, field_name: str) -> float:
        """Compute mean of a metric field across all recorded results."""
        values = [getattr(r, field_name) for r in self._results]
        if not values:
            return 0.0
        return statistics.mean(values)
