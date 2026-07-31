"""Comparison and reporting for retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rag_pipeline.evaluation.retrieval import RetrievalMetrics


@dataclass
class ComparisonResult:
    """Result of comparing a single metric between baseline and current."""

    metric: str
    baseline_value: float
    current_value: float
    change: float  # positive = improvement
    regression: bool  # True if dropped below threshold


class EvalComparator:
    """Compare two RetrievalMetrics and detect regressions."""

    def __init__(self, regression_threshold: float = 0.05) -> None:
        """Initialize comparator.

        Args:
            regression_threshold: Maximum acceptable drop as a fraction (0.05 = 5%).
        """
        self.regression_threshold = regression_threshold

    def compare(
        self, baseline: RetrievalMetrics, current: RetrievalMetrics
    ) -> list[ComparisonResult]:
        """Compare two metric sets, detect regressions.

        Args:
            baseline: Reference metrics (e.g. previous run).
            current: New metrics to compare against baseline.

        Returns:
            List of ComparisonResult for each metric.
        """
        results: list[ComparisonResult] = []

        # Scalar metrics
        results.append(self._compare_scalar("mrr", baseline.mrr, current.mrr))
        results.append(
            self._compare_scalar("hit_rate", baseline.hit_rate, current.hit_rate)
        )
        results.append(
            self._compare_scalar("map_score", baseline.map_score, current.map_score)
        )

        # Dict metrics — find union of keys
        all_p_k = sorted(set(baseline.precision_at_k) | set(current.precision_at_k))
        for k in all_p_k:
            b_val = baseline.precision_at_k.get(k, 0.0)
            c_val = current.precision_at_k.get(k, 0.0)
            results.append(self._compare_scalar(f"precision@{k}", b_val, c_val))

        all_r_k = sorted(set(baseline.recall_at_k) | set(current.recall_at_k))
        for k in all_r_k:
            b_val = baseline.recall_at_k.get(k, 0.0)
            c_val = current.recall_at_k.get(k, 0.0)
            results.append(self._compare_scalar(f"recall@{k}", b_val, c_val))

        all_n_k = sorted(set(baseline.ndcg_at_k) | set(current.ndcg_at_k))
        for k in all_n_k:
            b_val = baseline.ndcg_at_k.get(k, 0.0)
            c_val = current.ndcg_at_k.get(k, 0.0)
            results.append(self._compare_scalar(f"ndcg@{k}", b_val, c_val))

        return results

    def _compare_scalar(
        self, name: str, baseline: float, current: float
    ) -> ComparisonResult:
        """Compare a single scalar metric."""
        change = current - baseline
        # Regression if current dropped more than threshold relative to baseline
        if baseline > 0:
            regression = change < -self.regression_threshold * baseline
        else:
            # If baseline is 0, regression only if current is negative (unlikely)
            regression = current < baseline
        return ComparisonResult(
            metric=name,
            baseline_value=baseline,
            current_value=current,
            change=change,
            regression=regression,
        )

    def to_markdown(self, results: list[ComparisonResult]) -> str:
        """Generate markdown comparison table.

        Args:
            results: List of ComparisonResult to format.

        Returns:
            Markdown-formatted comparison table.
        """
        lines = [
            "| Metric | Baseline | Current | Change | Regression |",
            "|--------|----------|---------|--------|------------|",
        ]
        for r in results:
            change_str = f"{r.change:+.4f}"
            reg_str = "YES" if r.regression else "no"
            lines.append(
                f"| {r.metric} | {r.baseline_value:.4f} | "
                f"{r.current_value:.4f} | {change_str} | {reg_str} |"
            )
        return "\n".join(lines)

    def to_json(self, results: list[ComparisonResult]) -> dict:
        """Generate JSON comparison report.

        Args:
            results: List of ComparisonResult to serialize.

        Returns:
            Dict suitable for JSON serialization.
        """
        return {
            "threshold": self.regression_threshold,
            "total_metrics": len(results),
            "regressions": sum(1 for r in results if r.regression),
            "metrics": [
                {
                    "name": r.metric,
                    "baseline": r.baseline_value,
                    "current": r.current_value,
                    "change": r.change,
                    "regression": r.regression,
                }
                for r in results
            ],
        }
