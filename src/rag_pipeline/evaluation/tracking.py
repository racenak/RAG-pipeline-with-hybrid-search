"""Experiment tracking — config, metrics, storage, comparison, and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

# ------------------------------------------------------------------ #
#  Enums
# ------------------------------------------------------------------ #


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    description: str = ""
    model: str = ""
    embedding_model: str = ""
    retrieval_mode: str = "hybrid"
    top_k: int = 10
    rrf_k: int = 60
    rerank_enabled: bool = True
    temperature: float = 0.7
    max_tokens: int = 1024
    custom: dict = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    """Collected metrics for an experiment."""

    retrieval: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)


@dataclass
class Experiment:
    """A single experiment run."""

    id: str
    config: ExperimentConfig
    status: ExperimentStatus
    metrics: ExperimentMetrics
    created_at: str
    completed_at: str | None = None
    duration_ms: float = 0.0
    error: str | None = None
    tags: list[str] = field(default_factory=list)
    dataset_info: dict = field(default_factory=dict)


@dataclass
class MetricComparison:
    """Comparison of a single metric between baseline and current."""

    metric_name: str
    baseline_value: float
    current_value: float
    change_pct: float
    improved: bool


@dataclass
class ExperimentComparison:
    """Side-by-side comparison of two experiments."""

    baseline: Experiment
    current: Experiment
    comparisons: list[MetricComparison]
    regressions: list[MetricComparison]
    summary: str


# ------------------------------------------------------------------ #
#  Serialization helpers
# ------------------------------------------------------------------ #


def _experiment_to_dict(exp: Experiment) -> dict:
    """Serialize an Experiment to a JSON-safe dict."""
    return {
        "id": exp.id,
        "config": asdict(exp.config),
        "status": exp.status.value,
        "metrics": asdict(exp.metrics),
        "created_at": exp.created_at,
        "completed_at": exp.completed_at,
        "duration_ms": exp.duration_ms,
        "error": exp.error,
        "tags": exp.tags,
        "dataset_info": exp.dataset_info,
    }


def _experiment_from_dict(data: dict) -> Experiment:
    """Deserialize a dict back into an Experiment."""
    return Experiment(
        id=data["id"],
        config=ExperimentConfig(**data["config"]),
        status=ExperimentStatus(data["status"]),
        metrics=ExperimentMetrics(**data["metrics"]),
        created_at=data["created_at"],
        completed_at=data.get("completed_at"),
        duration_ms=data.get("duration_ms", 0.0),
        error=data.get("error"),
        tags=data.get("tags", []),
        dataset_info=data.get("dataset_info", {}),
    )


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------ #
#  ExperimentTracker
# ------------------------------------------------------------------ #


class ExperimentTracker:
    """Create, update, query, and compare experiment runs persisted as JSON files."""

    def __init__(self, storage_dir: str | Path = "experiments") -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ---- CRUD -------------------------------------------------------- #

    def start_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create and save a new experiment with status=RUNNING."""
        exp = Experiment(
            id=str(uuid4()),
            config=config,
            status=ExperimentStatus.RUNNING,
            metrics=ExperimentMetrics(),
            created_at=_now_iso(),
        )
        self._save(exp)
        return exp

    def complete_experiment(self, experiment_id: str, metrics: ExperimentMetrics) -> Experiment:
        """Mark experiment as COMPLETED with metrics."""
        exp = self.get_experiment(experiment_id)
        exp.status = ExperimentStatus.COMPLETED
        exp.metrics = metrics
        exp.completed_at = _now_iso()
        if exp.created_at:
            created = datetime.fromisoformat(exp.created_at)
            completed = datetime.fromisoformat(exp.completed_at)
            exp.duration_ms = (completed - created).total_seconds() * 1000
        self._save(exp)
        return exp

    def fail_experiment(self, experiment_id: str, error: str) -> Experiment:
        """Mark experiment as FAILED with error message."""
        exp = self.get_experiment(experiment_id)
        exp.status = ExperimentStatus.FAILED
        exp.error = error
        exp.completed_at = _now_iso()
        self._save(exp)
        return exp

    def log_metrics(self, experiment_id: str, metrics: ExperimentMetrics) -> Experiment:
        """Update metrics for a running experiment."""
        exp = self.get_experiment(experiment_id)
        exp.metrics = metrics
        self._save(exp)
        return exp

    def tag_experiment(self, experiment_id: str, tags: list[str]) -> Experiment:
        """Set tags on an experiment."""
        exp = self.get_experiment(experiment_id)
        exp.tags = tags
        self._save(exp)
        return exp

    def set_dataset_info(self, experiment_id: str, dataset_info: dict) -> Experiment:
        """Set dataset info on an experiment."""
        exp = self.get_experiment(experiment_id)
        exp.dataset_info = dataset_info
        self._save(exp)
        return exp

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Load experiment by ID."""
        path = self._file_path(experiment_id)
        if not path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        with path.open() as f:
            data = json.load(f)
        return _experiment_from_dict(data)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        tag: str | None = None,
    ) -> list[Experiment]:
        """List all experiments, optionally filtered by status or tag."""
        results: list[Experiment] = []
        for p in sorted(self._dir.glob("*.json")):
            with p.open() as f:
                data = json.load(f)
            exp = _experiment_from_dict(data)
            if status is not None and exp.status != status:
                continue
            if tag is not None and tag not in exp.tags:
                continue
            results.append(exp)
        return results

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment file. Returns True if deleted."""
        path = self._file_path(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # ---- Comparison -------------------------------------------------- #

    def compare(self, experiment_id_1: str, experiment_id_2: str) -> ExperimentComparison:
        """Compare two experiments side by side.

        ``experiment_id_1`` is treated as the baseline, ``experiment_id_2``
        as the current run.
        """
        baseline = self.get_experiment(experiment_id_1)
        current = self.get_experiment(experiment_id_2)
        return _compare_experiments(baseline, current)

    def get_baseline(self, experiment_name: str) -> Experiment | None:
        """Get the latest completed experiment with the given name."""
        candidates = [
            e
            for e in self.list_experiments(status=ExperimentStatus.COMPLETED)
            if e.config.name == experiment_name
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda e: e.created_at, reverse=True)
        return candidates[0]

    # ---- Internal ---------------------------------------------------- #

    def _file_path(self, experiment_id: str) -> Path:
        return self._dir / f"{experiment_id}.json"

    def _save(self, exp: Experiment) -> None:
        path = self._file_path(exp.id)
        with path.open("w") as f:
            json.dump(_experiment_to_dict(exp), f, indent=2)


# ------------------------------------------------------------------ #
#  Comparison logic
# ------------------------------------------------------------------ #


def _flatten_metrics(metrics: ExperimentMetrics) -> dict[str, float]:
    """Flatten all metric categories into a single dict with dot-notation keys."""
    flat: dict[str, float] = {}
    for category_name in ("retrieval", "generation", "latency", "cost"):
        category = getattr(metrics, category_name)
        if not isinstance(category, dict):
            continue
        for key, value in category.items():
            if isinstance(value, (int, float)):
                flat[f"{category_name}.{key}"] = float(value)
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        flat[f"{category_name}.{key}.{sub_key}"] = float(sub_val)
    return flat


def _compare_experiments(baseline: Experiment, current: Experiment) -> ExperimentComparison:
    """Build an ExperimentComparison from two experiments."""
    b_flat = _flatten_metrics(baseline.metrics)
    c_flat = _flatten_metrics(current.metrics)
    all_keys = sorted(set(b_flat) | set(c_flat))

    comparisons: list[MetricComparison] = []
    for key in all_keys:
        b_val = b_flat.get(key, 0.0)
        c_val = c_flat.get(key, 0.0)
        if b_val == 0.0 and c_val == 0.0:
            continue
        if b_val != 0.0:
            change_pct = ((c_val - b_val) / abs(b_val)) * 100
        else:
            change_pct = 100.0 if c_val > 0 else 0.0
        improved = c_val > b_val
        comparisons.append(
            MetricComparison(
                metric_name=key,
                baseline_value=b_val,
                current_value=c_val,
                change_pct=round(change_pct, 2),
                improved=improved,
            )
        )

    regressions = [c for c in comparisons if not c.improved]
    improvements = [c for c in comparisons if c.improved]

    lines = [
        f"Compared {baseline.config.name} ({baseline.id[:8]}) vs "
        f"{current.config.name} ({current.id[:8]}):",
        f"  {len(improvements)} improved, {len(regressions)} regressed "
        f"(out of {len(comparisons)} metrics).",
    ]
    if regressions:
        lines.append("  Regressions:")
        for r in regressions:
            lines.append(
                f"    - {r.metric_name}: {r.baseline_value:.4f} -> "
                f"{r.current_value:.4f} ({r.change_pct:+.1f}%)"
            )

    return ExperimentComparison(
        baseline=baseline,
        current=current,
        comparisons=comparisons,
        regressions=regressions,
        summary="\n".join(lines),
    )


# ------------------------------------------------------------------ #
#  ExperimentReporter
# ------------------------------------------------------------------ #


class ExperimentReporter:
    """Generate JSON and markdown reports for experiments."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self._tracker = tracker

    def to_json(self, experiment: Experiment) -> dict:
        """Serialize experiment to a JSON-safe dict."""
        return _experiment_to_dict(experiment)

    def to_markdown(self, experiment: Experiment) -> str:
        """Generate a markdown report for a single experiment."""
        lines = [
            f"# Experiment: {experiment.config.name}",
            "",
            f"- **ID**: `{experiment.id}`",
            f"- **Status**: {experiment.status.value}",
            f"- **Created**: {experiment.created_at}",
        ]
        if experiment.completed_at:
            lines.append(f"- **Completed**: {experiment.completed_at}")
        if experiment.duration_ms:
            lines.append(f"- **Duration**: {experiment.duration_ms:.0f}ms")
        if experiment.error:
            lines.append(f"- **Error**: {experiment.error}")
        if experiment.tags:
            lines.append(f"- **Tags**: {', '.join(experiment.tags)}")

        lines += [
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Model | {experiment.config.model} |",
            f"| Embedding Model | {experiment.config.embedding_model} |",
            f"| Retrieval Mode | {experiment.config.retrieval_mode} |",
            f"| Top-K | {experiment.config.top_k} |",
            f"| RRF K | {experiment.config.rrf_k} |",
            f"| Rerank Enabled | {experiment.config.rerank_enabled} |",
            f"| Temperature | {experiment.config.temperature} |",
            f"| Max Tokens | {experiment.config.max_tokens} |",
        ]

        lines += ["", "## Metrics", ""]
        for category in ("retrieval", "generation", "latency", "cost"):
            data = getattr(experiment.metrics, category)
            if data:
                lines.append(f"### {category.capitalize()}")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for key, value in sorted(data.items()):
                    lines.append(f"| {key} | {value} |")
                lines.append("")

        if experiment.dataset_info:
            lines += ["## Dataset", ""]
            for key, value in sorted(experiment.dataset_info.items()):
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        return "\n".join(lines)

    def comparison_to_markdown(self, comparison: ExperimentComparison) -> str:
        """Generate a markdown comparison report."""
        lines = [
            "# Experiment Comparison",
            "",
            f"- **Baseline**: {comparison.baseline.config.name} (`{comparison.baseline.id[:8]}`)",
            f"- **Current**: {comparison.current.config.name} (`{comparison.current.id[:8]}`)",
            "",
            "## Metrics",
            "",
            "| Metric | Baseline | Current | Change | Status |",
            "|--------|----------|---------|--------|--------|",
        ]

        for c in comparison.comparisons:
            status = "improved" if c.improved else "REGRESSED"
            lines.append(
                f"| {c.metric_name} | {c.baseline_value:.4f} | "
                f"{c.current_value:.4f} | {c.change_pct:+.1f}% | {status} |"
            )

        lines += [
            "",
            "## Summary",
            "",
            comparison.summary,
            "",
        ]

        return "\n".join(lines)

    def trend_report(self, experiments: list[Experiment]) -> str:
        """Generate text-based trend visualization showing metric changes."""
        if not experiments:
            return "No experiments to display."

        sorted_exps = sorted(experiments, key=lambda e: e.created_at)

        lines = [
            "# Experiment Trend Report",
            "",
            f"Total experiments: {len(sorted_exps)}",
            "",
        ]

        # Collect all metric keys across experiments
        all_keys: set[str] = set()
        flat_metrics_list: list[dict[str, float]] = []
        for exp in sorted_exps:
            flat = _flatten_metrics(exp.metrics)
            flat_metrics_list.append(flat)
            all_keys.update(flat.keys())

        if not all_keys:
            lines.append("No metrics recorded across experiments.")
            return "\n".join(lines)

        # Pick top metrics by variance (most interesting to show)
        key_variance: list[tuple[str, float]] = []
        for key in sorted(all_keys):
            vals = [fm.get(key, 0.0) for fm in flat_metrics_list]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
            else:
                var = 0.0
            key_variance.append((key, var))
        key_variance.sort(key=lambda x: x[1], reverse=True)
        display_keys = [k for k, _ in key_variance[:10]]

        for key in display_keys:
            values = [fm.get(key, 0.0) for fm in flat_metrics_list]
            lines.append(f"## {key}")
            lines.append("")
            bar_width = 30
            min_val = min(values) if values else 0.0
            max_val = max(values) if values else 1.0
            val_range = max_val - min_val if max_val != min_val else 1.0

            for exp, val in zip(sorted_exps, values, strict=True):
                bar_len = int(((val - min_val) / val_range) * bar_width)
                bar = "#" * max(bar_len, 1)
                label = exp.config.name[:20].ljust(20)
                lines.append(f"  {label} | {bar} {val:.4f}")
            lines.append("")

        return "\n".join(lines)
