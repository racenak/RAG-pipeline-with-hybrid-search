"""Tests for experiment tracking — data models, tracker, reporter, comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_pipeline.evaluation.tracking import (
    Experiment,
    ExperimentComparison,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentReporter,
    ExperimentStatus,
    ExperimentTracker,
    _experiment_from_dict,
    _experiment_to_dict,
    _flatten_metrics,
)

# ------------------------------------------------------------------ #
#  Helper
# ------------------------------------------------------------------ #


def _make_config(name: str = "test-exp", **overrides) -> ExperimentConfig:
    return ExperimentConfig(name=name, **overrides)


def _make_metrics(retrieval: dict | None = None, **overrides) -> ExperimentMetrics:
    return ExperimentMetrics(
        retrieval=retrieval or {"mrr": 0.8, "hit_rate": 0.9},
        **overrides,
    )


def _start_and_complete(
    tracker: ExperimentTracker,
    name: str = "test-exp",
    retrieval: dict | None = None,
    tags: list[str] | None = None,
) -> Experiment:
    config = _make_config(name=name)
    exp = tracker.start_experiment(config)
    if tags:
        exp.tags = tags
        tracker._save(exp)
    metrics = _make_metrics(retrieval=retrieval)
    return tracker.complete_experiment(exp.id, metrics)


# ------------------------------------------------------------------ #
#  Dataclass creation
# ------------------------------------------------------------------ #


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig(name="x")
        assert cfg.name == "x"
        assert cfg.retrieval_mode == "hybrid"
        assert cfg.top_k == 10
        assert cfg.rrf_k == 60
        assert cfg.rerank_enabled is True
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024
        assert cfg.custom == {}

    def test_custom_values(self):
        cfg = ExperimentConfig(
            name="y",
            retrieval_mode="bm25",
            top_k=5,
            custom={"key": "val"},
        )
        assert cfg.retrieval_mode == "bm25"
        assert cfg.top_k == 5
        assert cfg.custom == {"key": "val"}


class TestExperimentMetrics:
    def test_defaults(self):
        m = ExperimentMetrics()
        assert m.retrieval == {}
        assert m.generation == {}
        assert m.latency == {}
        assert m.cost == {}

    def test_with_values(self):
        m = ExperimentMetrics(retrieval={"mrr": 0.5}, cost={"api_calls": 10})
        assert m.retrieval["mrr"] == 0.5
        assert m.cost["api_calls"] == 10


class TestExperiment:
    def test_creation(self):
        exp = Experiment(
            id="abc-123",
            config=_make_config(),
            status=ExperimentStatus.RUNNING,
            metrics=ExperimentMetrics(),
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert exp.id == "abc-123"
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.completed_at is None
        assert exp.error is None
        assert exp.tags == []


# ------------------------------------------------------------------ #
#  Serialization roundtrip
# ------------------------------------------------------------------ #


class TestSerialization:
    def test_roundtrip(self):
        exp = Experiment(
            id="test-id",
            config=_make_config(name="rt-test", custom={"a": 1}),
            status=ExperimentStatus.COMPLETED,
            metrics=_make_metrics(),
            created_at="2025-01-01T00:00:00+00:00",
            completed_at="2025-01-01T00:01:00+00:00",
            duration_ms=60000.0,
            tags=["tag1", "tag2"],
            dataset_info={"total_queries": 42},
        )
        data = _experiment_to_dict(exp)
        restored = _experiment_from_dict(data)
        assert restored.id == exp.id
        assert restored.config.name == "rt-test"
        assert restored.config.custom == {"a": 1}
        assert restored.status == ExperimentStatus.COMPLETED
        assert restored.metrics.retrieval == {"mrr": 0.8, "hit_rate": 0.9}
        assert restored.tags == ["tag1", "tag2"]
        assert restored.dataset_info == {"total_queries": 42}

    def test_to_json_is_serializable(self):
        exp = Experiment(
            id="x",
            config=_make_config(),
            status=ExperimentStatus.RUNNING,
            metrics=ExperimentMetrics(),
            created_at="2025-01-01T00:00:00+00:00",
        )
        data = _experiment_to_dict(exp)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


# ------------------------------------------------------------------ #
#  ExperimentTracker
# ------------------------------------------------------------------ #


class TestExperimentTracker:
    def test_auto_creates_directory(self, tmp_path: Path):
        storage = tmp_path / "new_dir" / "sub"
        ExperimentTracker(storage_dir=storage)
        assert storage.exists()

    def test_start_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        assert exp.status == ExperimentStatus.RUNNING
        assert (tmp_path / f"{exp.id}.json").exists()

    def test_experiment_ids_are_unique(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        ids = {tracker.start_experiment(_make_config()).id for _ in range(10)}
        assert len(ids) == 10

    def test_complete_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        metrics = _make_metrics(retrieval={"mrr": 0.9})
        completed = tracker.complete_experiment(exp.id, metrics)
        assert completed.status == ExperimentStatus.COMPLETED
        assert completed.metrics.retrieval["mrr"] == 0.9
        assert completed.completed_at is not None
        assert completed.duration_ms >= 0

    def test_fail_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        failed = tracker.fail_experiment(exp.id, "something broke")
        assert failed.status == ExperimentStatus.FAILED
        assert failed.error == "something broke"
        assert failed.completed_at is not None

    def test_log_metrics(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        updated = tracker.log_metrics(exp.id, ExperimentMetrics(retrieval={"mrr": 0.5}))
        assert updated.metrics.retrieval["mrr"] == 0.5

    def test_tag_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        updated = tracker.tag_experiment(exp.id, ["prod", "v2"])
        assert updated.tags == ["prod", "v2"]
        loaded = tracker.get_experiment(exp.id)
        assert loaded.tags == ["prod", "v2"]

    def test_set_dataset_info(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        info = {"total_queries": 42, "source": "golden_dataset.json"}
        updated = tracker.set_dataset_info(exp.id, info)
        assert updated.dataset_info == info
        loaded = tracker.get_experiment(exp.id)
        assert loaded.dataset_info == info

    def test_get_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config(name="findme"))
        loaded = tracker.get_experiment(exp.id)
        assert loaded.config.name == "findme"

    def test_get_experiment_not_found(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.get_experiment("nonexistent-id")

    def test_list_experiments_all(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_experiment(_make_config("a"))
        tracker.start_experiment(_make_config("b"))
        assert len(tracker.list_experiments()) == 2

    def test_list_experiments_filter_status(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        tracker.start_experiment(_make_config("a"))
        _start_and_complete(tracker, "b")
        running = tracker.list_experiments(status=ExperimentStatus.RUNNING)
        completed = tracker.list_experiments(status=ExperimentStatus.COMPLETED)
        assert len(running) == 1
        assert len(completed) == 1

    def test_list_experiments_filter_tag(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        _start_and_complete(tracker, "a", tags=["prod"])
        _start_and_complete(tracker, "b", tags=["dev"])
        prod = tracker.list_experiments(tag="prod")
        assert len(prod) == 1
        assert prod[0].config.name == "a"

    def test_delete_experiment(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        exp = tracker.start_experiment(_make_config())
        exp_id = exp.id
        assert tracker.delete_experiment(exp_id) is True
        assert not (tmp_path / f"{exp_id}.json").exists()
        assert tracker.delete_experiment(exp_id) is False

    def test_compare_detects_improvement(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        e1 = _start_and_complete(tracker, "base", retrieval={"mrr": 0.5})
        e2 = _start_and_complete(tracker, "better", retrieval={"mrr": 0.8})
        comp = tracker.compare(e1.id, e2.id)
        assert isinstance(comp, ExperimentComparison)
        mrr_comp = next(c for c in comp.comparisons if "mrr" in c.metric_name)
        assert mrr_comp.improved is True
        assert mrr_comp.change_pct > 0

    def test_compare_detects_regression(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        e1 = _start_and_complete(tracker, "base", retrieval={"mrr": 0.9})
        e2 = _start_and_complete(tracker, "worse", retrieval={"mrr": 0.5})
        comp = tracker.compare(e1.id, e2.id)
        assert len(comp.regressions) > 0
        mrr_comp = next(c for c in comp.comparisons if "mrr" in c.metric_name)
        assert mrr_comp.improved is False

    def test_get_baseline(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        _start_and_complete(tracker, "myexp")
        tracker.start_experiment(_make_config("myexp"))  # running, should not match
        _start_and_complete(tracker, "other")
        baseline = tracker.get_baseline("myexp")
        assert baseline is not None
        assert baseline.config.name == "myexp"
        assert baseline.status == ExperimentStatus.COMPLETED

    def test_get_baseline_not_found(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        assert tracker.get_baseline("nonexistent") is None


# ------------------------------------------------------------------ #
#  _flatten_metrics
# ------------------------------------------------------------------ #


class TestFlattenMetrics:
    def test_flattens_nested(self):
        m = ExperimentMetrics(
            retrieval={"mrr": 0.8, "precision": {"1": 0.9}},
            latency={"total_ms": 100.0},
        )
        flat = _flatten_metrics(m)
        assert flat["retrieval.mrr"] == 0.8
        assert flat["retrieval.precision.1"] == 0.9
        assert flat["latency.total_ms"] == 100.0

    def test_ignores_non_numeric(self):
        m = ExperimentMetrics(retrieval={"label": "good"})
        flat = _flatten_metrics(m)
        assert flat == {}


# ------------------------------------------------------------------ #
#  ExperimentReporter
# ------------------------------------------------------------------ #


class TestExperimentReporter:
    def _setup(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        reporter = ExperimentReporter(tracker)
        exp = _start_and_complete(tracker, "report-test")
        return tracker, reporter, exp

    def test_to_json(self, tmp_path: Path):
        _, reporter, exp = self._setup(tmp_path)
        data = reporter.to_json(exp)
        assert data["id"] == exp.id
        assert data["config"]["name"] == "report-test"
        assert data["status"] == "completed"

    def test_to_markdown(self, tmp_path: Path):
        _, reporter, exp = self._setup(tmp_path)
        md = reporter.to_markdown(exp)
        assert "# Experiment: report-test" in md
        assert "## Configuration" in md
        assert "## Metrics" in md
        assert "| Model |" in md
        assert "| retrieval" in md.lower() or "retrieval" in md.lower()

    def test_comparison_to_markdown(self, tmp_path: Path):
        tracker, reporter, _ = self._setup(tmp_path)
        e1 = _start_and_complete(tracker, "base", retrieval={"mrr": 0.5})
        e2 = _start_and_complete(tracker, "better", retrieval={"mrr": 0.8})
        comp = tracker.compare(e1.id, e2.id)
        md = reporter.comparison_to_markdown(comp)
        assert "# Experiment Comparison" in md
        assert "## Metrics" in md
        assert "improved" in md or "REGRESSED" in md

    def test_trend_report_empty(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        reporter = ExperimentReporter(tracker)
        report = reporter.trend_report([])
        assert report == "No experiments to display."

    def test_trend_report_with_data(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        reporter = ExperimentReporter(tracker)
        e1 = _start_and_complete(tracker, "v1", retrieval={"mrr": 0.5})
        e2 = _start_and_complete(tracker, "v2", retrieval={"mrr": 0.8})
        report = reporter.trend_report([e1, e2])
        assert "# Experiment Trend Report" in report
        assert "Total experiments:" in report


# ------------------------------------------------------------------ #
#  _compare_experiments
# ------------------------------------------------------------------ #


class TestCompareExperiments:
    def test_summary_string(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        e1 = _start_and_complete(tracker, "a", retrieval={"mrr": 0.5})
        e2 = _start_and_complete(tracker, "b", retrieval={"mrr": 0.8})
        comp = tracker.compare(e1.id, e2.id)
        assert "improved" in comp.summary.lower() or "regressed" in comp.summary.lower()

    def test_empty_metrics(self, tmp_path: Path):
        tracker = ExperimentTracker(tmp_path)
        e1 = tracker.start_experiment(_make_config("a"))
        tracker.complete_experiment(e1.id, ExperimentMetrics())
        e2 = tracker.start_experiment(_make_config("b"))
        tracker.complete_experiment(e2.id, ExperimentMetrics())
        comp = tracker.compare(e1.id, e2.id)
        assert comp.comparisons == []
        assert comp.regressions == []
