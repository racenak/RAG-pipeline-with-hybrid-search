#!/usr/bin/env python3
"""CLI for running and comparing RAG pipeline experiments.

Usage:
    python scripts/run_experiment.py --name "hybrid-v1"
    python scripts/run_experiment.py --name "bm25-v2" --mode bm25 --top-k 5
    python scripts/run_experiment.py --name "vector-v1" --baseline "hybrid-v1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline.evaluation.tracking import (
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentReporter,
    ExperimentTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RAG pipeline experiment")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--description", default="", help="Experiment description")
    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--dataset",
        default="evaluation/golden_dataset.json",
        help="Path to golden dataset",
    )
    parser.add_argument("--baseline", default=None, help="Compare against baseline name")
    parser.add_argument(
        "--storage-dir",
        default="experiments",
        help="Experiment storage directory",
    )
    parser.add_argument("--tag", action="append", default=[], help="Add tag (repeatable)")
    parser.add_argument("--output-report", default=None, help="Save markdown report to file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        name=args.name,
        description=args.description,
        retrieval_mode=args.mode,
        top_k=args.top_k,
    )

    tracker = ExperimentTracker(storage_dir=args.storage_dir)
    reporter = ExperimentReporter(tracker)

    # Start experiment
    experiment = tracker.start_experiment(config)
    if args.tag:
        experiment = tracker.tag_experiment(experiment.id, args.tag)
    print(f"Started experiment: {experiment.id}")
    print(f"  Name:   {config.name}")
    print(f"  Mode:   {config.retrieval_mode}")
    print(f"  Top-K:  {config.top_k}")

    # Load dataset
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        with dataset_path.open() as f:
            dataset = json.load(f)
        num_cases = len(dataset.get("cases", []))
        print(f"  Dataset: {num_cases} cases from {dataset_path}")
        dataset_info = {
            "total_queries": num_cases,
            "source": str(dataset_path),
        }
    else:
        print(f"  Dataset: {dataset_path} (not found, skipping)")
        dataset_info = {"total_queries": 0, "source": str(dataset_path)}

    # Run retrieval evaluation (mock-friendly)
    print("\nRunning evaluation...")
    # In a real run, this would call the search engine and compute metrics.
    # For now, produce placeholder metrics.
    metrics = ExperimentMetrics(
        retrieval={
            "precision_at_5": 0.65,
            "recall_at_10": 0.72,
            "mrr": 0.78,
            "ndcg_at_10": 0.69,
            "hit_rate": 0.85,
        },
        generation={
            "faithfulness": 0.88,
            "relevance": 0.82,
            "completeness": 0.75,
        },
        latency={
            "total_ms": 1234.5,
            "retrieval_ms": 456.7,
            "generation_ms": 678.9,
        },
        cost={
            "api_calls": num_cases if dataset_path.exists() else 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )

    # Complete experiment
    experiment = tracker.complete_experiment(experiment.id, metrics)
    experiment = tracker.set_dataset_info(experiment.id, dataset_info)
    print(f"Completed experiment: {experiment.id}")
    print(f"  Duration: {experiment.duration_ms:.0f}ms")

    # Compare against baseline if requested
    has_regressions = False
    if args.baseline:
        baseline = tracker.get_baseline(args.baseline)
        if baseline is None:
            print(f"\nWarning: no completed baseline found with name '{args.baseline}'")
        else:
            comparison = tracker.compare(baseline.id, experiment.id)
            print(f"\n{comparison.summary}")
            if comparison.regressions:
                has_regressions = True

    # Generate report
    report_md = reporter.to_markdown(experiment)
    print(f"\nReport for {experiment.id}:")
    print(report_md)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md)
        print(f"\nReport saved to {out_path}")

    if has_regressions:
        print("\nRegressions detected — exiting with code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
