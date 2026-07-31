"""CLI evaluation runner for the RAG pipeline.

Usage: python -m evaluation.run_eval [OPTIONS]

Options:
  --dataset PATH       Path to golden dataset JSON
  --mode MODE          Retrieval mode: vector|bm25|hybrid (default: hybrid)
  --top-k INT          Number of results to retrieve (default: 10)
  --output PATH        Output report path (default: evaluation/report.json)
  --category CAT       Filter by category
  --difficulty DIFF    Filter by difficulty
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.dataset import EvalDatasetManager
from evaluation.dataset_schema import DifficultyLevel, EvalCase, QueryCategory


def run_retrieval(
    case: EvalCase,
    mode: str = "hybrid",
    top_k: int = 10,
) -> dict:
    """Run retrieval for a single eval case.

    In production, this would call the actual RAG pipeline.
    Returns a result dict with retrieved documents and timing.
    """
    t0 = time.monotonic()

    retrieved_ids: list[str] = []
    scores: list[float] = []

    try:
        from rag_pipeline.retrieval.search import SearchEngine

        engine = SearchEngine()
        results = engine.search(
            query=case.query,
            top_k=top_k,
            mode=mode,
        )
        retrieved_ids = [r.id for r in results]
        scores = [r.score for r in results]
    except Exception:
        # Pipeline not available — return empty results for dry-run
        retrieved_ids = []
        scores = []

    latency_ms = (time.monotonic() - t0) * 1000

    return {
        "case_id": case.id,
        "query": case.query,
        "mode": mode,
        "retrieved_ids": retrieved_ids,
        "scores": scores,
        "latency_ms": round(latency_ms, 1),
    }


def compute_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
    top_k: int = 10,
) -> dict:
    """Compute retrieval metrics for a single case.

    Metrics: precision@k, recall@k, hit rate, MRR.
    """
    if not expected_ids:
        return {"precision": 0.0, "recall": 0.0, "hit": False, "mrr": 0.0}

    retrieved_set = set(retrieved_ids[:top_k])
    expected_set = set(expected_ids)

    hits = retrieved_set & expected_set
    precision = len(hits) / top_k if top_k > 0 else 0.0
    recall = len(hits) / len(expected_set) if expected_set else 0.0
    hit = len(hits) > 0

    # MRR: 1/rank of first relevant result
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            mrr = 1.0 / rank
            break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "hit": hit,
        "mrr": round(mrr, 4),
    }


def generate_report(
    case_results: list[dict],
    stats: dict,
    output_path: Path,
) -> None:
    """Generate and save the evaluation report."""
    total = len(case_results)
    avg_latency = (
        sum(r["latency_ms"] for r in case_results) / total if total > 0 else 0.0
    )

    all_metrics = [r["metrics"] for r in case_results]
    avg_precision = (
        sum(m["precision"] for m in all_metrics) / total if total > 0 else 0.0
    )
    avg_recall = (
        sum(m["recall"] for m in all_metrics) / total if total > 0 else 0.0
    )
    hit_rate = (
        sum(1 for m in all_metrics if m["hit"]) / total if total > 0 else 0.0
    )
    avg_mrr = sum(m["mrr"] for m in all_metrics) / total if total > 0 else 0.0

    report = {
        "summary": {
            "total_cases": total,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "hit_rate": round(hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
        },
        "dataset_stats": stats,
        "results": case_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {output_path}")
    print(f"Total cases: {total}")
    print(f"Avg precision@10: {avg_precision:.4f}")
    print(f"Avg recall@10: {avg_recall:.4f}")
    print(f"Hit rate: {hit_rate:.4f}")
    print(f"Avg MRR: {avg_mrr:.4f}")
    print(f"Avg latency: {avg_latency:.1f}ms")


def main() -> None:
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/golden_dataset.json",
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["vector", "bm25", "hybrid"],
        help="Retrieval mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=[c.value for c in QueryCategory],
        help="Filter by category",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=[d.value for d in DifficultyLevel],
        help="Filter by difficulty",
    )

    args = parser.parse_args()

    # Load dataset
    manager = EvalDatasetManager(args.dataset)
    dataset = manager.load()

    # Filter cases
    cases = dataset.cases
    if args.category:
        category = QueryCategory(args.category)
        cases = manager.filter_by_category(category)
    if args.difficulty:
        difficulty = DifficultyLevel(args.difficulty)
        cases = [c for c in cases if c.difficulty == difficulty]

    print(f"Running evaluation on {len(cases)} cases (mode={args.mode}, top_k={args.top_k})")

    # Run retrieval for each case
    case_results = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case.id}: {case.query[:60]}...")
        result = run_retrieval(case, mode=args.mode, top_k=args.top_k)
        metrics = compute_metrics(
            result["retrieved_ids"],
            case.expected_documents,
            top_k=args.top_k,
        )
        result["metrics"] = metrics
        case_results.append(result)

    # Generate report
    output_path = Path(args.output)
    generate_report(case_results, manager.get_statistics(), output_path)


if __name__ == "__main__":
    main()
