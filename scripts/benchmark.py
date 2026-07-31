#!/usr/bin/env python3
"""Benchmark script — measure RAG pipeline performance.

Usage: python scripts/benchmark.py [OPTIONS]

Options:
  --iterations INT    Number of iterations per test (default: 100)
  --query TEXT         Test query (default: "What is the embedding dimension?")
  --output PATH       Save results to JSON file
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def benchmark_retrieval(query: str, iterations: int) -> dict:
    """Benchmark retrieval pipeline (BM25-only path — no OpenSearch needed)."""
    from rag_pipeline.retrieval.bm25 import BM25
    from rag_pipeline.retrieval.hybrid import HybridSearch, HybridSearchConfig
    from rag_pipeline.retrieval.vector import SearchResult

    bm25 = BM25()
    for i in range(20):
        bm25.add_document(f"doc{i}", f"Content about topic {i}")

    bm25_results = bm25.search(query, top_k=10)
    search_results = [
        SearchResult(id=r["id"], score=r["score"], content=r["content"])
        for r in bm25_results
    ]

    config = HybridSearchConfig(vector_top_k=10, bm25_top_k=10, rrf_k=60)
    search = HybridSearch(config=config)

    # Warmup
    for _ in range(5):
        search.fuse(bm25_results=search_results)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        search.fuse(bm25_results=search_results)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "test": "retrieval_hybrid",
        "iterations": iterations,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "stdev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        },
    }


def benchmark_query_processing(query: str, iterations: int) -> dict:
    """Benchmark query processing pipeline."""
    from rag_pipeline.retrieval.query import get_query_processor

    processor = get_query_processor()

    # Warmup
    for _ in range(5):
        processor.process(query)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        processor.process(query)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "test": "query_processing",
        "iterations": iterations,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
    }


def benchmark_bm25_index(query: str, iterations: int) -> dict:
    """Benchmark BM25 indexing and search."""
    from rag_pipeline.retrieval.bm25 import BM25

    bm25 = BM25()

    sample_docs = [
        ("doc1", "The embedding dimension is 1024 for BGE-large model"),
        ("doc2", "OpenSearch supports kNN vector search with HNSW algorithm"),
        ("doc3", "Redis provides TTL-based caching for query results"),
        ("doc4", "PostgreSQL stores document metadata and chunk information"),
        ("doc5", "FastAPI provides async endpoints for the RAG pipeline"),
    ]
    for doc_id, content in sample_docs:
        bm25.add_document(doc_id, content)

    # Warmup
    for _ in range(5):
        bm25.search(query)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        bm25.search(query)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return {
        "test": "bm25_search",
        "iterations": iterations,
        "documents_indexed": len(sample_docs),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--query", type=str, default="What is the embedding dimension?")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"RAG Pipeline Benchmark — {args.iterations} iterations")
    print("=" * 60)

    results = []

    print("\n[1/3] Benchmarking query processing...")
    r = benchmark_query_processing(args.query, args.iterations)
    results.append(r)
    print(f"  Mean: {r['latency_ms']['mean']}ms  P95: {r['latency_ms']['p95']}ms")

    print("\n[2/3] Benchmarking BM25 search...")
    r = benchmark_bm25_index(args.query, args.iterations)
    results.append(r)
    print(f"  Mean: {r['latency_ms']['mean']}ms  P95: {r['latency_ms']['p95']}ms")

    print("\n[3/3] Benchmarking hybrid search...")
    r = benchmark_retrieval(args.query, args.iterations)
    results.append(r)
    print(f"  Mean: {r['latency_ms']['mean']}ms  P95: {r['latency_ms']['p95']}ms")

    print("\n" + "=" * 60)
    print("Benchmark complete.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
