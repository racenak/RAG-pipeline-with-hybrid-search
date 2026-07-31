#!/usr/bin/env python3
"""CLI — crawl a website and ingest into the RAG pipeline.

Usage:
    python scripts/crawl.py --url https://docs.prefect.io/v3/concepts --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl a website and ingest into RAG pipeline")
    parser.add_argument("--url", required=True, help="Seed URL to crawl")
    parser.add_argument("--limit", type=int, default=100, help="Max pages to crawl (default: 100)")
    parser.add_argument("--index", default=None, help="OpenSearch index name (default: rag-chunks-v1)")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy imports so --help is fast
    from rag_pipeline.pipeline import crawl_and_ingest

    storage = None
    if not args.no_s3:
        try:
            from rag_pipeline.data.storage import S3Storage
            storage = S3Storage()
            logging.getLogger(__name__).info("S3 storage connected")
        except Exception as e:
            logging.getLogger(__name__).warning("S3 unavailable, skipping: %s", e)

    print(f"\n{'='*60}")
    print(f"  Crawling: {args.url}")
    print(f"  Limit:    {args.limit} pages")
    print(f"  Index:    {args.index or 'rag-chunks-v1'}")
    print(f"{'='*60}\n")

    t0 = time.monotonic()
    result = crawl_and_ingest(
        url=args.url,
        limit=args.limit,
        storage=storage,
        index_name=args.index,

    )
    elapsed = time.monotonic() - t0

    print(f"\n{'='*60}")
    print(f"  Crawl complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Pages crawled:    {result.pages_crawled}")
    print(f"  Pages succeeded:  {result.pages_succeeded}")
    print(f"  Total chunks:     {result.total_chunks}")
    print(f"  OpenSearch:       {'indexed' if result.indexed_in_opensearch else 'FAILED'}")
    print(f"  PostgreSQL:       {'registered' if result.registered_in_postgres else 'FAILED'}")
    if result.errors:
        print(f"  Errors:           {len(result.errors)}")
        for err in result.errors[:5]:
            print(f"    - {err}")
    print("  Timings:")
    for step, t in result.timings.items():
        print(f"    {step:20s} {t:.2f}s")
    print(f"{'='*60}\n")

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
