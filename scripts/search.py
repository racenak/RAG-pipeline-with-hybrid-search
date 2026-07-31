#!/usr/bin/env python3
"""CLI for searching the RAG pipeline.

Usage:
    python scripts/search.py "How do deployments work?"
    python scripts/search.py "What are tasks?" --mode bm25
    python scripts/search.py "What are tasks?" --mode vector
    python scripts/search.py "What are tasks?" --mode hybrid --top-k 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _build_opensearch_client():
    from rag_pipeline.config import get_settings
    from rag_pipeline.storage.opensearch import OpenSearchClient

    s = get_settings().storage
    return OpenSearchClient(
        host=s.opensearch_host,
        port=s.opensearch_port,
        scheme=s.opensearch_scheme,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the RAG pipeline")
    parser.add_argument("query", help="Search query text")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "vector"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--index",
        default=None,
        help="OpenSearch index name (default: from config)",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        help="Metadata filters as key=value pairs",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    args = parser.parse_args()

    # Parse metadata filters
    metadata_filter = None
    if args.filter:
        metadata_filter = {}
        for f in args.filter:
            if "=" in f:
                k, v = f.split("=", 1)
                metadata_filter[k] = v

    # Build components
    print(f"  Query:  {args.query}")
    print(f"  Mode:   {args.mode}")
    print(f"  Top-k:  {args.top_k}")
    print()

    os_client = _build_opensearch_client()

    from rag_pipeline.config import get_settings
    settings = get_settings()
    index_name = args.index
    if index_name is None:
        index_name = f"{settings.storage.opensearch_index_prefix}-chunks-v1"

    # Build search engine
    from rag_pipeline.retrieval.bm25_index import OpenSearchBM25
    from rag_pipeline.retrieval.hybrid import HybridSearchConfig
    from rag_pipeline.retrieval.search import SearchEngine
    from rag_pipeline.retrieval.vector import OpenSearchVectorSearch

    bm25 = OpenSearchBM25(os_client, index_name)
    vector = OpenSearchVectorSearch(os_client, index_name)

    hybrid_config = HybridSearchConfig(
        vector_top_k=args.top_k,
        bm25_top_k=args.top_k,
        rrf_k=settings.retrieval.rrf_k,
    )
    engine = SearchEngine(
        vector_search=vector,
        bm25_search=bm25,
        hybrid_config=hybrid_config,
    )

    # Check query cache
    query_cache = _get_query_cache(settings)
    if query_cache is not None:
        try:
            cached = query_cache.get(args.query, args.top_k)
            if cached is not None:
                from rag_pipeline.retrieval.vector import SearchResult
                results = [SearchResult(**r) for r in cached]
                elapsed = 0
                print("  (cache hit)")
            else:
                raise ValueError("cache miss")
        except Exception:
            results = None
            elapsed = None
    else:
        results = None
        elapsed = None

    if results is None:
        # Embed query for vector search
        query_vector = None
        if args.mode in ("vector", "hybrid"):
            embedder = _get_embedder(settings)

            # Try embedding cache
            emb_cache = _get_embedding_cache(settings)
            if emb_cache is not None:
                with contextlib.suppress(Exception):
                    query_vector = emb_cache.get(args.query)

            if query_vector is None:
                query_vector = embedder.embed_single(args.query)
                if emb_cache is not None:
                    with contextlib.suppress(Exception):
                        emb_cache.set(args.query, query_vector)

        # Search
        t0 = time.monotonic()
        results = engine.search(
            query=args.query,
            query_vector=query_vector,
            top_k=args.top_k,
            mode=args.mode,
            metadata_filter=metadata_filter,
        )
        elapsed = time.monotonic() - t0

        # Cache results
        if query_cache is not None:
            with contextlib.suppress(Exception):
                query_cache.set(args.query, [r.to_dict() for r in results], args.top_k)

    # Output
    if args.json:
        output = {
            "query": args.query,
            "mode": args.mode,
            "count": len(results),
            "elapsed_ms": round(elapsed * 1000, 1),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n  Results: {len(results)} ({elapsed * 1000:.1f}ms)\n")
        for i, r in enumerate(results, 1):
            title = ""
            source_url = ""
            if r.metadata:
                title = r.metadata.get("title", "")
                source_url = r.metadata.get("source_url", "")

            header = f"  [{i}] score={r.score:.4f}"
            if title:
                header += f"  {title}"
            print(header)
            if source_url:
                print(f"      {source_url}")

            # Show content preview
            preview = r.content[:200].replace("\n", " ").strip()
            if args.verbose:
                print(f"      {r.content[:500]}")
            else:
                print(f"      {preview}...")
            print()


def _get_query_cache(settings):
    try:
        from rag_pipeline.storage.redis_cache import QueryCache
        s = settings.storage
        return QueryCache(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            ttl_seconds=s.redis_ttl_seconds,
        )
    except Exception:
        return None


def _get_embedding_cache(settings):
    try:
        from rag_pipeline.storage.redis_cache import RedisEmbeddingCache
        s = settings.storage
        return RedisEmbeddingCache(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            ttl_seconds=s.redis_ttl_seconds,
        )
    except Exception:
        return None


def _get_embedder(settings):
    from rag_pipeline.embeddings.generator import EmbeddingGenerator

    return EmbeddingGenerator(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        normalize=settings.embedding.normalize,
    )


if __name__ == "__main__":
    main()
