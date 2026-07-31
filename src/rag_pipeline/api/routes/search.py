"""Search API routes — with Redis caching, PostgreSQL enrichment."""

from __future__ import annotations

import contextlib
import logging
import time

from fastapi import APIRouter, HTTPException

from rag_pipeline.api.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ------------------------------------------------------------------ #
#  Component factories
# ------------------------------------------------------------------ #


def _get_search_components():
    """Build search engine from config — OpenSearch backends + embedder."""
    from rag_pipeline.config import get_settings
    from rag_pipeline.embeddings.generator import EmbeddingGenerator
    from rag_pipeline.retrieval.bm25_index import OpenSearchBM25
    from rag_pipeline.retrieval.hybrid import HybridSearchConfig
    from rag_pipeline.retrieval.search import SearchEngine
    from rag_pipeline.retrieval.vector import OpenSearchVectorSearch
    from rag_pipeline.storage.opensearch import OpenSearchClient

    settings = get_settings()
    s = settings.storage

    os_client = OpenSearchClient(
        host=s.opensearch_host,
        port=s.opensearch_port,
        scheme=s.opensearch_scheme,
    )

    index_name = f"{s.opensearch_index_prefix}-chunks-v1"
    bm25 = OpenSearchBM25(os_client, index_name)
    vector = OpenSearchVectorSearch(os_client, index_name)

    hybrid_config = HybridSearchConfig(
        vector_top_k=settings.retrieval.vector_top_k,
        bm25_top_k=settings.retrieval.bm25_top_k,
        rrf_k=settings.retrieval.rrf_k,
    )

    engine = SearchEngine(
        vector_search=vector,
        bm25_search=bm25,
        hybrid_config=hybrid_config,
    )

    embedder = EmbeddingGenerator(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        normalize=settings.embedding.normalize,
    )

    return engine, embedder


def _get_pg_client():
    """Create a PostgresClient from config."""
    from rag_pipeline.config import get_settings
    from rag_pipeline.storage.postgres import PostgresClient

    settings = get_settings()
    s = settings.storage
    return PostgresClient(
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_database,
        user=s.postgres_user,
        password=s.postgres_password,
    )


def _get_query_cache():
    """Create a QueryCache from config. Returns None if Redis is unavailable."""
    try:
        from rag_pipeline.config import get_settings
        from rag_pipeline.storage.redis_cache import QueryCache

        settings = get_settings()
        s = settings.storage
        return QueryCache(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            ttl_seconds=s.redis_ttl_seconds,
        )
    except Exception as e:
        logger.debug("Redis QueryCache unavailable: %s", e)
        return None


def _get_embedding_cache():
    """Create a RedisEmbeddingCache from config. Returns None if Redis is unavailable."""
    try:
        from rag_pipeline.config import get_settings
        from rag_pipeline.storage.redis_cache import RedisEmbeddingCache

        settings = get_settings()
        s = settings.storage
        return RedisEmbeddingCache(
            host=s.redis_host,
            port=s.redis_port,
            db=s.redis_db,
            ttl_seconds=s.redis_ttl_seconds,
        )
    except Exception as e:
        logger.debug("Redis EmbeddingCache unavailable: %s", e)
        return None


# ------------------------------------------------------------------ #
#  PostgreSQL enrichment
# ------------------------------------------------------------------ #


def _enrich_with_pg_metadata(results, metadata_filter=None):
    """Enrich search results with document metadata from PostgreSQL."""
    if not results:
        return []

    try:
        pg = _get_pg_client()

        # If metadata_filter provided, query PostgreSQL for matching docs
        pg_docs = {}
        if metadata_filter:
            docs = pg.find_documents_by_metadata(**metadata_filter)
            pg_docs = {d.id: d for d in docs}

        # Enrich each result with document metadata
        enriched = []
        for r in results:
            enriched_meta = dict(r.metadata) if r.metadata else {}

            # Find the document in PostgreSQL by crawl_group
            doc = None
            crawl_group = enriched_meta.get("crawl_group", "")
            if crawl_group and crawl_group in pg_docs:
                doc = pg_docs[crawl_group]
            elif crawl_group:
                with contextlib.suppress(Exception):
                    doc = pg.get_document(crawl_group)

            if doc:
                enriched_meta["pg_document_id"] = doc.id
                enriched_meta["pg_filename"] = doc.filename
                enriched_meta["pg_chunk_count"] = doc.chunk_count
                if doc.created_at:
                    enriched_meta["pg_created_at"] = doc.created_at.isoformat()

            enriched.append(SearchResultItem(
                id=r.id,
                score=r.score,
                content=r.content,
                metadata=enriched_meta,
            ))

        return enriched

    except Exception as e:
        logger.warning("PostgreSQL enrichment failed (returning raw results): %s", e)
        return [
            SearchResultItem(
                id=r.id,
                score=r.score,
                content=r.content,
                metadata=r.metadata,
            )
            for r in results
        ]


# ------------------------------------------------------------------ #
#  Search endpoint
# ------------------------------------------------------------------ #


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search the RAG index using vector, BM25, or hybrid mode.

    Flow:
      0. Query processing (normalize, validate, detect type)
      1. Check Redis query cache → return if hit
      2. Embed query (with Redis embedding cache)
      3. Search OpenSearch (vector + BM25 + RRF)
      4. Rerank with cross-encoder
      5. Enrich with PostgreSQL metadata
      6. Cache results in Redis
    """
    # 0. Query processing
    from rag_pipeline.retrieval.query import get_query_processor

    query_processor = get_query_processor()
    try:
        processed = query_processor.process(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {e}") from None

    search_query = processed.original  # use preprocessed query
    logger.debug("Query type: %s, rewritten: %d variants", processed.query_type, len(processed.rewritten))

    # 1. Check query cache
    query_cache = _get_query_cache()
    if query_cache is not None:
        try:
            cached = query_cache.get(search_query, request.top_k)
            if cached is not None:
                logger.info("Cache HIT for query: %s", search_query[:50])
                return SearchResponse(
                    query=request.query,
                    mode=request.mode,
                    count=len(cached),
                    elapsed_ms=0.0,
                    results=[SearchResultItem(**r) for r in cached],
                )
        except Exception as e:
            logger.debug("Query cache read failed: %s", e)

    # 2. Embed query (with Redis embedding cache)
    query_vector = None
    if request.mode in ("vector", "hybrid"):
        try:
            engine, embedder = _get_search_components()
        except Exception as e:
            logger.error("Failed to initialize search: %s", e)
            raise HTTPException(status_code=503, detail=f"Search unavailable: {e}") from None

        # Try embedding cache first
        emb_cache = _get_embedding_cache()
        if emb_cache is not None:
            try:
                cached_vector = emb_cache.get(search_query)
                if cached_vector is not None:
                    query_vector = cached_vector
                    logger.debug("Embedding cache HIT")
            except Exception as e:
                logger.debug("Embedding cache read failed: %s", e)

        # Compute embedding if not cached
        if query_vector is None:
            try:
                query_vector = embedder.embed_single(search_query)
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e}") from None

            # Cache the embedding
            if emb_cache is not None:
                try:
                    emb_cache.set(search_query, query_vector)
                except Exception as e:
                    logger.debug("Embedding cache write failed: %s", e)
    else:
        engine, embedder = _get_search_components()

    # 3. Search OpenSearch — search with original + rewritten queries for better recall
    t0 = time.monotonic()
    all_results_map: dict[str, float] = {}  # id → best score
    all_results_list: list = []

    queries_to_search = processed.all_queries[:5]  # limit to 5 queries max
    for q in queries_to_search:
        try:
            q_vector = embedder.embed_single(q) if request.mode in ("vector", "hybrid") else None
            q_results = engine.search(
                query=q,
                query_vector=q_vector,
                top_k=request.top_k,
                mode=request.mode,
                metadata_filter=request.metadata_filter,
                threshold=request.threshold,
            )
            for r in q_results:
                if r.id not in all_results_map or r.score > all_results_map[r.id]:
                    all_results_map[r.id] = r.score
                    # Update or add
                    existing_idx = next((i for i, x in enumerate(all_results_list) if x.id == r.id), None)
                    if existing_idx is not None:
                        all_results_list[existing_idx] = r
                    else:
                        all_results_list.append(r)
        except Exception as e:
            logger.warning("Search failed for rewritten query '%s': %s", q[:30], e)
            # Fall back to original query only
            if not all_results_list:
                raise

    # Sort by best score
    results = sorted(all_results_list, key=lambda x: x.score, reverse=True)[:request.top_k * 2]
    time.monotonic() - t0

    # 4. Rerank with cross-encoder
    from rag_pipeline.config import get_settings
    from rag_pipeline.retrieval.reranking import get_reranker

    settings = get_settings()
    reranker = get_reranker(settings.retrieval)
    try:
        results = reranker.rerank(search_query, results, top_k=request.top_k)
    except Exception as e:
        logger.warning("Reranking failed, using original results: %s", e)
        results = results[: request.top_k]

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    # 5. Enrich with PostgreSQL metadata
    enriched_results = _enrich_with_pg_metadata(results, request.metadata_filter)

    # 6. Cache results in Redis
    if query_cache is not None:
        try:
            cache_data = [r.model_dump() for r in enriched_results]
            query_cache.set(search_query, cache_data, request.top_k)
        except Exception as e:
            logger.debug("Query cache write failed: %s", e)

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        count=len(enriched_results),
        elapsed_ms=elapsed_ms,
        results=enriched_results,
    )

