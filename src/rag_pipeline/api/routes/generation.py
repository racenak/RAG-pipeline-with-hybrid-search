"""Generation API routes — full RAG pipeline: search → rerank → generate."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator  # noqa: TC003

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag_pipeline.api.schemas.generation import (
    CitationBundleResponse,
    CitationResponse,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generate", tags=["generation"])


def _get_search_and_rerank():
    """Import and return search + rerank helpers."""
    from rag_pipeline.api.routes.search import _get_search_components
    from rag_pipeline.config import get_settings
    from rag_pipeline.retrieval.reranking import get_reranker

    settings = get_settings()
    engine, embedder = _get_search_components()
    reranker = get_reranker(settings.retrieval)
    return engine, embedder, reranker, settings


def _search_chunks(request: GenerationRequest):
    """Run search + rerank and return Chunk objects."""
    from rag_pipeline.data.chunking import Chunk

    engine, embedder, reranker, _settings = _get_search_and_rerank()

    # Embed query
    query_vector = None
    if request.mode in ("vector", "hybrid"):
        query_vector = embedder.embed_single(request.query)

    # Search
    results = engine.search(
        query=request.query,
        query_vector=query_vector,
        top_k=request.top_k * 2,
        mode=request.mode,
    )

    # Rerank
    results = reranker.rerank(request.query, results, top_k=request.top_k)

    # Convert to chunks
    chunks = []
    for i, r in enumerate(results):
        chunks.append(Chunk(
            id=r.id,
            document_id=r.metadata.get("document_id", "") if r.metadata else "",
            content=r.content,
            index=i,
            token_count=0,
            metadata={**(r.metadata or {}), "score": r.score},
        ))

    return chunks, results


@router.post("", response_model=GenerationResponse)
async def generate(request: GenerationRequest) -> GenerationResponse:
    """Run full RAG pipeline: search → rerank → generate answer."""
    from rag_pipeline.generation.generator import RAGGenerator
    from rag_pipeline.generation.llm import GenerationConfig

    t0 = time.monotonic()

    try:
        chunks, search_results = _search_chunks(request)
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Search failed: {e}") from None

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    gen_config = GenerationConfig(
        model=request.model or "gpt-4",
        temperature=request.temperature if request.temperature is not None else 0.7,
        max_tokens=request.max_tokens or 1024,
    )

    generator = RAGGenerator()
    try:
        result = await generator.agenerate(request.query, chunks, config=gen_config)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from None

    sources = [
        {"id": r.id, "content": r.content[:200], "score": r.score}
        for r in search_results
    ]

    citations_resp = None
    if result.citations is not None:
        citations_resp = CitationBundleResponse(
            citations=[
                CitationResponse(
                    marker=c.marker,
                    source_document=c.source_document,
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    page=c.page,
                    score=c.score,
                    text_snippet=c.text_snippet,
                )
                for c in result.citations.citations
            ],
            formatted_sources=result.citations.formatted_sources,
            validation_warnings=result.citations.validation_warnings,
        )

    return GenerationResponse(
        answer=result.answer,
        context_used=result.context_used,
        model=result.model,
        tokens_used=result.tokens_used,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
        sources=sources,
        citations=citations_resp,
    )


@router.post("/stream")
async def generate_stream(request: GenerationRequest) -> StreamingResponse:
    """SSE streaming endpoint for RAG generation."""

    async def event_stream() -> AsyncGenerator[str, None]:
        from rag_pipeline.generation.generator import RAGGenerator
        from rag_pipeline.generation.llm import GenerationConfig

        try:
            chunks, search_results = _search_chunks(request)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        if not chunks:
            yield f"data: {json.dumps({'error': 'No relevant documents found'})}\n\n"
            return

        gen_config = GenerationConfig(
            model=request.model or "gpt-4",
            temperature=request.temperature if request.temperature is not None else 0.7,
            max_tokens=request.max_tokens or 1024,
        )

        generator = RAGGenerator()
        try:
            async for token in generator.stream_generate(request.query, chunks, config=gen_config):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Send sources at the end
        sources = [
            {"id": r.id, "content": r.content[:200], "score": r.score}
            for r in search_results
        ]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
