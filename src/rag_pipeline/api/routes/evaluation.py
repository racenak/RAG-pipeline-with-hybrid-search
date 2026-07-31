"""Evaluation API routes — run retrieval and generation evaluation."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from rag_pipeline.api.schemas.evaluation import EvalRequest, EvalResponse

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """Run evaluation on the golden dataset."""
    from evaluation.dataset import EvalDatasetManager
    from rag_pipeline.evaluation.retrieval import RetrievalEvaluator

    dataset_path = request.dataset_path or "evaluation/golden_dataset.json"
    try:
        manager = EvalDatasetManager(dataset_path)
        dataset = manager.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    cases = dataset.cases
    if request.category:
        cases = [c for c in cases if c.category.value == request.category]
    if request.difficulty:
        cases = [c for c in cases if c.difficulty.value == request.difficulty]

    evaluator = RetrievalEvaluator()
    # In real usage, pass a search function here
    return EvalResponse(
        status="completed",
        total_queries=len(cases),
        metrics={
            "message": "Evaluation pipeline ready. Provide a search function for real evaluation."
        },
    )


@router.get("/datasets")
async def list_datasets():
    """List available evaluation datasets."""
    dataset_dir = Path("evaluation")
    datasets = list(dataset_dir.glob("*.json"))
    return {"datasets": [d.name for d in datasets]}


@router.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get info about a specific dataset."""
    from evaluation.dataset import EvalDatasetManager

    path = Path("evaluation") / dataset_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    manager = EvalDatasetManager(path)
    dataset = manager.load()
    stats = manager.get_statistics()

    return {
        "name": dataset_name,
        "version": dataset.version,
        "description": dataset.description,
        "statistics": stats,
    }


@router.get("/metrics")
async def get_available_metrics():
    """List available evaluation metrics."""
    return {
        "retrieval": [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "ndcg_at_k",
            "hit_rate",
            "map",
        ],
        "generation": [
            "rouge_1",
            "rouge_l",
            "bleu",
            "word_overlap",
            "faithfulness",
            "relevance",
            "completeness",
        ],
        "latency": [
            "total_ms",
            "retrieval_ms",
            "generation_ms",
            "ttft_ms",
            "queries_per_second",
        ],
    }
