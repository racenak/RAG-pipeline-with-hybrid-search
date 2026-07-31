"""Evaluation — generation quality, latency, retrieval metrics, and experiment tracking."""

from rag_pipeline.evaluation.generation import (
    GenerationMetrics,
    GenerationResult,
    LLMJudge,
    bleu,
    detect_hallucination,
    rouge_1,
    rouge_l,
    word_overlap,
)
from rag_pipeline.evaluation.latency import LatencyMetrics, LatencyTracker
from rag_pipeline.evaluation.tracking import (
    Experiment,
    ExperimentComparison,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentReporter,
    ExperimentStatus,
    ExperimentTracker,
    MetricComparison,
)

__all__ = [
    "Experiment",
    "ExperimentComparison",
    "ExperimentConfig",
    "ExperimentMetrics",
    "ExperimentReporter",
    "ExperimentStatus",
    "ExperimentTracker",
    "GenerationMetrics",
    "GenerationResult",
    "LLMJudge",
    "LatencyMetrics",
    "LatencyTracker",
    "MetricComparison",
    "bleu",
    "detect_hallucination",
    "rouge_1",
    "rouge_l",
    "word_overlap",
]
