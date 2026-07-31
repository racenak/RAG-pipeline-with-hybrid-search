# Evaluation Framework

Evaluation is the feedback loop that turns a RAG pipeline from a prototype into a
reliable system. Without it, quality regressions go unnoticed, cost increases go
unexplained, and "it seems to work" becomes the only test. This document defines
how we measure retrieval quality, answer quality, latency, cost, and how we
automate those measurements in CI/CD.

---

## Table of Contents

1. [Evaluation Framework Overview](#1-evaluation-framework-overview)
2. [Retrieval Evaluation Metrics](#2-retrieval-evaluation-metrics)
3. [Answer Quality Metrics](#3-answer-quality-metrics)
4. [End-to-End RAG Evaluation](#4-end-to-end-rag-evaluation)
5. [Latency and Performance Metrics](#5-latency-and-performance-metrics)
6. [Benchmark Test Suite](#6-benchmark-test-suite)
7. [Observability and Monitoring](#7-observability-and-monitoring)
8. [Evaluation Configuration](#8-evaluation-configuration)

---

## 1. Evaluation Framework Overview

### Why Evaluation Is Critical

RAG pipelines are composite systems. A single user query touches embedding
generation, vector search, BM25 search, fusion, reranking, prompt assembly, and
LLM generation. A failure in any component silently degrades the output. Without
systematic evaluation, you cannot:

- Know which component is broken when answers are wrong.
- Detect regressions after infrastructure or model changes.
- Compare alternative configurations (different embeddings, rerankers, chunk
  sizes).
- Justify cost increases from switching to a more expensive model.

### Evaluation Dimensions

| Dimension            | What it measures                         | Frequency           |
|----------------------|------------------------------------------|---------------------|
| Retrieval quality    | Are the right documents retrieved?       | Every pipeline run  |
| Answer quality       | Is the generated answer correct and      | Every pipeline run  |
|                      | grounded in retrieved context?           |                     |
| Latency              | How fast does the pipeline respond?      | Every pipeline run  |
| Cost                 | How much does each query cost to serve?  | Aggregated daily    |
| Robustness           | Does it handle adversarial/edge queries? | Weekly              |

### Automated vs. Human Evaluation

**Automated evaluation** handles the bulk of quality measurement. It is fast,
reproducible, and cheap. It covers retrieval metrics (precision, recall, MRR),
reference-based answer metrics (ROUGE, BERTScore), and latency measurements.

**Human evaluation** is reserved for cases automated metrics miss: nuanced
correctness, factual accuracy beyond surface similarity, tone, and whether the
answer is genuinely helpful. Human evaluation runs on a sample of queries
(categorised by difficulty) and is used to calibrate automated metrics, not
replace them.

**LLM-as-judge** bridges the two. A stronger (more expensive) LLM evaluates
answers produced by the production pipeline. It scales better than human
evaluation but introduces its own biases and hallucination risks. Use it for
dimensions that are hard to measure with n-gram overlap: faithfulness, relevance,
completeness. Always validate LLM-as-judge scores against human ratings on a
subset.

### Benchmark Dataset Structure

Every evaluation dataset entry has this shape:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvalCase:
    """A single evaluation case for RAG pipeline assessment."""

    # Identity
    id: str                                          # Unique identifier
    question: str                                    # User query

    # Ground truth
    expected_answer: Optional[str] = None            # Reference answer (may be None for retrieval-only eval)
    expected_document_ids: list[str] = field(default_factory=list)  # Documents that SHOULD be retrieved
    expected_chunk_ids: list[str] = field(default_factory=list)     # Chunks that SHOULD be retrieved (finer-grained)

    # Metadata
    metadata: dict = field(default_factory=dict)     # Arbitrary key-value pairs for filtering
    difficulty: str = "medium"                       # easy | medium | hard | adversarial
    category: str = "general"                        # factual | multi_hop | summarization | comparison | edge_case

    # Filtering requirements (which index/metadata filters must apply)
    filter_requirements: dict = field(default_factory=dict)

    # Expected citations (for citation correctness evaluation)
    expected_citations: list[str] = field(default_factory=list)

    # Human annotation (populated after human review)
    human_quality_score: Optional[float] = None      # 1-5 scale
    human_notes: Optional[str] = None
```

Dataset files are stored in `evaluation/datasets/` as versioned YAML or JSONL
files. Each version is immutable once published.

---

## 2. Retrieval Evaluation Metrics

Retrieval evaluation answers one question: *did the pipeline find the right
evidence?* The answer depends on what "right" means and how the results are
ranked.

### Precision@k

The fraction of the top-k retrieved results that are relevant.

```
Precision@k = |{relevant docs in top-k}| / k
```

**When to use:** When retrieving too many irrelevant documents is costly (e.g.,
filling a limited context window for the LLM). A pipeline that retrieves 10
documents where only 2 are relevant wastes context and risks confusing the
generator.

**Limitation:** Does not penalise if the one relevant document is at position k
instead of position 1.

### Recall@k

The fraction of all relevant documents that appear in the top-k results.

```
Recall@k = |{relevant docs in top-k}| / |{all relevant docs}|
```

**When to use:** When missing relevant information is costly (e.g., medical or
legal Q&A where incomplete answers are dangerous). Recall@5 = 1.0 means every
relevant document appears in the top 5.

**Limitation:** Does not care about order. A recall of 1.0 with the relevant
document at position 5 is scored the same as at position 1.

### Mean Reciprocal Rank (MRR)

The average of the reciprocal ranks of the first relevant result across all
queries.

```
MRR = (1/|Q|) * Σ (1 / rank_i)
where rank_i is the position of the first relevant result for query i
```

**When to use:** When the user typically needs only one correct answer and
care mostly about whether it appears near the top. Common in single-answer
retrieval tasks.

**Limitation:** Only considers the first relevant result. Two relevant
documents at positions 1 and 2 score the same as one at position 1.

### Normalized Discounted Cumulative Gain (NDCG)

Measures ranking quality by rewarding relevant documents that appear higher in
the results, with logarithmic discounting.

```
DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

where `rel_i` is the graded relevance of the document at position i, and IDCG
is the ideal DCG (documents sorted by relevance).

**When to use:** When relevance is graded (not binary) and ranking order
matters. For example, a "highly relevant" document at position 1 is much better
than at position 5, and a "partially relevant" document is better than an
irrelevant one.

**Limitation:** Requires graded relevance labels, which are expensive to
produce.

### Hit Rate

A binary metric: did any of the top-k retrieved chunks contain the answer?

```
Hit Rate@k = |{queries where at least one relevant chunk is in top-k}| / |Q|
```

**When to use:** As a simple, intuitive baseline. "Did the retrieval step
produce at least one useful chunk?" is often the minimum bar for a functional
RAG system.

**Limitation:** Treats 1 relevant chunk the same as 5. Does not measure
ranking quality.

### Mean Average Precision (MAP)

The average of Average Precision (AP) across all queries. AP is the average of
precision values at each relevant document position.

```
AP = (1/|R|) * Σ_{k=1}^{n} (P@k * rel_k)
MAP = (1/|Q|) * Σ_{q=1}^{|Q|} AP_q
```

**When to use:** When you want a single number that captures both retrieval
completeness and ranking quality. MAP penalises both missing relevant
documents and ranking them poorly.

**Limitation:** Assumes binary relevance. Less suitable when relevance is
graded.

### When to Use Which Metric

| Scenario                                | Primary Metric | Secondary Metric |
|-----------------------------------------|----------------|------------------|
| Single-answer factual retrieval         | MRR            | Hit Rate@k       |
| Comprehensive evidence gathering        | Recall@k       | NDCG@k           |
| Limited context window (cost-sensitive) | Precision@k    | Recall@k         |
| Graded relevance judgments available    | NDCG@k         | MAP              |
| Quick smoke test                       | Hit Rate@k     | Precision@k      |
| Full ranking quality assessment         | MAP            | NDCG@k           |

### Implementation Notes

```python
import numpy as np
from typing import List, Set

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_retrieved) / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_retrieved) / len(relevant)

def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of first relevant result."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[str], relevance_scores: dict, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k with graded relevance."""
    def dcg(scores: List[float], k: int) -> float:
        return sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(scores[:k])
        )

    retrieved_scores = [relevance_scores.get(doc, 0.0) for doc in retrieved[:k]]
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]

    actual_dcg = dcg(retrieved_scores, k)
    ideal_dcg = dcg(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Binary: does any top-k result contain a relevant document?"""
    retrieved_at_k = retrieved[:k]
    return 1.0 if any(doc in relevant for doc in retrieved_at_k) else 0.0

def mean_average_precision(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]]
) -> float:
    """MAP across a set of queries."""
    aps = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        if not relevant:
            continue
        hits = 0
        sum_precision = 0.0
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                hits += 1
                sum_precision += hits / i
        ap = sum_precision / len(relevant) if relevant else 0.0
        aps.append(ap)
    return np.mean(aps) if aps else 0.0
```

---

## 3. Answer Quality Metrics

Answer quality evaluation goes beyond "did we find the right documents" to "did
we produce the right answer?" This is harder to automate because correctness
is often subjective.

### Faithfulness

Does the generated answer stay grounded in the retrieved context? A faithful
answer does not assert facts that cannot be traced to the retrieved documents.

**Scoring approach (LLM-as-judge):**

```python
FAITHFULNESS_PROMPT = """
You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Instructions:
1. Identify every factual claim in the answer.
2. For each claim, determine if it is supported by the context.
3. Score faithfulness:
   - 1.0: Every claim is supported by the context.
   - 0.8: Most claims are supported, minor unsupported details.
   - 0.5: Some claims are supported, some are not.
   - 0.2: Most claims are unsupported.
   - 0.0: The answer contradicts the context or is entirely unsupported.

Return ONLY a JSON object:
{{"score": <float>, "unsupported_claims": [<list of unsupported claims>]}}
"""
```

**Limitations:** LLM judges can miss subtle unsupported claims or penalise
reasonable inferences. Calibrate against human annotations.

### Relevance

Does the answer actually address the user's question? A technically correct
answer that misses the point of the question scores low on relevance.

**Scoring approach:**

```python
RELEVANCE_PROMPT = """
You are evaluating whether an answer is relevant to the question.

Question:
{question}

Answer:
{answer}

Score relevance:
- 1.0: Directly and completely answers the question.
- 0.8: Answers the question with minor tangential information.
- 0.5: Partially addresses the question.
- 0.2: Tangentially related but does not answer the question.
- 0.0: Completely irrelevant.

Return ONLY a JSON object:
{{"score": <float>, "explanation": "<brief explanation>"}}
"""
```

### Completeness

Does the answer cover all aspects of the question? A question with multiple
sub-questions should receive a completeness score reflecting how many
sub-questions are answered.

**Scoring approach:** Use the same LLM-as-judge pattern. The prompt should
list the sub-aspects of the question and check each one.

### Hallucination Detection

Hallucination occurs when the model asserts facts not present in the retrieved
context or known world knowledge. In a RAG context, we distinguish:

- **Context hallucination:** The answer claims something not in the retrieved
  documents.
- **External hallucination:** The answer contradicts established facts.

Detection strategies:

1. **NLI-based:** Use a Natural Language Inference model to check if each
   answer sentence is entailed by the context.
2. **LLM-as-judge:** Ask a stronger model to verify each claim.
3. **Citation checking:** If the system generates citations, verify the cited
   documents actually support the cited claim.

```python
HALLUCINATION_DETECTION_PROMPT = """
You are a fact-checker for a RAG system.

Context (retrieved documents):
{context}

Generated Answer:
{answer}

Task:
1. Extract every factual claim from the answer.
2. For each claim, check if it is supported by the context.
3. Flag any claim that is:
   - NOT present in the context (hallucination)
   - CONTRADICTED by the context (contradiction)

Return ONLY a JSON object:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "hallucinated_claims": [<list of strings>],
  "contradicted_claims": [<list of strings>],
  "hallucination_rate": <float between 0 and 1>
}}
"""
```

### LLM-as-Judge Approach

Using a stronger (usually larger or more expensive) LLM to evaluate answers
from the production pipeline. This scales better than human evaluation and
provides structured, interpretable scores.

**Best practices:**

- Use a model at least one tier larger than the production model (e.g.,
  GPT-4o to evaluate GPT-4o-mini outputs).
- Provide explicit scoring rubrics (not just "rate this 1-5").
- Include the full retrieved context in the evaluation prompt.
- Run multiple evaluations per query and average the scores to reduce
  variance.
- Validate LLM-as-judge scores against human ratings on at least 50 queries.
- Log the judge model's version alongside the scores.

**Known biases:**

- Length bias: longer answers tend to score higher.
- Position bias: the first option in a comparison tends to score higher.
- Self-preference: a model may rate its own outputs higher.

Mitigate by shuffling answer order, controlling for length, and using a
different model as judge than the one that generated the answer.

### Reference-Based Metrics

These compare the generated answer against a reference answer.

**BLEU (Bilingual Evaluation Understudy):** Measures n-gram overlap between
the generated and reference answer. Primarily designed for translation but
commonly used for text generation. Scores 0-1.

```python
from rouge_score import rouge_scorer

def compute_rouge(generated: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference, generated)
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
    }
```

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures
recall-oriented n-gram overlap. ROUGE-1 measures unigram overlap, ROUGE-2
measures bigram overlap, ROUGE-L measures longest common subsequence. Better
suited for summarisation tasks than BLEU.

**BERTScore:** Uses contextual embeddings to compute semantic similarity
between generated and reference answers. Captures meaning even when the
surface text differs.

```python
from bert_score import score as bert_score

def compute_bertscore(generated: str, reference: str, model_type: str = "microsoft/deberta-v3-large") -> dict:
    """Compute BERTScore between generated and reference."""
    P, R, F1 = bert_score(
        cands=[generated],
        refs=[reference],
        model_type=model_type,
        verbose=False
    )
    return {
        'precision': P.item(),
        'recall': R.item(),
        'f1': F1.item(),
    }
```

**When to use reference-based metrics:**

- BLEU: When exact wording matters (e.g., code generation, formulae).
- ROUGE: When recall of key information matters (e.g., summarisation).
- BERTScore: When semantic equivalence matters more than exact wording.

**Limitation:** All reference-based metrics require a reference answer, which
is expensive to produce. They also penalise correct answers that use different
wording than the reference.

### Reference-Free Metrics

These evaluate answer quality without a reference answer.

- **Coherence:** Is the answer well-structured and logically consistent?
- **Fluency:** Is the answer grammatically correct and natural-sounding?
- **Informativeness:** Does the answer contain substantive information?

These are typically scored by LLM-as-judge with a rubric prompt.

---

## 4. End-to-End RAG Evaluation

End-to-end evaluation treats the pipeline as a black box: query in, answer out.
This captures the combined effect of retrieval, generation, and their
interaction.

### RAGAS Framework Integration

RAGAS (Retrieval-Augmented Generation Assessment) provides a standardised set
of metrics for end-to-end RAG evaluation:

| Metric              | What it measures                                          |
|---------------------|-----------------------------------------------------------|
| Context Precision   | Are the retrieved chunks relevant to the answer?          |
| Context Recall      | Does the context contain all information needed for the   |
|                     | answer?                                                   |
| Faithfulness        | Is the answer grounded in the context?                    |
| Answer Relevancy    | Does the answer address the question?                     |
| Answer Similarity   | How similar is the answer to the reference answer?        |

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_similarity,
)

def evaluate_rag_pipeline(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truth: list[str],
) -> dict:
    """Run RAGAS evaluation on a set of queries."""
    dataset = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_similarity,
        ],
    )
    return result
```

### Evaluating the Full Pipeline

The full evaluation flow:

```
Query
  │
  ▼
┌─────────────────┐
│ Query Processing │  ← Is the query rewritten/expanded correctly?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Hybrid Search  │  ← BM25 + Vector search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Fusion       │  ← Reciprocal Rank Fusion or similar
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranking      │  ← Cross-encoder or LLM-based reranker
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prompt Assembly │  ← Context + query + instructions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Generation │  ← Answer generation
└────────┬────────┘
         │
         ▼
    Final Answer
```

At each stage, we can inject evaluation hooks:

1. **Retrieval stage:** Log retrieved document IDs and scores. Compare against
   expected document IDs.
2. **Reranking stage:** Log reranked order. Measure NDCG improvement over
   pre-reranking order.
3. **Generation stage:** Log the assembled prompt and generated answer. Run
   faithfulness/relevance checks.
4. **Post-generation:** Run hallucination detection and citation verification.

---

## 5. Latency and Performance Metrics

### End-to-End Latency Distribution

Track the full distribution, not just the mean:

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class LatencyMeasurement:
    """Records latency for a single pipeline execution."""
    query_id: str
    total_ms: float
    retrieval_ms: float
    reranking_ms: float
    generation_ms: float
    first_token_ms: Optional[float] = None  # Time to first token from LLM

def compute_latency_percentiles(measurements: list[LatencyMeasurement]) -> dict:
    """Compute latency percentiles across a set of measurements."""
    totals = sorted([m.total_ms for m in measurements])
    n = len(totals)
    return {
        "p50": totals[int(n * 0.5)],
        "p90": totals[int(n * 0.9)],
        "p95": totals[int(n * 0.95)],
        "p99": totals[int(n * 0.99)],
        "mean": sum(totals) / n,
        "min": totals[0],
        "max": totals[-1],
    }
```

Target baselines (adjust based on use case):

| Metric          | Target    | Notes                              |
|-----------------|-----------|------------------------------------|
| E2E latency p50 | < 2s      | User-facing search                 |
| E2E latency p95 | < 5s      | Acceptable for most use cases      |
| E2E latency p99 | < 10s     | Outliers acceptable occasionally   |
| Time-to-first-token | < 1.5s | Critical for streaming UX          |

### Retrieval Latency Breakdown

Measure each component separately to identify bottlenecks:

```python
@dataclass
class RetrievalLatencyBreakdown:
    """Detailed breakdown of retrieval-stage latency."""
    bm25_search_ms: float
    vector_search_ms: float
    fusion_ms: float
    reranking_ms: float
    total_retrieval_ms: float
```

Typical breakdown targets:

| Component       | Target    | Notes                                    |
|-----------------|-----------|------------------------------------------|
| BM25 search     | < 50ms    | Depends on index size                    |
| Vector search   | < 100ms   | Depends on index size and embedding dim  |
| Fusion          | < 10ms    | Typically negligible                      |
| Reranking       | < 500ms   | Cross-encoder is the main bottleneck     |
| **Total**       | < 600ms   | Target for retrieval stage               |

### Generation Latency

```python
@dataclass
class GenerationLatency:
    """Latency metrics for the LLM generation step."""
    time_to_first_token_ms: float
    total_generation_ms: float
    tokens_generated: int
    tokens_per_second: float
```

### Throughput

```python
def compute_throughput(
    measurements: list[LatencyMeasurement],
    window_seconds: float = 60.0,
) -> float:
    """Queries per second over a time window."""
    if not measurements:
        return 0.0
    # Assuming measurements have timestamps
    return len(measurements) / window_seconds
```

### Resource Utilization

Track alongside latency:

- **Memory usage:** Peak RSS during pipeline execution.
- **GPU utilization:** If running local embeddings or reranking models.
- **API costs:** Per-query cost broken down by component (embedding API,
  reranking API, LLM API).

```python
@dataclass
class CostBreakdown:
    """Cost breakdown for a single query."""
    embedding_cost_usd: float
    reranking_cost_usd: float
    llm_cost_usd: float
    total_cost_usd: float
    llm_input_tokens: int
    llm_output_tokens: int
```

---

## 6. Benchmark Test Suite

### Golden Dataset Structure and Creation

The golden dataset is the single source of truth for evaluation. It must be:

- **Versioned:** Every change is tracked in git.
- **Immutable once published:** New versions are created, never mutated.
- **Diverse:** Covers all query categories and difficulty levels.
- **Audited:** Regularly reviewed for correctness of ground truth labels.

Dataset creation process:

1. **Seed from production logs:** Sample real queries from production traffic.
2. **Augment with synthetic queries:** Use an LLM to generate queries of
   specific types (factual, multi-hop, summarisation, comparison).
3. **Expert annotation:** Domain experts verify ground truth answers and
   relevant document IDs.
4. **Adversarial examples:** Manually crafted edge cases that expose known
   failure modes.

### Test Categories

| Category        | Description                                        | Example query                              |
|-----------------|----------------------------------------------------|---------------------------------------------|
| Factual         | Single-hop factual questions                       | "What is the capital of France?"            |
| Multi-hop       | Requires information from multiple documents       | "How does X compare to Y based on Z?"       |
| Summarisation   | Requires synthesising information across documents | "Summarise the key findings about X"        |
| Comparison      | Requires comparing two or more entities            | "What are the differences between A and B?" |
| Edge cases      | Queries that stress the pipeline                   | Empty queries, very long queries, typos     |
| Adversarial     | Queries designed to cause hallucination            | "Explain quantum gravity in detail" (when no relevant docs exist) |
| Temporal        | Time-sensitive queries                             | "What happened last week about X?"          |
| Ambiguous       | Queries with multiple valid interpretations        | "Tell me about the bank"                    |

### Regression Testing

Compare pipeline runs over time to detect quality regressions.

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    run_id: str
    timestamp: str
    dataset_version: str
    pipeline_version: str
    embedding_model: str
    reranker_model: str
    metrics: dict          # {metric_name: value}
    latency_stats: dict    # {p50: ms, p95: ms, ...}
    total_queries: int
    error_count: int

class RegressionDetector:
    """Detects quality regressions between evaluation runs."""

    def __init__(self, baseline: EvaluationResult):
        self.baseline = baseline

    def check_regression(
        self,
        current: EvaluationResult,
        thresholds: dict[str, float] | None = None,
    ) -> dict:
        """
        Compare current results against baseline.

        thresholds: {metric_name: max_acceptable_drop}
        Example: {"recall@5": 0.05, "mrr": 0.03} means recall@5
        can drop by at most 0.05 before flagging a regression.
        """
        if thresholds is None:
            thresholds = {
                "recall@5": 0.05,
                "mrr": 0.03,
                "ndcg@10": 0.05,
                "faithfulness": 0.05,
                "p95_latency_ms": 500.0,
            }

        regressions = []
        for metric_name, max_drop in thresholds.items():
            baseline_val = self.baseline.metrics.get(metric_name)
            current_val = current.metrics.get(metric_name)
            if baseline_val is None or current_val is None:
                continue
            drop = baseline_val - current_val
            if drop > max_drop:
                regressions.append({
                    "metric": metric_name,
                    "baseline": baseline_val,
                    "current": current_val,
                    "drop": drop,
                    "threshold": max_drop,
                    "severity": "HIGH" if drop > max_drop * 2 else "MEDIUM",
                })

        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "baseline_run": self.baseline.run_id,
            "current_run": current.run_id,
        }
```

### CI/CD Integration

Evaluation runs on every PR that modifies:

- Retrieval configuration (BM25, vector search, fusion, reranking).
- Embedding model.
- Chunking strategy.
- Prompt templates.
- LLM model or parameters.

```yaml
# .github/workflows/eval.yml (conceptual)
name: Evaluation
on:
  pull_request:
    paths:
      - 'src/retrieval/**'
      - 'src/generation/**'
      - 'config/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - name: Run evaluation
        run: python -m evaluation.run --dataset evaluation/datasets/v2.jsonl --output evaluation/results/
      - name: Check regressions
        run: python -m evaluation.regression_check --baseline evaluation/baselines/main.json --current evaluation/results/latest.json
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: evaluation/results/
```

---

## 7. Observability and Monitoring

### Logging Strategy

All pipeline events use structured JSON logs. Every log entry includes:

```json
{
  "timestamp": "2026-07-27T10:30:00Z",
  "level": "info",
  "event_type": "retrieval",
  "trace_id": "abc-123-def",
  "query_id": "q-001",
  "pipeline_version": "v2.1.0",
  "latency_ms": 450,
  "details": {
    "bm25_results": 10,
    "vector_results": 10,
    "fused_results": 15,
    "reranked_results": 5,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
  }
}
```

Event types to log:

| Event Type          | Key fields                                           |
|---------------------|------------------------------------------------------|
| `query_received`    | query_text, user_id, timestamp                       |
| `query_rewritten`   | original_query, rewritten_query                      |
| `retrieval`         | bm25_results, vector_results, fused_results          |
| `reranking`         | pre_rerank_count, post_rerank_count, reranker_model  |
| `generation`        | prompt_tokens, completion_tokens, model              |
| `answer_generated`  | answer_length, latency_ms                            |
| `error`             | error_type, error_message, component                 |

### Tracing with OpenTelemetry

Distributed tracing tracks a single request through all pipeline components.

```python
from opentelemetry import trace

tracer = trace.get_tracer("rag-pipeline")

def process_query(query: str) -> str:
    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("query.text", query)

        with tracer.start_as_current_span("retrieve") as retrieve_span:
            documents = retrieve_documents(query)
            retrieve_span.set_attribute("retrieval.count", len(documents))

        with tracer.start_as_current_span("rerank") as rerank_span:
            reranked = rerank_documents(query, documents)
            rerank_span.set_attribute("rerank.count", len(reranked))

        with tracer.start_as_current_span("generate") as generate_span:
            answer = generate_answer(query, reranked)
            generate_span.set_attribute("generation.tokens", len(answer.split()))

        return answer
```

### Metrics Collection

Use Prometheus for metrics collection and Grafana for dashboards.

Key metrics to export:

```python
from prometheus_client import Histogram, Counter, Gauge

# Latency histograms
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval stage latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Generation stage latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

E2E_LATENCY = Histogram(
    "rag_e2e_latency_seconds",
    "End-to-end pipeline latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

# Counters
QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total queries processed",
    ["status"],  # success, error
)

RETRIEVAL_HITS = Counter(
    "rag_retrieval_hits_total",
    "Retrieval hit rate",
    ["k"],  # hit@1, hit@3, hit@5, hit@10
)

# Gauges
ACTIVE_DOCUMENTS = Gauge(
    "rag_active_documents",
    "Number of documents in the index",
)

INDEX_SIZE_BYTES = Gauge(
    "rag_index_size_bytes",
    "Size of the search index in bytes",
)
```

### Alerting Thresholds

| Alert                        | Condition                              | Severity |
|------------------------------|----------------------------------------|----------|
| High latency                 | p95 > 10s for 5 minutes                | Warning  |
| Very high latency            | p99 > 20s for 5 minutes                | Critical |
| Retrieval hit rate drop      | hit@5 < baseline - 0.1 for 1 hour     | Warning  |
| High error rate              | error_rate > 5% for 5 minutes          | Critical |
| Index staleness              | last_index_update > 24 hours           | Warning  |
| Cost spike                   | avg_cost_per_query > 2x baseline       | Warning  |

### Dashboard Design

**Overview dashboard:**
- Query volume (queries/minute)
- E2E latency (p50, p95, p99 over time)
- Error rate
- Active document count

**Retrieval dashboard:**
- BM25 vs vector search latency
- Fusion latency
- Reranking latency
- Retrieval hit rate (hit@k over time)

**Generation dashboard:**
- LLM latency (TTFT, total)
- Token usage (input/output)
- Hallucination rate (if measured)
- Answer length distribution

**Cost dashboard:**
- Cost per query (by component)
- Daily/weekly cost trends
- Cost by query category

---

## 8. Evaluation Configuration

### Configuring Evaluation Runs

Evaluation runs are configured via YAML files:

```yaml
# evaluation/configs/run_v2.yaml
evaluation:
  name: "hybrid-search-v2-eval"
  dataset: "evaluation/datasets/golden_v2.jsonl"
  output_dir: "evaluation/results/"

pipeline:
  embedding_model: "BAAI/bge-large-en-v1.5"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_retrieval: 20
  top_k_rerank: 5
  bm25_weight: 0.3
  vector_weight: 0.7

metrics:
  retrieval:
    - precision@5
    - recall@5
    - recall@10
    - mrr
    - ndcg@10
    - hit_rate@5
  answer:
    - faithfulness
    - relevance
    - completeness
  latency:
    - p50
    - p95
    - p99

judge:
  model: "gpt-4o"
  temperature: 0.0
  max_evals_per_query: 3  # Run multiple evaluations and average

regression:
  baseline: "evaluation/baselines/main.json"
  thresholds:
    recall@5: 0.05
    mrr: 0.03
    ndcg@10: 0.05
    faithfulness: 0.05
```

### Output Format and Report Generation

Evaluation results are stored in two formats:

1. **Machine-readable JSON** for programmatic consumption:

```json
{
  "run_id": "eval-20260727-103000",
  "timestamp": "2026-07-27T10:30:00Z",
  "config": {
    "dataset": "golden_v2.jsonl",
    "pipeline_version": "v2.1.0",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
  },
  "metrics": {
    "retrieval": {
      "precision@5": 0.72,
      "recall@5": 0.85,
      "recall@10": 0.93,
      "mrr": 0.78,
      "ndcg@10": 0.82,
      "hit_rate@5": 0.95
    },
    "answer": {
      "faithfulness": 0.88,
      "relevance": 0.91,
      "completeness": 0.84,
      "hallucination_rate": 0.06
    },
    "latency": {
      "p50_ms": 1200,
      "p95_ms": 3500,
      "p99_ms": 8000,
      "mean_ms": 1800
    },
    "cost": {
      "mean_cost_per_query_usd": 0.012
    }
  },
  "per_category": {
    "factual": { "recall@5": 0.92, "faithfulness": 0.93 },
    "multi_hop": { "recall@5": 0.74, "faithfulness": 0.82 },
    "summarization": { "recall@5": 0.88, "faithfulness": 0.86 }
  },
  "total_queries": 500,
  "error_count": 3
}
```

2. **Human-readable Markdown** for review:

```markdown
# Evaluation Report: hybrid-search-v2-eval

**Date:** 2026-07-27
**Dataset:** golden_v2.jsonl (500 queries)
**Pipeline:** v2.1.0

## Retrieval Metrics

| Metric       | Score | Baseline | Delta  |
|--------------|-------|----------|--------|
| Precision@5  | 0.72  | 0.70     | +0.02  |
| Recall@5     | 0.85  | 0.82     | +0.03  |
| Recall@10    | 0.93  | 0.91     | +0.02  |
| MRR          | 0.78  | 0.76     | +0.02  |
| NDCG@10      | 0.82  | 0.80     | +0.02  |
| Hit Rate@5   | 0.95  | 0.93     | +0.02  |

## Answer Quality

| Metric            | Score | Baseline | Delta  |
|-------------------|-------|----------|--------|
| Faithfulness      | 0.88  | 0.85     | +0.03  |
| Relevance         | 0.91  | 0.89     | +0.02  |
| Completeness      | 0.84  | 0.82     | +0.02  |
| Hallucination Rate| 0.06  | 0.08     | -0.02  |

## Latency

| Percentile | Time (ms) |
|------------|-----------|
| p50        | 1,200     |
| p95        | 3,500     |
| p99        | 8,000     |
| mean       | 1,800     |

## Regression Check: PASSED

No regressions detected against baseline main.json.
```

### Comparison Reports Between Pipeline Versions

When comparing two pipeline versions, generate a diff report:

```python
def generate_comparison_report(
    baseline: EvaluationResult,
    current: EvaluationResult,
) -> str:
    """Generate a Markdown comparison report."""
    lines = [
        "# Pipeline Comparison Report\n",
        f"**Baseline:** {baseline.pipeline_version} ({baseline.run_id})",
        f"**Current:** {current.pipeline_version} ({current.run_id})\n",
        "## Metric Comparison\n",
        "| Metric | Baseline | Current | Delta | Status |",
        "|--------|----------|---------|-------|--------|",
    ]

    for metric_name in baseline.metrics:
        base_val = baseline.metrics[metric_name]
        curr_val = current.metrics.get(metric_name, 0)
        delta = curr_val - base_val
        status = "PASS" if delta >= 0 else "REGRESSION"

        lines.append(
            f"| {metric_name} | {base_val:.3f} | {curr_val:.3f} | "
            f"{delta:+.3f} | {status} |"
        )

    return "\n".join(lines)
```

---

## Appendix: Evaluation Run Checklist

Before running a full evaluation:

- [ ] Dataset version is correct and up to date.
- [ ] Pipeline configuration matches the version being tested.
- [ ] LLM judge model is available and its version is logged.
- [ ] Baseline results exist for regression comparison.
- [ ] Output directory is clean or run ID is unique.
- [ ] Cost budget is approved (LLM judge calls are not free).

After running evaluation:

- [ ] Results are stored in both JSON and Markdown formats.
- [ ] Regression check has been run.
- [ ] Results are reviewed by at least one engineer.
- [ ] Any regressions have GitHub issues created.
- [ ] Results are linked in the relevant PR.
