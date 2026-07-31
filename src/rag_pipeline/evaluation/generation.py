"""Generation quality metrics — reference-based scoring, hallucination detection, LLM-as-judge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.generation.llm import LLMBackend


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #


@dataclass
class GenerationMetrics:
    """Metrics for a single generation or aggregated."""

    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    hallucination_rate: float = 0.0


@dataclass
class GenerationResult:
    """Evaluated generation result."""

    query_id: str
    query: str
    answer: str
    expected_answer: str
    context_used: str
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)


# ------------------------------------------------------------------ #
#  Tokenisation helpers
# ------------------------------------------------------------------ #


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, drop empty tokens."""
    return [t for t in text.lower().split() if t]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _content_words(text: str) -> set[str]:
    """Extract meaningful content words (non-stopword, length >= 2)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "it",
        "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
        "what", "which", "who", "whom",
    }
    words = set()
    for token in re.findall(r'[a-z]+', text.lower()):
        if len(token) >= 2 and token not in stopwords:
            words.add(token)
    return words


# ------------------------------------------------------------------ #
#  Reference-based metrics
# ------------------------------------------------------------------ #


def rouge_1(answer: str, reference: str) -> float:
    """Unigram overlap F1 between *answer* and *reference*."""
    a_tokens = _tokenize(answer)
    r_tokens = _tokenize(reference)
    if not a_tokens or not r_tokens:
        return 0.0
    a_set = {}
    for t in a_tokens:
        a_set[t] = a_set.get(t, 0) + 1
    r_set = {}
    for t in r_tokens:
        r_set[t] = r_set.get(t, 0) + 1
    overlap = sum(min(a_set[t], r_set[t]) for t in a_set if t in r_set)
    precision = overlap / len(a_tokens) if a_tokens else 0.0
    recall = overlap / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def rouge_l(answer: str, reference: str) -> float:
    """Longest Common Subsequence F1 between *answer* and *reference*."""
    a_tokens = _tokenize(answer)
    r_tokens = _tokenize(reference)
    if not a_tokens or not r_tokens:
        return 0.0
    lcs_len = _lcs_length(a_tokens, r_tokens)
    precision = lcs_len / len(a_tokens) if a_tokens else 0.0
    recall = lcs_len / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute LCS length using dynamic programming."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def bleu(answer: str, reference: str, max_n: int = 4) -> float:
    """BLEU score with brevity penalty."""
    a_tokens = _tokenize(answer)
    r_tokens = _tokenize(reference)
    if not a_tokens or not r_tokens:
        return 0.0

    ans_len = len(a_tokens)
    ref_len = len(r_tokens)
    if ans_len == 0:
        return 0.0
    bp = min(1.0, __import__("math").exp(1.0 - ref_len / ans_len))

    log_avg = 0.0
    n_count = 0
    for n in range(1, max_n + 1):
        a_ngrams = _get_ngrams(a_tokens, n)
        r_ngrams = _get_ngrams(r_tokens, n)
        if not a_ngrams:
            continue
        clipped = sum(min(count, r_ngrams.get(ng, 0)) for ng, count in a_ngrams.items())
        total = sum(a_ngrams.values())
        precision = clipped / total if total > 0 else 0.0
        if precision == 0:
            return 0.0
        log_avg += __import__("math").log(precision)
        n_count += 1

    if n_count == 0:
        return 0.0
    return bp * __import__("math").exp(log_avg / n_count)


def _get_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    """Extract n-gram counts from a token list."""
    ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


def word_overlap(answer: str, reference: str) -> float:
    """Jaccard-like token overlap ratio."""
    a_tokens = set(_tokenize(answer))
    r_tokens = set(_tokenize(reference))
    if not a_tokens and not r_tokens:
        return 0.0
    intersection = a_tokens & r_tokens
    union = a_tokens | r_tokens
    return len(intersection) / len(union) if union else 0.0


# ------------------------------------------------------------------ #
#  Hallucination detection
# ------------------------------------------------------------------ #


def detect_hallucination(answer: str, context: str) -> tuple[float, list[str]]:
    """Detect hallucinated sentences.

    A sentence is flagged if >50% of its content words are NOT in the context.
    Returns (hallucination_rate, list_of_hallucinated_sentences).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0, []

    ctx_words = _content_words(context)
    hallucinated: list[str] = []

    for sent in sentences:
        sent_words = _content_words(sent)
        if not sent_words:
            continue
        missing = sent_words - ctx_words
        if len(missing) / len(sent_words) > 0.5:
            hallucinated.append(sent)

    rate = len(hallucinated) / len(sentences) if sentences else 0.0
    return rate, hallucinated


# ------------------------------------------------------------------ #
#  Heuristic scoring (no LLM required)
# ------------------------------------------------------------------ #


def _heuristic_faithfulness(answer: str, context: str) -> float:
    """Token overlap between answer and context."""
    return word_overlap(answer, context)


def _heuristic_relevance(query: str, answer: str) -> float:
    """Token overlap between query terms and answer."""
    return word_overlap(query, answer)


def _heuristic_completeness(query: str, answer: str, expected: str) -> float:
    """Token overlap between answer and expected."""
    return word_overlap(answer, expected)


# ------------------------------------------------------------------ #
#  LLM-as-Judge
# ------------------------------------------------------------------ #

_JUDGE_PROMPT_FAITHFULNESS = (
    "Rate the following answer on a scale of 0.0 to 1.0 for faithfulness "
    "(how well it is grounded in the provided context). "
    "Context: {context}\nAnswer: {answer}\nScore:"
)

_JUDGE_PROMPT_RELEVANCE = (
    "Rate the following answer on a scale of 0.0 to 1.0 for relevance "
    "(how well it addresses the query). "
    "Query: {query}\nAnswer: {answer}\nScore:"
)

_JUDGE_PROMPT_COMPLETENESS = (
    "Rate the following answer on a scale of 0.0 to 1.0 for completeness "
    "(how complete it is compared to the expected answer). "
    "Query: {query}\nAnswer: {answer}\nExpected: {expected}\nScore:"
)


class LLMJudge:
    """LLM-as-judge evaluation. Uses LLM to score quality."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    async def score_faithfulness(self, answer: str, context: str) -> float:
        """Score 0-1: Is the answer grounded in the context?"""
        if self._backend is None:
            return _heuristic_faithfulness(answer, context)
        return await self._llm_score(_JUDGE_PROMPT_FAITHFULNESS.format(context=context, answer=answer))

    async def score_relevance(self, query: str, answer: str) -> float:
        """Score 0-1: Is the answer relevant to the query?"""
        if self._backend is None:
            return _heuristic_relevance(query, answer)
        return await self._llm_score(_JUDGE_PROMPT_RELEVANCE.format(query=query, answer=answer))

    async def score_completeness(self, query: str, answer: str, expected: str) -> float:
        """Score 0-1: How complete is the answer vs expected?"""
        if self._backend is None:
            return _heuristic_completeness(query, answer, expected)
        return await self._llm_score(
            _JUDGE_PROMPT_COMPLETENESS.format(query=query, answer=answer, expected=expected)
        )

    async def _llm_score(self, prompt: str) -> float:
        """Send prompt to LLM and parse a numeric score."""
        from rag_pipeline.generation.llm import GenerationConfig

        response = await self._backend.generate(  # type: ignore[union-attr]
            [{"role": "user", "content": prompt}],
            GenerationConfig(temperature=0.0, max_tokens=10),
        )
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.0
