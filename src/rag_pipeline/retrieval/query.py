"""Query processing — preprocessing, validation, type detection, and rewriting.

Provides a modular pipeline for transforming user queries before retrieval:
whitespace normalization, validation, type detection, and optional rewriting
strategies (expansion, HyDE, multi-query, step-back).

All rewriting strategies are deterministic and template-based — no LLM required.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Query types
# ------------------------------------------------------------------ #


class QueryType(StrEnum):
    """Detected query type."""

    FACTUAL = "factual"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    HOW_TO = "how_to"
    GENERAL = "general"


# ------------------------------------------------------------------ #
#  Processed query result
# ------------------------------------------------------------------ #


@dataclass
class ProcessedQuery:
    """Result of query processing — original + rewritten variants."""

    original: str
    rewritten: list[str] = field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def all_queries(self) -> list[str]:
        """Return original + rewritten queries (deduplicated, order preserved)."""
        seen: set[str] = set()
        queries: list[str] = []
        for q in [self.original, *self.rewritten]:
            normalized = q.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                queries.append(normalized)
        return queries


# ------------------------------------------------------------------ #
#  Abstract preprocessing / rewriting step
# ------------------------------------------------------------------ #


class QueryStep(ABC):
    """Base class for query processing steps."""

    @abstractmethod
    def process(self, query: str, ctx: dict[str, Any]) -> str | list[str]:
        """Process the query.

        Args:
            query: Current query text.
            ctx: Shared context dict (may carry metadata, rewritten queries, etc.).

        Returns:
            A single string (modified query) or a list of strings (multiple queries).
        """


# ------------------------------------------------------------------ #
#  Preprocessing steps
# ------------------------------------------------------------------ #


class WhitespaceNormalizer(QueryStep):
    """Collapse multiple whitespace characters and strip."""

    def process(self, query: str, ctx: dict[str, Any]) -> str:  # noqa: ARG002
        # Collapse runs of whitespace to a single space, then strip
        return re.sub(r"\s+", " ", query).strip()


class SpecialCharacterHandler(QueryStep):
    """Remove or escape special characters that may confuse search.

    Removes characters that are unlikely to be meaningful in search queries
    while preserving common punctuation (hyphens, apostrophes, dots).
    """

    # Characters to remove (control chars, backticks, etc.)
    _REMOVE_RE = re.compile(r"[`~\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def process(self, query: str, ctx: dict[str, Any]) -> str:  # noqa: ARG002
        return self._REMOVE_RE.sub("", query).strip()


class QueryValidator(QueryStep):
    """Validate that the query is non-empty and within length limits.

    Raises ValueError if the query fails validation.
    """

    def __init__(self, min_length: int = 1, max_length: int = 500) -> None:
        self._min_length = min_length
        self._max_length = max_length

    def process(self, query: str, ctx: dict[str, Any]) -> str:  # noqa: ARG002
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        stripped = query.strip()
        if len(stripped) < self._min_length:
            raise ValueError(
                f"Query too short ({len(stripped)} chars, minimum {self._min_length})"
            )
        if len(stripped) > self._max_length:
            raise ValueError(
                f"Query too long ({len(stripped)} chars, maximum {self._max_length})"
            )
        return stripped


class QueryTypeDetector(QueryStep):
    """Detect the query type using simple keyword heuristics.

    Populates ctx["query_type"] with the detected QueryType.
    """

    _PATTERNS: ClassVar[dict[QueryType, list[re.Pattern[str]]]] = {
        QueryType.FACTUAL: [
            re.compile(r"\b(who|what|when|where|which|how many|how much)\b", re.I),
            re.compile(r"\b(define|definition of|meaning of)\b", re.I),
        ],
        QueryType.SUMMARIZATION: [
            re.compile(r"\b(summarize|summary|summarise|briefly explain)\b", re.I),
            re.compile(r"\b(overview of|tldr|tl;dr)\b", re.I),
        ],
        QueryType.COMPARISON: [
            re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.I),
            re.compile(r"\b(pros and cons|advantages and disadvantages)\b", re.I),
        ],
        QueryType.HOW_TO: [
            re.compile(r"\b(how (do|can|to|should)|steps to|guide to)\b", re.I),
            re.compile(r"\b(tutorial|walkthrough|instructions for)\b", re.I),
        ],
    }

    def process(self, query: str, ctx: dict[str, Any]) -> str:
        detected = QueryType.GENERAL
        for qtype, patterns in self._PATTERNS.items():
            for pat in patterns:
                if pat.search(query):
                    detected = qtype
                    break
            if detected != QueryType.GENERAL:
                break

        ctx["query_type"] = detected
        return query


# ------------------------------------------------------------------ #
#  Rewriting steps
# ------------------------------------------------------------------ #


class QueryExpansion(QueryStep):
    """Expand query with synonyms/related terms from a built-in dictionary.

    Uses a small hand-curated synonym dictionary. No LLM required.
    """

    # Minimal synonym map — extend as needed
    _SYNONYMS: ClassVar[dict[str, list[str]]] = {
        "fix": ["repair", "resolve", "troubleshoot"],
        "error": ["bug", "issue", "problem", "fault"],
        "fast": ["quick", "rapid", "speedy"],
        "big": ["large", "huge", "massive"],
        "small": ["tiny", "little", "compact"],
        "start": ["begin", "initiate", "launch"],
        "stop": ["halt", "terminate", "end"],
        "create": ["make", "build", "generate"],
        "delete": ["remove", "destroy", "erase"],
        "help": ["assist", "support", "aid"],
        "use": ["utilize", "employ", "apply"],
        "show": ["display", "present", "reveal"],
        "find": ["locate", "search for", "discover"],
        "run": ["execute", "launch", "start"],
        "set": ["configure", "establish", "setup"],
        "get": ["retrieve", "obtain", "fetch"],
        "update": ["modify", "change", "alter"],
        "check": ["verify", "validate", "inspect"],
        "test": ["validate", "verify", "check"],
        "install": ["set up", "deploy", "configure"],
        "configure": ["setup", "adjust", "tune"],
        "improve": ["enhance", "optimize", "refine"],
        "performance": ["speed", "efficiency", "latency"],
        "security": ["safety", "protection", "auth"],
        "database": ["db", "datastore", "storage"],
        "request": ["call", "invocation", "api call"],
        "response": ["reply", "result", "output"],
        "input": ["parameter", "argument", "value"],
        "output": ["result", "return", "response"],
        "user": ["person", "individual", "end user"],
        "system": ["platform", "environment", "setup"],
        "data": ["information", "records", "content"],
        "file": ["document", "record", "artifact"],
        "function": ["method", "procedure", "routine"],
        "class": ["type", "object", "structure"],
        "variable": ["field", "property", "attribute"],
        "code": ["source", "implementation", "script"],
        "log": ["record", "entry", "event"],
        "message": ["notification", "alert", "communication"],
        "version": ["release", "iteration", "edition"],
        "config": ["configuration", "settings", "setup"],
        "auth": ["authentication", "authorization", "login"],
        "api": ["interface", "endpoint", "contract"],
        "url": ["link", "address", "uri"],
        "path": ["route", "directory", "location"],
        "error message": ["exception", "traceback", "stack trace"],
    }

    def process(self, query: str, ctx: dict[str, Any]) -> str:  # noqa: ARG002
        words = query.lower().split()
        expansions: list[str] = []

        for word in words:
            clean = word.strip(".,;:!?")
            if clean in self._SYNONYMS:
                # Add the first synonym that isn't already in the query
                for syn in self._SYNONYMS[clean]:
                    if syn not in query.lower():
                        expansions.append(syn)
                        break

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query


class HyDE(QueryStep):
    """Hypothetical Document Embeddings (template-based, no LLM).

    Wraps the query in a hypothetical-answer framing so that when embedded,
    the vector is closer to answer-space than question-space.

    The rewritten query is meant to be embedded and used for vector search.
    """

    _PREFIX = (
        "This document provides a detailed answer about: "
    )

    def process(self, query: str, ctx: dict[str, Any]) -> str:
        rewritten = f"{self._PREFIX}{query}"
        ctx["hyde_original"] = query
        return rewritten


class MultiQuery(QueryStep):
    """Generate multiple paraphrases of the query using templates.

    Produces 3-5 query variations to improve recall. No LLM required.
    """

    def __init__(self, num_queries: int = 4) -> None:
        self._num_queries = max(1, min(num_queries, 6))

    def process(self, query: str, ctx: dict[str, Any]) -> list[str]:  # noqa: ARG002
        queries: list[str] = [query]  # always include original

        templates = [
            "What is {q}?",
            "How does {q} work?",
            "Explain {q} in detail.",
            "Tell me about {q}.",
            "What should I know about {q}?",
            "Give me an overview of {q}.",
        ]

        # Strip trailing question marks for cleaner templating
        clean_q = query.rstrip("?.!").strip()

        for tmpl in templates[: self._num_queries - 1]:
            queries.append(tmpl.format(q=clean_q))

        return queries[: self._num_queries]


class StepBackPrompting(QueryStep):
    """Generate a broader "step-back" query for complex questions.

    Extracts key nouns/entities and creates a more general query
    to retrieve broader context before answering the specific question.
    """

    # Common stop words to filter out when extracting key terms
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "above", "below", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very", "just",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "whose", "when", "where", "why", "how", "i", "me", "my", "myself",
        "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
        "its", "they", "them", "their",
    })

    def process(self, query: str, ctx: dict[str, Any]) -> str:
        # Extract meaningful words (length > 2, not stop words)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query)
        key_terms = [w for w in words if w.lower() not in self._STOP_WORDS]

        if not key_terms:
            # Fallback: just ask for general context
            return "General information and context about the topic"

        # Create a broader query from key terms
        broad_query = " ".join(key_terms[:5])  # limit to top 5 terms
        step_back = f"Overview and background information about {broad_query}"

        ctx["step_back_original"] = query
        return step_back


# ------------------------------------------------------------------ #
#  QueryProcessor pipeline
# ------------------------------------------------------------------ #


@dataclass
class QueryProcessorConfig:
    """Configuration for the query processor pipeline."""

    # Preprocessing
    normalize_whitespace: bool = True
    handle_special_chars: bool = True
    validate_query: bool = True
    detect_query_type: bool = True
    min_query_length: int = 1
    max_query_length: int = 500

    # Rewriting (all disabled by default)
    expand_query: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    multi_query_count: int = 4
    use_step_back: bool = False


class QueryProcessor:
    """Chains preprocessing and optional rewriting steps.

    Usage::

        processor = QueryProcessor(config)
        result = processor.process("What is RAG?")
        print(result.all_queries)  # ['What is RAG?']
    """

    def __init__(self, config: QueryProcessorConfig | None = None) -> None:
        self._config = config or QueryProcessorConfig()
        self._preprocess_steps: list[QueryStep] = []
        self._rewrite_steps: list[QueryStep] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Assemble the processing pipeline from config."""
        cfg = self._config

        # Preprocessing (always in this order)
        if cfg.normalize_whitespace:
            self._preprocess_steps.append(WhitespaceNormalizer())
        if cfg.handle_special_chars:
            self._preprocess_steps.append(SpecialCharacterHandler())
        if cfg.validate_query:
            self._preprocess_steps.append(
                QueryValidator(cfg.min_query_length, cfg.max_query_length)
            )
        if cfg.detect_query_type:
            self._preprocess_steps.append(QueryTypeDetector())

        # Rewriting
        if cfg.expand_query:
            self._rewrite_steps.append(QueryExpansion())
        if cfg.use_hyde:
            self._rewrite_steps.append(HyDE())
        if cfg.use_multi_query:
            self._rewrite_steps.append(MultiQuery(cfg.multi_query_count))
        if cfg.use_step_back:
            self._rewrite_steps.append(StepBackPrompting())

    def process(self, query: str) -> ProcessedQuery:
        """Run the full query processing pipeline.

        Args:
            query: Raw user query.

        Returns:
            ProcessedQuery with original, rewritten queries, type, and latency.
        """
        start = time.perf_counter()
        ctx: dict[str, Any] = {}
        current = query

        # Preprocessing steps
        for step in self._preprocess_steps:
            try:
                result = step.process(current, ctx)
                if isinstance(result, str):
                    current = result
            except ValueError:
                # Validation failed — re-raise so caller knows
                raise
            except Exception:
                logger.warning("Query step %s failed, skipping", type(step).__name__)

        # Rewriting steps — collect all rewritten variants
        rewritten: list[str] = []
        for step in self._rewrite_steps:
            try:
                result = step.process(current, ctx)
                if isinstance(result, list):
                    rewritten.extend(result)
                elif isinstance(result, str) and result != current:
                    rewritten.append(result)
            except Exception:
                logger.warning("Rewrite step %s failed, skipping", type(step).__name__)

        latency_ms = (time.perf_counter() - start) * 1000

        return ProcessedQuery(
            original=current,
            rewritten=rewritten,
            query_type=ctx.get("query_type", QueryType.GENERAL),
            metadata=ctx,
            latency_ms=latency_ms,
        )


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #


def get_query_processor(
    settings: Any | None = None,
    **overrides: bool | int,
) -> QueryProcessor:
    """Create a QueryProcessor from settings or defaults.

    Args:
        settings: Optional RetrievalSettings or similar config object.
                  If None, uses defaults.
        **overrides: Override specific config fields
                     (e.g., expand_query=True, use_hyde=True).

    Returns:
        Configured QueryProcessor instance.
    """
    config = QueryProcessorConfig()

    # Try to pull config from settings object if provided
    if settings is not None:
        for attr in (
            "normalize_whitespace",
            "handle_special_chars",
            "validate_query",
            "detect_query_type",
            "expand_query",
            "use_hyde",
            "use_multi_query",
            "use_step_back",
        ):
            if hasattr(settings, attr):
                setattr(config, attr, getattr(settings, attr))

    # Apply explicit overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return QueryProcessor(config)
