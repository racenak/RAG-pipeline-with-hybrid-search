"""BM25 scoring engine — pure Python implementation.

Okapi BM25 with configurable k1, b parameters.
Used for dev/testing; production uses OpenSearch-backed BM25.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Text preprocessing
# ------------------------------------------------------------------ #

# Common English stopwords
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
}


def tokenize(
    text: str,
    lower: bool = True,
    remove_stopwords: bool = False,
    stem: bool = False,
) -> list[str]:
    """Tokenize text into words.

    Args:
        text: Input text.
        lower: Lowercase tokens.
        remove_stopwords: Remove English stopwords.
        stem: Apply simple suffix stemming.
    """
    # Split on non-alphanumeric, keep alphanumeric tokens
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower() if lower else text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if stem:
        tokens = [_simple_stem(t) for t in tokens]

    return tokens


def _simple_stem(word: str) -> str:
    """Very simple suffix stripping — not a real stemmer, just reduces variants."""
    suffixes = ("ing", "tion", "ness", "ment", "ible", "able", "ies", "ed", "er", "ly", "s")
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ------------------------------------------------------------------ #
#  BM25 scoring
# ------------------------------------------------------------------ #


@dataclass
class BM25Doc:
    """A document in the BM25 index."""

    id: str
    content: str
    tokens: list[str] = field(default_factory=list)
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25:
    """Okapi BM25 scoring engine.

    Pure Python implementation for dev/testing. For production,
    use OpenSearchBM25 which delegates to OpenSearch's built-in BM25.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = False,
        stem: bool = False,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.stem = stem

        self._docs: dict[str, BM25Doc] = {}
        self._doc_freqs: dict[str, int] = {}  # term → number of docs containing term
        self._avg_dl: float = 0.0
        self._total_docs: int = 0

    @property
    def doc_count(self) -> int:
        return self._total_docs

    def _tokenize(self, text: str) -> list[str]:
        return tokenize(
            text,
            remove_stopwords=self.remove_stopwords,
            stem=self.stem,
        )

    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term using the Okapi formula."""
        n = self._doc_freqs.get(term, 0)
        # IDF = ln((N - n + 0.5) / (n + 0.5) + 1)
        return math.log((self._total_docs - n + 0.5) / (n + 0.5) + 1)

    def _score_doc(self, query_tokens: list[str], doc: BM25Doc) -> float:
        """Score a single document against query tokens."""
        if not doc.tokens:
            return 0.0

        score = 0.0
        dl = doc.token_count
        avg_dl = self._avg_dl if self._avg_dl > 0 else dl

        # Count term frequencies in document
        tf: dict[str, int] = {}
        for token in doc.tokens:
            tf[token] = tf.get(token, 0) + 1

        for term in query_tokens:
            if term not in tf:
                continue

            term_tf = tf[term]
            idf = self._compute_idf(term)

            # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avg_dl))
            numerator = term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
            score += idf * (numerator / denominator)

        return score

    # ------------------------------------------------------------------ #
    #  Index management
    # ------------------------------------------------------------------ #

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(content)
        doc = BM25Doc(
            id=doc_id,
            content=content,
            tokens=tokens,
            token_count=len(tokens),
            metadata=metadata or {},
        )
        self._docs[doc_id] = doc

        # Update document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        self._total_docs = len(self._docs)
        self._recompute_avg_dl()

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """Bulk-add documents.

        Each dict must have 'id' and 'content' keys.
        Optional: 'metadata' dict.
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata"),
            )

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index. Returns True if found."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False

        # Rebuild doc_freqs from scratch (simple but correct)
        self._doc_freqs.clear()
        for d in self._docs.values():
            for token in set(d.tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        self._total_docs = len(self._docs)
        self._recompute_avg_dl()
        return True

    def _recompute_avg_dl(self) -> None:
        if self._total_docs == 0:
            self._avg_dl = 0.0
        else:
            total = sum(d.token_count for d in self._docs.values())
            self._avg_dl = total / self._total_docs

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the index. Returns list of {id, score, content, metadata}."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        results: list[dict[str, Any]] = []
        for doc in self._docs.values():
            # Apply metadata filter
            if metadata_filter and not self._matches_filter(doc, metadata_filter):
                continue

            score = self._score_doc(query_tokens, doc)
            if score > 0:
                results.append(
                    {
                        "id": doc.id,
                        "score": score,
                        "content": doc.content,
                        "metadata": doc.metadata,
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _matches_filter(doc: BM25Doc, metadata_filter: dict[str, Any]) -> bool:
        """Check if a document matches metadata filter."""
        return all(doc.metadata.get(k) == v for k, v in metadata_filter.items())

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save index to JSON file."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "remove_stopwords": self.remove_stopwords,
            "stem": self.stem,
            "docs": {
                doc_id: {
                    "id": doc.id,
                    "content": doc.content,
                    "tokens": doc.tokens,
                    "token_count": doc.token_count,
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self._docs.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Saved BM25 index (%d docs) to %s", self._total_docs, path)

    @classmethod
    def load(cls, path: str | Path) -> BM25:
        """Load index from JSON file."""
        data = json.loads(Path(path).read_text())
        bm25 = cls(
            k1=data["k1"],
            b=data["b"],
            remove_stopwords=data["remove_stopwords"],
            stem=data["stem"],
        )
        for doc_data in data["docs"].values():
            bm25.add_document(
                doc_id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata"),
            )
        logger.info("Loaded BM25 index (%d docs) from %s", bm25.doc_count, path)
        return bm25
