# Retrieval System Design

This document defines the retrieval architecture for the RAG pipeline. It covers every stage from raw user query to final reranked context, with configuration schemas, code examples, and practical guidance.

## Table of Contents

1. [Vector Retrieval (Dense)](#1-vector-retrieval-dense)
2. [BM25 Retrieval (Sparse)](#2-bm25-retrieval-sparse)
3. [Hybrid Retrieval](#3-hybrid-retrieval)
4. [Reciprocal Rank Fusion (RRF)](#4-reciprocal-rank-fusion-rrf)
5. [Reranking](#5-reranking)
6. [Query Rewriting](#6-query-rewriting)
7. [Context Construction](#7-context-construction)
8. [Retrieval Configuration](#8-retrieval-configuration)

---

## End-to-End Retrieval Flow

```
+----------------+
|  User Query    |
+-------+--------+
        |
        v
+-------+--------+       +---------------------+
| Query Rewrite  |------>| Rewritten Queries   |
| (optional)     |       | (original preserved)|
+-------+--------+       +-----------+---------+
        |                            |
        +----------------------------+
        |
        v
+-------+--------+     +-------------------+
| BM25 Retrieval |---->| Sparse Results    |
+-------+--------+     +--------+----------+
        |                       |
        +-------+-------+-------+
                |
                v
+-------+--------+     +-------------------+
| Dense Vector   |---->| Dense Results     |
| Retrieval      |     +--------+----------+
+-------+--------+              |
        |                       |
        +-------+-------+-------+
                |
                v
        +-------+--------+
        | Reciprocal     |
        | Rank Fusion    |
        +-------+--------+
                |
                v
        +-------+--------+
        | Candidate Set  |
        | (merged,       |
        |  deduplicated) |
        +-------+--------+
                |
                v
        +-------+--------+
        | Cross-Encoder  |
        | Reranker       |
        +-------+--------+
                |
                v
        +-------+--------+
        | Final Top-K    |
        | Context        |
        +----------------+
```

---

## 1. Vector Retrieval (Dense)

Dense retrieval encodes both queries and documents into fixed-length vectors in a shared embedding space, then finds the nearest neighbors by geometric distance.

### How It Works

1. **Indexing**: Each document chunk is passed through an embedding model to produce a dense vector (e.g., 768 or 1024 dimensions). The vector is stored in a vector index.
2. **Querying**: The user query is embedded with the same model. The index returns the `k` vectors closest to the query vector.

### Embedding Models

| Model | Dimensions | Language | Notes |
|---|---|---|---|
| `BAAI/bge-large-en-v1.5` | 1024 | English | Strong general-purpose |
| `BAAI/bge-m3` | 1024 | Multilingual | Multi-lingual, multi-granularity |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | English | Fast, lighter weight |
| `text-embedding-3-small` (OpenAI) | 1536 | Multilingual | Managed API |
| `text-embedding-3-large` (OpenAI) | 3072 | Multilingual | Higher quality, higher cost |

**Critical rule**: Query and document embeddings must use the same model with identical configuration. Mixing models produces meaningless similarity scores.

### Distance Metrics

| Metric | Formula | Range | When to Use |
|---|---|---|---|
| **Cosine Similarity** | `dot(a,b) / (||a|| * ||b||)` | [-1, 1] | Default for normalized embeddings. Orientation matters, magnitude does not. |
| **Dot Product** | `dot(a,b)` | (-inf, inf) | When vectors are not normalized and magnitude carries meaning. |
| **L2 (Euclidean)** | `sqrt(sum((a-b)^2))` | [0, inf) | When absolute distance in embedding space matters. |

Most embedding models are trained with cosine similarity. Use cosine unless you have a specific reason not to.

```python
# Example: Computing similarity metrics manually
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
```

### Approximate Nearest Neighbor (ANN) Algorithms

Exact nearest-neighbor search is O(n) per query, which does not scale. ANN algorithms trade a small amount of accuracy for orders-of-magnitude speedup.

#### HNSW (Hierarchical Navigable Small World)

- **How it works**: Builds a multi-layer graph where each node connects to nearby neighbors. Search starts at the top (sparse) layer and descends to the bottom (dense) layer, greedily following closest neighbors.
- **Parameters**:
  - `M` (max connections per node): Higher = better recall, more memory. Typical: 16-64.
  - `ef_construction` (build-time search width): Higher = better graph quality, slower indexing. Typical: 100-200.
  - `ef_search` (query-time search width): Higher = better recall, slower query. Typical: 50-200.
- **Pros**: Best recall/speed tradeoff for most workloads. No training step.
- **Cons**: Memory-intensive. Full graph must fit in RAM.
- **Use when**: You need high recall with low latency and can afford the memory.

#### IVF (Inverted File Index)

- **How it works**: Partitions vectors into `nlist` clusters using k-means. At query time, probes the `nprobe` nearest clusters and searches only within those.
- **Parameters**:
  - `nlist` (number of clusters): sqrt(n) to 4*sqrt(n) is typical.
  - `nprobe` (clusters probed at query time): Higher = better recall, slower.
- **Pros**: Lower memory than HNSW. Supports on-disk indexes.
- **Cons**: Requires training step. Recall degrades if cluster boundaries are poor.
- **Use when**: Dataset is very large and memory is constrained.

#### PQ (Product Quantization)

- **How it works**: Compresses vectors by splitting them into subvectors and quantizing each to a codebook centroid. Enables sub-linear memory.
- **Parameters**:
  - `m` (number of subvectors): More = better accuracy, more memory/compute.
  - `nbits` (bits per centroid): Usually 8.
- **Pros**: Dramatic memory reduction (e.g., 1024-dim float32 -> 128 bytes).
- **Cons**: Lower recall. Often combined with IVF (IVF-PQ).
- **Use when**: Memory is the binding constraint and some recall loss is acceptable.

#### Choosing an Algorithm

```
Dataset size < 1M vectors  -->  HNSW (fast, high recall)
Dataset size > 1M vectors  -->  HNSW if memory allows, IVF-PQ otherwise
Hard real-time latency      -->  HNSW with bounded ef_search
Memory-constrained          -->  IVF-PQ
```

### Vector Store Options

| Store | Type | HNSW | IVF | On-disk | Filtering | Notes |
|---|---|---|---|---|---|---|
| **FAISS** | Library | Yes | Yes | Yes (faiss-cpu) | Pre-filter or post-filter | Meta's library. Best raw performance. No server. |
| **Chroma** | Embedded DB | Yes (via hnswlib) | No | Yes | Metadata filters | Simple API. Good for prototyping. |
| **Qdrant** | Server DB | Yes | Yes | Yes | Payload filters | Production-ready. gRPC + REST API. |
| **pgvector** | Postgres ext | Yes (IVFFlat, HNSW) | Yes | Yes (Postgres) | SQL WHERE | Leverages existing Postgres infrastructure. |
| **OpenSearch k-NN** | Server DB | Yes | Yes | Yes | Index-level filters | Already in our stack. |

### Configuration

```yaml
dense_retrieval:
  embedding_model: "BAAI/bge-large-en-v1.5"
  dimension: 1024
  distance_metric: "cosine"          # cosine | dot_product | l2
  index_type: "hnsw"                 # hnsw | ivf | ivf_pq
  top_k: 20                          # candidates to retrieve
  similarity_threshold: 0.0          # minimum similarity score
  ef_search: 128                     # HNSW query-time parameter
  collection: "documents_dense"
  filters:
    enabled: true
    supported_fields:
      - "source"
      - "language"
      - "document_type"
      - "created_at"
```

### When Dense Retrieval Works Well

- **Semantic similarity**: "What is the capital of France?" matches a document about "Paris, the French capital."
- **Paraphrase tolerance**: "How to fix a bug" and "debugging techniques" land near each other.
- **Cross-lingual** (with multilingual models): English query retrieves relevant German document.
- **Conceptual queries**: "What are the risks of machine learning?" retrieves documents discussing bias, overfitting, etc.

### When Dense Retrieval Struggles

- **Exact keywords / proper nouns**: "Python 3.11.4 release notes" may not match well if the embedding doesn't preserve version numbers precisely.
- **Numerical queries**: "Products under $50" — vectors blur numerical distinctions.
- **Rare or technical terms**: Uncommon jargon may be poorly represented in embedding space.
- **Very long queries**: Some models truncate or average away important signal.
- **Sparse queries**: Single-word queries produce weak vector representations.

---

## 2. BM25 Retrieval (Sparse)

BM25 is a lexical matching algorithm that scores documents based on term frequency and inverse document frequency. It remains one of the strongest single-stage retrieval baselines.

### BM25 Scoring Algorithm

The BM25 score for a query `Q` containing terms `q1, q2, ..., qn` against document `d`:

```
score(Q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))
```

Where:
- `IDF(qi)` = inverse document frequency of term `qi`
- `f(qi, d)` = term frequency of `qi` in document `d`
- `k1` = term frequency saturation parameter (typically 1.2-2.0)
- `b` = length normalization parameter (typically 0.75)
- `|d|` = document length
- `avgdl` = average document length in the corpus

**Intuition**:
- Terms appearing in many documents get lower weight (IDF).
- Terms appearing multiple times in a document boost the score, but with diminishing returns (k1 saturation).
- Longer documents are penalized but not eliminated (b parameter).

### Tokenization and Preprocessing Pipeline

```
Raw Text
   |
   v
Lowercasing
   |
   v
Unicode normalization (NFKC)
   |
   v
Stop word removal (optional)
   |
   v
Stemming / Lemmatization
   |
   v
N-gram generation (optional)
   |
   v
Tokens
```

**Language-specific considerations**:

| Language | Tokenizer | Stemmer | Notes |
|---|---|---|---|
| English | Whitespace + punctuation | Snowball / Porter | Straightforward |
| German | Compound splitting (egerhorn) | Snowball | Compound words are common |
| Chinese | Jieba / pkuseg | None (character n-grams) | No whitespace boundaries |
| Japanese | MeCab / SudachiPy | None | Requires morphological analysis |
| Arabic | CAMeL Tools | Snowball | Right-to-left, diacritics |

```python
# Example: Multilingual tokenization pipeline
import re
from typing import List

class Tokenizer:
    """Base tokenizer with language-aware preprocessing."""

    def __init__(self, language: str = "en", lowercase: bool = True,
                 remove_stopwords: bool = False):
        self.language = language
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = self._load_stopwords(language)

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        # Basic whitespace tokenization
        tokens = re.findall(r'\w+', text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def _load_stopwords(self, language: str) -> set:
        # Placeholder — load from NLTK, spaCy, or custom list
        return set()
```

### IDF Weighting and Term Frequency

**IDF variants**:

```
Standard:    IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
Probabilistic: IDF(q) = log((N - n(q)) / n(q))
```

Where `N` = total documents, `n(q)` = documents containing term `q`.

**Term frequency variants**:

| Variant | Formula | Behavior |
|---|---|---|
| Raw count | `f(t,d)` | Linear scaling |
| Boolean | `1 if f(t,d) > 0 else 0` | Presence only |
| Log-normalized | `1 + log(f(t,d) if f(t,d) > 0 else 0)` | Diminishing returns |
| BM25 saturation | `f(t,d) * (k1+1) / (f(t,d) + k1)` | Controlled saturation |

### Index Construction

```python
# Example: Building a BM25 index with rank_bm25
from rank_bm25 import BM25Okapi
import json

class BM25Index:
    """BM25 index with metadata support."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = []
        self.metadata = []
        self.bm25 = None

    def add_documents(self, documents: list[dict]):
        """
        documents: list of {"id": str, "content": str, "metadata": dict}
        """
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            self.corpus_tokens.append(tokens)
            self.metadata.append({
                "id": doc["id"],
                "content": doc["content"],
                **doc.get("metadata", {})
            })
        self.bm25 = BM25Okapi(
            self.corpus_tokens,
            k1=self.k1,
            b=self.b
        )

    def search(self, query: str, top_k: int = 10,
               filters: dict = None) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call add_documents first.")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters (post-filter approach)
        if filters:
            for i, meta in enumerate(self.metadata):
                if not self._matches_filters(meta, filters):
                    scores[i] = -1.0  # Exclude from results

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] < 0:
                continue
            results.append({
                "chunk_id": self.metadata[idx]["id"],
                "content": self.metadata[idx]["content"],
                "score": float(scores[idx]),
                "rank": rank + 1,
                "retrieval_method": "bm25",
                "metadata": {k: v for k, v in self.metadata[idx].items()
                             if k not in ("id", "content")}
            })
        return results

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r'\w+', text.lower())

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
```

### Query Expansion Techniques

Query expansion adds related terms to the original query to improve recall.

**Pseudo-relevance feedback (PRF)**:
1. Retrieve top-N documents with the original query.
2. Extract the most frequent/important terms from those documents.
3. Append those terms to the query and re-retrieve.

```python
def expand_query_bm25(bm25_index, original_query: str,
                      top_n: int = 5, num_expansion_terms: int = 3) -> str:
    """Expand query using pseudo-relevance feedback."""
    # Step 1: Retrieve top-N documents
    initial_results = bm25_index.search(original_query, top_k=top_n)

    # Step 2: Extract frequent terms from results
    from collections import Counter
    term_counts = Counter()
    for result in initial_results:
        tokens = bm25_index._tokenize(result["content"])
        term_counts.update(tokens)

    # Remove terms already in the query
    query_tokens = set(bm25_index._tokenize(original_query))
    expansion_terms = [
        term for term, count in term_counts.most_common(num_expansion_terms * 3)
        if term not in query_tokens and len(term) > 2
    ][:num_expansion_terms]

    # Step 3: Build expanded query
    expanded = f"{original_query} {' '.join(expansion_terms)}"
    return expanded
```

**Other expansion approaches**:
- **Synonym expansion**: Use a thesaurus or WordNet to add synonyms.
- **LLM expansion**: Ask an LLM to rewrite the query with additional keywords (see [Query Rewriting](#6-query-rewriting)).
- **Word2Vec neighbors**: Add words that are close in word embedding space.

### When BM25 Works Well

- **Exact keyword matches**: "CVE-2024-1234" finds the exact advisory.
- **Proper nouns, codes, identifiers**: "React useEffect" matches precisely.
- **Boolean-style queries**: Users who type specific terms.
- **Domain-specific jargon**: Technical terms that embed poorly.
- **No training data needed**: Works out of the box.

### When BM25 Struggles

- **Semantic understanding**: "affordable housing options" won't match "low-cost residences."
- **Synonym gaps**: "car" vs "automobile" vs "vehicle."
- **Cross-lingual**: English query cannot find French documents.
- **Long, complex queries**: Term frequency signal gets diluted.
- **Typographical errors**: "pythn" won't match "python" without fuzzy matching.

---

## 3. Hybrid Retrieval

Hybrid retrieval combines lexical (BM25) and semantic (dense vector) search to get the best of both approaches.

### Why Hybrid Beats Either Alone

| Scenario | BM25 Alone | Dense Alone | Hybrid |
|---|---|---|---|
| Exact keyword match | Strong | Weak | Strong |
| Semantic similarity | Weak | Strong | Strong |
| Synonym tolerance | Weak | Strong | Strong |
| Rare term precision | Strong | Weak | Strong |
| Cross-lingual | None | Possible | Possible |
| Robustness to noise | Moderate | Moderate | High |

BM25 and dense retrieval make **complementary errors**. Documents that BM25 misses (semantic matches), dense retrieval often finds, and vice versa. Hybrid retrieval captures both.

### Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +--------+---------+         +--------+---------+
     | BM25 Index       |         | Vector Index     |
     | (Sparse Search)  |         | (Dense Search)   |
     +--------+---------+         +--------+---------+
              |                             |
              v                             v
     +--------+---------+         +--------+---------+
     | BM25 Results     |         | Dense Results    |
     | (doc, score,     |         | (doc, score,     |
     |  rank)           |         |  rank)           |
     +--------+---------+         +--------+---------+
              |                             |
              +--------------+--------------+
                             |
                             v
                    +--------+---------+
                    | Score Fusion     |
                    | (RRF)            |
                    +--------+---------+
                             |
                             v
                    +--------+---------+
                    | Merged Results   |
                    +------------------+
```

### Query Processing

The same query is sent to both engines. No transformation is applied at the hybrid layer — individual engines may apply their own preprocessing.

```python
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievalResult:
    document_id: str
    chunk_id: str
    content: str
    score: float
    retrieval_method: str  # "bm25" | "dense" | "hybrid"
    rank: int
    metadata: dict

class HybridRetriever:
    """Runs BM25 and dense retrieval in parallel, fuses with RRF."""

    def __init__(self, bm25_index, vector_index, rrf_k: int = 60):
        self.bm25 = bm25_index
        self.vector = vector_index
        self.rrf_k = rrf_k

    async def retrieve(self, query: str, top_k: int = 10,
                       filters: dict = None) -> list[RetrievalResult]:
        """Run both retrievers in parallel, fuse results."""

        # Run both in parallel
        bm25_task = asyncio.create_task(
            self._run_bm25(query, top_k * 2, filters)
        )
        dense_task = asyncio.create_task(
            self._run_dense(query, top_k * 2, filters)
        )
        bm25_results, dense_results = await asyncio.gather(
            bm25_task, dense_task
        )

        # Fuse with RRF
        fused = self._rrf_fusion(bm25_results, dense_results)

        return fused[:top_k]

    async def _run_bm25(self, query: str, top_k: int,
                        filters: dict) -> list[RetrievalResult]:
        # BM25 is synchronous; run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.bm25.search(query, top_k, filters)
        )

    async def _run_dense(self, query: str, top_k: int,
                         filters: dict) -> list[RetrievalResult]:
        return await self.vector.search(query, top_k, filters)

    def _rrf_fusion(self, *result_lists: list[RetrievalResult]) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion across multiple result lists."""
        from collections import defaultdict
        scores = defaultdict(float)
        chunk_map = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                key = result.chunk_id
                scores[key] += 1.0 / (self.rrf_k + rank + 1)
                if key not in chunk_map:
                    chunk_map[key] = result

        # Sort by fused score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        fused = []
        for rank, key in enumerate(sorted_keys):
            result = chunk_map[key]
            fused.append(RetrievalResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=scores[key],
                retrieval_method="hybrid",
                rank=rank + 1,
                metadata=result.metadata
            ))
        return fused
```

### Latency Considerations

The total latency of hybrid retrieval is determined by the **slowest** retriever, since both run in parallel:

```
Total Latency = max(BM25 Latency, Dense Latency) + Fusion Latency
```

Typical latencies (on a reasonably sized index):

| Component | Latency (p50) | Latency (p99) |
|---|---|---|
| BM25 (10K docs) | ~2ms | ~10ms |
| BM25 (1M docs) | ~10ms | ~50ms |
| Dense (HNSW, 10K docs) | ~3ms | ~15ms |
| Dense (HNSW, 1M docs) | ~8ms | ~40ms |
| Fusion (RRF) | <1ms | <1ms |
| **Total (parallel)** | **~10ms** | **~50ms** |

If one engine is significantly slower (e.g., dense retrieval on a large corpus without GPU), consider:
- Running dense retrieval with a smaller `top_k` candidate set.
- Using a dedicated embedding service with GPU acceleration.
- Setting a timeout on the slower engine and falling back to the faster one.

---

## 4. Reciprocal Rank Fusion (RRF)

### Formula

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))
```

Where:
- `d` is a document
- `rank_i(d)` is the rank of document `d` in retrieval method `i` (1-indexed)
- `k` is a smoothing constant (typically 60)
- The sum runs over all retrieval methods (BM25, dense, etc.)

### Choosing the k Parameter

The `k` parameter controls how much rank position matters:

| k Value | Effect |
|---|---|
| `k = 1` | Aggressive — top-ranked documents dominate |
| `k = 10` | Moderate — still favors high ranks |
| `k = 60` | Standard — balanced distribution |
| `k = 100` | Conservative — flatter score distribution |
| `k = 200` | Very flat — rank differences matter less |

**Why k=60 is standard**: The original RRF paper (Cormack, Clarke, Butt 2009) found k=60 worked well across multiple TREC collections. It provides enough smoothing that rank-1 and rank-2 documents are not vastly different, while still giving higher-ranked documents appropriate preference.

```python
def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    weights: list[float] = None
) -> list[dict]:
    """
    Fuse multiple ranked lists using RRF.

    Args:
        result_lists: List of ranked result lists. Each list contains
                      dicts with at least 'chunk_id' and 'content'.
        k: Smoothing parameter (default 60).
        weights: Optional weights for each result list.
                 If None, equal weights are used.

    Returns:
        Merged list sorted by RRF score descending.
    """
    from collections import defaultdict

    if weights is None:
        weights = [1.0] * len(result_lists)

    assert len(result_lists) == len(weights), \
        "Number of result lists must match number of weights"

    scores = defaultdict(float)
    chunk_data = {}

    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] += weight / (k + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    # Sort by score descending, then by chunk_id for determinism
    sorted_ids = sorted(
        scores.keys(),
        key=lambda cid: (-scores[cid], cid)
    )

    merged = []
    for rank, chunk_id in enumerate(sorted_ids):
        merged.append({
            **chunk_data[chunk_id],
            "score": scores[chunk_id],
            "rank": rank + 1,
            "retrieval_method": "rrf_hybrid"
        })

    return merged
```

### Weighted RRF

When one retrieval method is more reliable than another, apply weights:

```python
# Example: Dense retrieval is stronger for this domain
fused = reciprocal_rank_fusion(
    [bm25_results, dense_results],
    k=60,
    weights=[0.4, 0.6]  # Favor dense retrieval
)
```

**Weight tuning guidelines**:
- Start with equal weights `[1.0, 1.0]`.
- If evaluation shows BM25 is underperforming, reduce its weight.
- If the domain is heavily technical (code, medical), increase BM25 weight.
- If the domain is conversational, increase dense weight.
- Tune weights on an evaluation set, not by intuition.

### Handling Ties and Duplicates

**Duplicate documents** across retrieval methods are the common case — they should be merged, not duplicated:

```python
# The RRF function above handles this naturally:
# If document D appears in both BM25 (rank 3) and dense (rank 5),
# its RRF score = 1/(60+3) + 1/(60+5) = 0.01587 + 0.01538 = 0.03125
# A document in only one method at rank 3: 1/(60+3) = 0.01587
# Documents in both methods are rewarded, which is the desired behavior.
```

**Ties in rank**: When two documents have the same rank within one method, break ties deterministically (e.g., by chunk_id lexicographic order).

### Why RRF Over Score-Based Fusion

| Approach | Problem |
|---|---|
| **Raw score addition** | BM25 scores (0-15) and cosine similarities (0-1) are on different scales. Direct addition is meaningless. |
| **Min-max normalization** | Sensitive to outliers. A single very high score distorts the range. |
| **Z-score normalization** | Assumes score distributions are Gaussian, which they are not. |
| **RRF** | Uses only rank positions, not raw scores. Naturally handles different scales. Robust to outliers. |

RRF's key insight: **rank is a universal currency**. Regardless of whether a retrieval method returns a score of 0.003 or 47.2, the rank (1st, 2nd, 3rd) is directly comparable across methods.

---

## 5. Reranking

Reranking is a second-stage process that takes the candidate set from initial retrieval and re-scores it using a more powerful (but slower) model.

### Why Rerank After Initial Retrieval?

Initial retrieval (BM25 + dense + fusion) must be **fast** because it operates over the entire corpus. It uses approximate methods (ANN, inverted index) that trade quality for speed.

Reranking operates on a **small candidate set** (typically 20-100 documents). At this scale, a more expensive model is feasible, and it can:
- Capture fine-grained query-document interactions that bi-encoders miss.
- Better understand the relationship between the full query and full document.
- Correct ranking errors from the initial retrieval.

```
Initial Retrieval: 1M documents -> top 50 candidates  (fast, approximate)
Reranking:          50 candidates -> top 5 results      (slow, precise)
```

### Cross-Encoder Reranking

Cross-encoder models take both the query and document as a single input and output a relevance score. Unlike bi-encoders (which embed query and document independently), cross-encoders can attend to every token in both inputs.

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Cross-encoder reranker for candidate documents."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 512):
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            query: Original user query.
            candidates: List of dicts with 'chunk_id', 'content', 'score'.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list with cross-encoder scores.
        """
        if not candidates:
            return []

        # Build query-document pairs
        pairs = [(query, c["content"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Return top-k with final ranking
        results = []
        for rank, item in enumerate(reranked[:top_k]):
            results.append({
                **item,
                "score": item["rerank_score"],
                "rank": rank + 1,
                "retrieval_method": "reranked"
            })

        return results
```

### Popular Reranking Models

| Model | Type | Quality | Speed | Notes |
|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | Cross-encoder | High | Medium | Multilingual, strong default |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | Good | Fast | English, very fast |
| `Cohere rerank-v3.5` | API | High | Fast (API) | Managed service, per-query cost |
| `jinaai/jina-reranker-v1-turbo-en` | Cross-encoder | Good | Fast | Optimized for speed |
| `llm-based` (GPT-4, Claude) | LLM | Highest | Slowest | See LLM reranking below |

### LLM-Based Reranking

For highest quality, use an LLM to score or rank documents:

```python
LLM_RERANK_PROMPT = """Given the following query and documents, rank them by relevance.

Query: {query}

Documents:
{documents}

Return a JSON list of document IDs ranked from most to least relevant.
Only return the JSON array, e.g., ["doc3", "doc1", "doc2"]
"""
```

**Tradeoffs**:
- Higher quality understanding of complex queries.
- 10-100x slower than cross-encoders.
- Per-query cost (API calls).
- Only practical when candidate set is very small (5-10 documents).

### Configuration

```yaml
reranking:
  enabled: true
  model: "BAAI/bge-reranker-v2-m3"
  max_length: 512
  top_k_candidates: 50       # How many candidates to feed to reranker
  top_k_final: 5             # How many results to return after reranking
  batch_size: 32             # Batch size for cross-encoder inference
  timeout_ms: 5000           # Timeout for reranking call
```

### Latency vs. Quality Tradeoffs

| Candidate Set Size | Reranking Latency (p50) | Quality Impact |
|---|---|---|
| 10 | ~20ms | Marginal improvement |
| 20 | ~40ms | Good improvement |
| 50 | ~100ms | Significant improvement |
| 100 | ~200ms | Diminishing returns |
| 200+ | ~500ms+ | Usually not worth it |

**Recommendation**: Rerank 20-50 candidates. Beyond that, the initial retrieval quality dominates and reranking provides diminishing returns.

---

## 6. Query Rewriting

Query rewriting transforms the user's raw query into a form that improves retrieval quality. Rewriting happens **before** the retrieval stage and is optional.

**Critical rule**: Always preserve the original query. Log both original and rewritten queries for evaluation.

### Query Expansion with LLM

Use an LLM to add relevant keywords or rephrase the query:

```python
EXPANSION_PROMPT = """You are a search query optimizer. Expand the following query
to improve document retrieval. Add relevant keywords and synonyms while
preserving the original intent.

Original query: {query}

Return ONLY the expanded query, nothing else."""

def expand_query_llm(query: str, llm_client) -> str:
    """Expand query using LLM."""
    response = llm_client.complete(EXPANSION_PROMPT.format(query=query))
    return response.strip()
```

### HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical answer to the query, then uses that hypothetical document's embedding for retrieval instead of the query embedding.

```python
HYPOTHESIS_PROMPT = """Write a short, factual paragraph that would answer this question.
Do not include the question itself. Just write the answer as if it were
a document snippet.

Question: {query}

Answer:"""

def hyde_retrieve(query: str, llm_client, vector_index, top_k: int = 10):
    """Retrieve using HyDE."""
    # Step 1: Generate hypothetical document
    hypothetical = llm_client.complete(HYPOTHESIS_PROMPT.format(query=query))

    # Step 2: Embed the hypothetical document
    # (use the same embedding model as the index)
    hypothesis_embedding = embed(hypothetical)

    # Step 3: Retrieve using the hypothetical embedding
    results = vector_index.search_by_vector(hypothesis_embedding, top_k=top_k)
    return results
```

**When HyDE helps**:
- Short, ambiguous queries.
- Conceptual queries where the embedding of the question is weak.
- Cross-lingual retrieval (LLM generates answer in target language).

**When HyDE hurts**:
- Factual queries where the LLM may hallucinate incorrect information.
- Latency-sensitive applications (adds LLM call latency).
- Queries about very recent events the LLM hasn't seen.

### Multi-Query Generation

Generate multiple related queries to increase recall:

```python
MULTI_QUERY_PROMPT = """Generate {n} alternative search queries for the following question.
Each query should approach the topic from a slightly different angle.

Original question: {query}

Return queries separated by newlines, one per line."""

def multi_query_retrieve(query: str, llm_client, retriever,
                         n_queries: int = 3, top_k: int = 5) -> list[dict]:
    """Retrieve using multiple query variations and merge."""
    response = llm_client.complete(
        MULTI_QUERY_PROMPT.format(query=query, n=n_queries)
    )
    queries = [query] + [q.strip() for q in response.strip().split("\n") if q.strip()]

    all_results = []
    for q in queries:
        results = retriever.retrieve(q, top_k=top_k)
        all_results.append(results)

    # Fuse all result sets with RRF
    fused = reciprocal_rank_fusion(all_results, k=60)
    return fused[:top_k]
```

### Step-Back Prompting

For complex queries, decompose into simpler sub-queries:

```python
STEP_BACK_PROMPT = """Given this complex question, break it into 2-4 simpler
sub-questions that together would help answer the original.

Complex question: {query}

Return sub-questions separated by newlines."""
```

### Query Rewriting Configuration

```yaml
query_rewriting:
  enabled: false                 # Disabled by default
  strategy: "expansion"          # expansion | hyde | multi_query | step_back

  expansion:
    model: "gpt-4o-mini"
    max_tokens: 100
    preserve_original: true

  hyde:
    model: "gpt-4o-mini"
    max_tokens: 200
    embedding_model: "BAAI/bge-large-en-v1.5"

  multi_query:
    model: "gpt-4o-mini"
    n_queries: 3
    fusion_method: "rrf"

  logging:
    log_original: true
    log_rewritten: true
    store_both: true
```

---

## 7. Context Construction

After retrieval and reranking, the final stage is assembling the retrieved chunks into a prompt-ready context for the LLM.

### Assembling Retrieved Chunks

```python
@dataclass
class ContextWindow:
    chunks: list[dict]          # Ordered list of chunks
    total_tokens: int           # Approximate token count
    sources: list[str]          # Unique source documents
    truncated: bool             # Whether chunks were dropped due to budget

def build_context(
    results: list[dict],
    max_tokens: int = 3000,
    tokenizer=None,             # For accurate token counting
    include_metadata: bool = True
) -> ContextWindow:
    """
    Assemble retrieved chunks into a context window.

    Respects token budget by adding chunks until budget is exhausted.
    """
    if tokenizer is None:
        # Rough approximation: 1 token ~= 4 characters
        def tokenizer(text: str) -> int:
            return len(text) // 4

    selected_chunks = []
    total_tokens = 0
    sources = set()
    truncated = False

    for result in results:
        chunk_tokens = tokenizer(result["content"])
        if total_tokens + chunk_tokens > max_tokens:
            truncated = True
            break

        selected_chunks.append(result)
        total_tokens += chunk_tokens
        sources.add(result.get("metadata", {}).get("source", "unknown"))

    return ContextWindow(
        chunks=selected_chunks,
        total_tokens=total_tokens,
        sources=list(sources),
        truncated=truncated
    )
```

### Token Budget Management

A practical budget allocation for a typical RAG prompt:

```
+-----------------------------------------------+
| System Prompt / Instructions    ~200 tokens   |
+-----------------------------------------------+
| Context (Retrieved Chunks)      ~3000 tokens  |
+-----------------------------------------------+
| User Query                     ~100 tokens    |
+-----------------------------------------------+
| Model Output Budget            ~500 tokens    |
+-----------------------------------------------+
| Total (for 4K context model)   ~3800 tokens   |
+-----------------------------------------------+
```

For models with larger context windows (8K, 32K, 128K), you can allocate more to context, but **more context is not always better**. LLMs can suffer from "lost in the middle" — ignoring information placed in the middle of long contexts.

### Chunk Ordering and Deduplication

```python
def order_chunks(
    chunks: list[dict],
    strategy: "relevance" | "source" | "chronological" = "relevance",
    deduplicate: bool = True
) -> list[dict]:
    """Order and deduplicate retrieved chunks."""

    # Deduplication by content hash
    if deduplicate:
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            content_hash = hash(chunk["content"][:200])  # First 200 chars
            if content_hash not in seen:
                seen.add(content_hash)
                unique_chunks.append(chunk)
        chunks = unique_chunks

    if strategy == "relevance":
        # Already ordered by relevance from retrieval — keep as-is
        return chunks

    elif strategy == "source":
        # Group by source document, maintain relevance within groups
        from collections import defaultdict
        by_source = defaultdict(list)
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            by_source[source].append(chunk)
        ordered = []
        for source_chunks in by_source.values():
            ordered.extend(source_chunks)
        return ordered

    elif strategy == "chronological":
        return sorted(
            chunks,
            key=lambda c: c.get("metadata", {}).get("created_at", ""),
            reverse=True
        )

    return chunks
```

### Prompt Template Design

```python
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question based on the
provided context. If the context does not contain enough information to
answer the question, say so clearly. Do not make up information.

## Context

{context}

## Question

{question}

## Answer

Provide a clear, concise answer. Cite your sources using [Source N] notation."""

def format_rag_prompt(
    context_window: ContextWindow,
    query: str,
    source_prefix: str = "Source"
) -> str:
    """Format the final RAG prompt with context and citations."""

    # Format context with source numbers
    context_parts = []
    for i, chunk in enumerate(context_window.chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_parts.append(
            f"[{source_prefix} {i}] ({source})\n{chunk['content']}"
        )

    context_str = "\n\n".join(context_parts)

    return RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=query
    )
```

### Citation and Source Attribution

Include metadata in each chunk to support citations:

```python
def enrich_chunk_for_citation(chunk: dict, chunk_index: int) -> str:
    """Format a chunk with citation metadata."""
    metadata = chunk.get("metadata", {})
    parts = [f"[Source {chunk_index + 1}]"]

    if "title" in metadata:
        parts.append(f"Title: {metadata['title']}")
    if "url" in metadata:
        parts.append(f"URL: {metadata['url']}")
    if "page" in metadata:
        parts.append(f"Page: {metadata['page']}")
    if "section" in metadata:
        parts.append(f"Section: {metadata['section']}")

    header = " | ".join(parts)
    return f"{header}\n{chunk['content']}"
```

### Handling Context Window Limits

Strategies when retrieved context exceeds the model's context window:

| Strategy | Description | Tradeoff |
|---|---|---|
| **Truncation** | Take top-K chunks that fit | Loses potentially relevant chunks |
| **Chunk summarization** | Summarize long chunks to fit | Loses detail |
| **Map-reduce** | Summarize chunks individually, then combine | Multiple LLM calls |
| **Context compression** | Use a small model to extract relevant sentences | Adds latency |
| **Sliding window** | Process in overlapping windows | Multiple calls, complex |

**Recommendation**: Start with simple truncation. It is fast, predictable, and often sufficient. Upgrade to compression or map-reduce only if evaluation shows truncation is losing critical information.

---

## 8. Retrieval Configuration

All retrieval parameters are centralized in a YAML configuration file.

### Configuration Schema

```yaml
# retrieval_config.yaml
# Complete retrieval pipeline configuration

# === Query Rewriting ===
query_rewriting:
  enabled: false
  strategy: "expansion"          # expansion | hyde | multi_query | step_back

  expansion:
    model: "gpt-4o-mini"
    max_tokens: 100
    temperature: 0.3

  hyde:
    model: "gpt-4o-mini"
    max_tokens: 200
    embedding_model: "BAAI/bge-large-en-v1.5"

  multi_query:
    model: "gpt-4o-mini"
    n_queries: 3
    temperature: 0.7

# === BM25 Retrieval ===
bm25:
  enabled: true
  index_type: "in_memory"        # in_memory | opensearch | elasticsearch
  analyzer: "standard"
  language: "en"
  k1: 1.5                       # Term frequency saturation
  b: 0.75                       # Length normalization
  top_k: 20                     # Candidates to retrieve
  field_weights:
    title: 2.0                  # Boost title matches
    headings: 1.5               # Boost heading matches
    body: 1.0                   # Body text weight
  stop_words: null               # null = use language default
  enable_stemming: false

# === Dense Vector Retrieval ===
dense:
  enabled: true
  embedding_model: "BAAI/bge-large-en-v1.5"
  dimension: 1024
  distance_metric: "cosine"     # cosine | dot_product | l2
  index_type: "hnsw"            # hnsw | ivf | ivf_pq
  top_k: 20                     # Candidates to retrieve
  similarity_threshold: 0.0     # Minimum similarity score

  hnsw:
    M: 32                       # Max connections per node
    ef_construction: 200        # Build-time search width
    ef_search: 128              # Query-time search width

  vector_store: "opensearch"    # opensearch | faiss | qdrant | chroma | pgvector
  collection: "documents_dense"

# === Hybrid Retrieval ===
hybrid:
  enabled: true
  fusion_method: "rrf"          # rrf | weighted_score
  rrf_k: 60                     # RRF smoothing parameter
  weights:
    bm25: 1.0
    dense: 1.0
  max_candidates: 50            # Max candidates after fusion

# === Reranking ===
reranking:
  enabled: true
  model: "BAAI/bge-reranker-v2-m3"
  model_type: "cross_encoder"   # cross_encoder | llm | api
  max_length: 512
  top_k_candidates: 50          # Input to reranker
  top_k_final: 5               # Output from reranker
  batch_size: 32
  timeout_ms: 5000

  # LLM reranking (if model_type = "llm")
  llm:
    model: "gpt-4o-mini"
    max_tokens: 100
    temperature: 0.0

  # API reranking (if model_type = "api")
  api:
    provider: "cohere"
    api_key_env: "COHERE_API_KEY"

# === Context Construction ===
context:
  max_tokens: 3000              # Token budget for context
  chunk_ordering: "relevance"   # relevance | source | chronological
  deduplication: true
  include_metadata: true
  source_format: "bracketed"    # bracketed | footnote | inline

  prompt_template: "default"    # Reference to prompt template
  prompt_templates:
    default: |
      You are a helpful assistant. Answer based on the provided context.
      If the context does not contain enough information, say so.
      Cite sources using [Source N] notation.

      ## Context
      {context}

      ## Question
      {question}

      ## Answer

# === Evaluation ===
evaluation:
  enabled: true
  metrics:
    - recall_at_k
    - hit_rate_at_k
    - mrr
    - ndcg
  k_values: [1, 3, 5, 10]
  log_retrieval_results: true
  benchmark_queries: "eval/queries.jsonl"
```

### Tunable Parameters Summary

| Stage | Parameter | Default | Impact |
|---|---|---|---|
| **BM25** | `k1` | 1.5 | Higher = less TF saturation |
| **BM25** | `b` | 0.75 | Higher = more length penalty |
| **BM25** | `top_k` | 20 | More candidates = better recall, slower |
| **Dense** | `ef_search` | 128 | Higher = better recall, slower |
| **Dense** | `similarity_threshold` | 0.0 | Raise to filter low-quality matches |
| **Hybrid** | `rrf_k` | 60 | Higher = flatter rank distribution |
| **Hybrid** | `weights` | [1.0, 1.0] | Shift balance between BM25/dense |
| **Reranker** | `top_k_candidates` | 50 | More candidates = better final quality |
| **Reranker** | `top_k_final` | 5 | Final output count |
| **Context** | `max_tokens` | 3000 | More = more context, but may lose focus |

### Default vs. Production Configurations

| Parameter | Default (Development) | Production |
|---|---|---|
| BM25 `top_k` | 10 | 20 |
| Dense `top_k` | 10 | 20 |
| RRF candidates | 20 | 50 |
| Reranker candidates | 20 | 50 |
| Final `top_k` | 3 | 5 |
| Context tokens | 2000 | 3000 |
| Query rewriting | Off | On (based on eval) |
| Reranking | Off | On |

Start with defaults during development. Tune based on evaluation metrics before enabling production settings.

---

## Summary: Complete Retrieval Pipeline

```
User Query
    |
    v
[Optional] Query Rewriting
    |  - LLM expansion, HyDE, multi-query
    |  - Preserve original query
    |
    +----> BM25 Retrieval (top 20)
    |         |
    +----> Dense Vector Retrieval (top 20)
    |         |
    v         v
    Reciprocal Rank Fusion (k=60)
    |  - Merge results by rank
    |  - Deduplicate documents
    |  - ~50 candidates
    |
    v
Cross-Encoder Reranker
    |  - Re-score top 50 candidates
    |  - Return top 5
    |
    v
Context Construction
    |  - Order chunks by relevance
    |  - Deduplicate
    |  - Fit within token budget
    |  - Format with citations
    |
    v
Final Context -> LLM
```

### Evaluation Metrics

The retrieval system should be benchmarked with:

- **Recall@K**: What fraction of relevant documents appear in top-K?
- **Hit Rate@K**: What fraction of queries have at least one relevant document in top-K?
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result.
- **NDCG@K**: Normalized Discounted Cumulative Gain — accounts for ranking order.
- **Latency**: End-to-end retrieval time (p50, p95, p99).

Benchmark all combinations:
1. BM25 only
2. Dense only
3. Hybrid (RRF)
4. Hybrid + Reranker

Do not claim hybrid is better without evaluation evidence on your specific data.
