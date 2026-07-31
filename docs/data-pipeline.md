# Data Pipeline Design

This document defines the design and behavior of the ingestion, parsing, chunking, embedding, and indexing pipeline for the RAG system.

---

## Table of Contents

1. [Document Ingestion](#1-document-ingestion)
2. [Document Parsing](#2-document-parsing)
3. [Semantic Chunking](#3-semantic-chunking)
4. [Metadata Extraction](#4-metadata-extraction)
5. [Embedding Generation](#5-embedding-generation)
6. [Indexing Pipeline](#6-indexing-pipeline)
7. [Pipeline Orchestration](#7-pipeline-orchestration)

---

## 1. Document Ingestion

### 1.1 Supported Input Formats

| Format   | MIME Type                  | Parser            | Notes                                     |
| -------- | -------------------------- | ----------------- | ----------------------------------------- |
| PDF      | `application/pdf`          | pdfplumber        | Primary. PyPDF2 fallback for scanning.    |
| DOCX     | `application/vnd.openxml...` | python-docx     | Tables and styles extracted separately.   |
| TXT      | `text/plain`               | Plain read        | UTF-8 with BOM handling.                  |
| Markdown | `text/markdown`            | Markdown parser   | Preserves heading structure natively.     |
| HTML     | `text/html`                | BeautifulSoup     | Strips scripts/styles, extracts content.  |
| CSV      | `text/csv`                 | pandas / csv      | Each row becomes a candidate chunk.       |

### 1.2 File Validation

Every ingested file passes through a validation gate before entering the pipeline:

```
Raw File
  │
  ├─ File exists?          → ERROR: FileNotFoundError
  ├─ Readable?             → ERROR: PermissionError
  ├─ Non-zero size?        → ERROR: EmptyFileError
  ├─ MIME type supported?  → ERROR: UnsupportedFormatError
  ├─ Under size limit?     → ERROR: FileTooLargeError (default: 100MB)
  └─ Passes checksum?     → SKIP: DuplicateFileError (already processed)
```

**Validation rules:**

```python
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
}

@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None
    file_hash: str | None = None

def validate_file(path: Path) -> ValidationResult:
    if not path.exists():
        return ValidationResult(valid=False, error=f"File not found: {path}")
    if not path.is_file():
        return ValidationResult(valid=False, error=f"Not a file: {path}")
    if path.stat().st_size == 0:
        return ValidationResult(valid=False, error="File is empty")
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return ValidationResult(valid=False, error="File exceeds size limit")

    import mimetypes
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in SUPPORTED_MIME_TYPES:
        return ValidationResult(valid=False, error=f"Unsupported type: {mime_type}")

    import hashlib
    content = path.read_bytes()
    file_hash = hashlib.sha256(content).hexdigest()
    return ValidationResult(valid=True, file_hash=file_hash)
```

### 1.3 Batch vs. Single Document Ingestion

**Single document ingestion** processes one file through the full pipeline. Used for API-driven, on-demand ingestion.

**Batch ingestion** processes a directory or list of files. Used for bulk imports and initial data loading.

```python
from pathlib import Path
from typing import Iterator

def discover_files(directory: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield all ingestible files under a directory."""
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".html", ".csv"}:
            yield path

def batch_ingest(directory: Path, config: PipelineConfig) -> BatchResult:
    """Ingest all files in a directory."""
    results = []
    for file_path in discover_files(directory):
        result = ingest_single(file_path, config)
        results.append(result)
    return BatchResult(results=results)
```

**Batch processing considerations:**

- Process files sequentially to avoid memory exhaustion from large documents.
- Track progress with a simple counter or progress bar.
- Allow resuming: skip files whose `file_hash` already exists in the processing log.
- Emit structured logs per file for observability.

### 1.4 Directory Watching / Incremental Ingestion

For continuous ingestion, a file watcher monitors a directory and triggers the pipeline for new or modified files.

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, config: PipelineConfig):
        self.config = config

    def on_created(self, event):
        if not event.is_directory:
            self._process(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._process(Path(event.src_path))

    def _process(self, path: Path):
        validation = validate_file(path)
        if validation.valid:
            ingest_single(path, self.config)

def start_watcher(directory: Path, config: PipelineConfig):
    handler = IngestionHandler(config)
    observer = Observer()
    observer.schedule(handler, str(directory), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

**Deduplication strategy:**

- Compute SHA-256 hash of file content at ingestion time.
- Store `(file_hash, source_path, ingested_at)` in a `processed_files` table.
- Before processing, check if `file_hash` already exists. If so, skip.
- When a file is modified, its hash changes, so re-processing happens naturally.


### 1.5 URL Ingestion (Firecrawl)

For scraping documentation from web applications, we use **Firecrawl** — a managed web scraping API that handles JavaScript rendering, anti-bot protection, and content extraction.

**Why Firecrawl over raw httpx/BeautifulSoup:**
- JavaScript-rendered pages (SPAs, dynamic docs)
- Built-in anti-bot bypasses
- Returns clean markdown from any page
- Managed crawling with rate limiting
- Handles redirects, timeouts, retries automatically

**Two modes:**

| Mode | Function | Use case |
|---|---|---|
| **Single page** | `fetch_url(url)` | Scrape one docs page |
| **Crawl** | `crawl_site(url, limit)` | Scrape entire docs site |

**Flow:**

```
URL (http://...)
    │
    ▼
┌───────────────────┐
│  Firecrawl API    │  scrape_url() or crawl_url()
│  (managed)        │  Returns clean markdown
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  FetchedContent   │  content (markdown), metadata, source_url
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Same pipeline    │  clean → chunk → embed → index
└───────────────────┘
```

**Implementation:**

```python
from firecrawl import FirecrawlApp
from dataclasses import dataclass

@dataclass
class FetchedContent:
    content: str          # Markdown from Firecrawl
    metadata: dict        # Title, description, etc.
    source_url: str       # Original URL
    success: bool

class URLFetcher:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def fetch_url(self, url: str) -> FetchedContent:
        """Scrape a single URL → markdown content."""
        result = self.app.scrape_url(url, params={"formats": ["markdown"]})
        return FetchedContent(
            content=result.get("markdown", ""),
            metadata=result.get("metadata", {}),
            source_url=url,
            success=result.get("success", False),
        )

    def crawl_site(self, url: str, limit: int = 100) -> list[FetchedContent]:
        """Crawl a site starting from url, up to limit pages."""
        crawl_result = self.app.crawl_url(
            url,
            params={"limit": limit, "formats": ["markdown"]},
        )
        return [
            FetchedContent(
                content=page.get("markdown", ""),
                metadata=page.get("metadata", {}),
                source_url=page.get("metadata", {}).get("sourceURL", url),
                success=True,
            )
            for page in crawl_result.get("data", [])
        ]
```

**Deduplication for URLs:**
- Compute SHA-256 of fetched content (not the URL itself).
- Store `(content_hash, source_url, fetched_at)` in `processed_urls` table.
- Skip re-fetching if content hash unchanged.

**Configuration:**

```yaml
# config/defaults.yaml
firecrawl:
  api_key: ""  # Set via FIRECRAWL_API_KEY env var
  timeout: 30
  crawl_limit: 100
```

---

## 2. Document Parsing

### 2.1 Format-Specific Parsers

Each format has a dedicated parser. All parsers implement the same interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ParsedDocument:
    content: str                # Full extracted text
    metadata: dict              # Format-specific metadata
    tables: list[dict]          # Extracted tables (if any)
    sections: list["Section"]   # Structural sections (if any)

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        ...
```

#### PDF Parser

Primary: `pdfplumber`. Fallback: `PyPDF2` for scanned/image-based PDFs.

```python
import pdfplumber

class PDFParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        text_parts = []
        tables = []
        metadata = {}

        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)

                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append({
                        "page": i + 1,
                        "data": table,
                        "as_text": self._table_to_text(table),
                    })

        return ParsedDocument(
            content="\n\n".join(text_parts),
            metadata=metadata,
            tables=tables,
            sections=[],  # Sections inferred during chunking
        )

    def _table_to_text(self, table: list[list]) -> str:
        """Convert a table to a readable text representation."""
        if not table:
            return ""
        headers = table[0]
        rows = table[1:]
        lines = [" | ".join(str(cell or "") for cell in headers)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(cell or "") for cell in row))
        return "\n".join(lines)
```

#### DOCX Parser

```python
from docx import Document as DocxDocument

class DOCXParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        doc = DocxDocument(str(file_path))
        text_parts = []
        tables = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        for i, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append([cell.text for cell in row.cells])
            tables.append({
                "index": i,
                "data": rows,
                "as_text": self._table_to_text(rows),
            })

        return ParsedDocument(
            content="\n\n".join(text_parts),
            metadata={"title": doc.core_properties.title or ""},
            tables=tables,
            sections=[],
        )

    def _table_to_text(self, table: list[list]) -> str:
        if not table:
            return ""
        headers = table[0]
        rows = table[1:]
        lines = [" | ".join(headers)]
        for row in rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)
```

#### Markdown Parser

Markdown is parsed structurally to preserve heading hierarchy:

```python
import re

class MarkdownParser(BaseParser):
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse(self, file_path: Path) -> ParsedDocument:
        content = file_path.read_text(encoding="utf-8")

        # Extract headings for structural metadata
        sections = []
        for match in self.HEADING_PATTERN.finditer(content):
            level = len(match.group(1))
            title = match.group(2)
            sections.append(Section(level=level, title=title, offset=match.start()))

        return ParsedDocument(
            content=content,
            metadata={"format": "markdown"},
            tables=[],
            sections=sections,
        )
```

#### HTML Parser

```python
from bs4 import BeautifulSoup

class HTMLParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        title_tag = soup.find("title")
        title = title_tag.get_text() if title_tag else ""

        return ParsedDocument(
            content=text,
            metadata={"title": title, "format": "html"},
            tables=[],
            sections=[],
        )
```

#### CSV Parser

```python
import csv

class CSVParser(BaseParser):
    def parse(self, file_path: Path) -> ParsedDocument:
        rows = []
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert each row to a text block
                text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                rows.append(text)

        return ParsedDocument(
            content="\n\n".join(rows),
            metadata={"format": "csv", "row_count": len(rows)},
            tables=[],
            sections=[],
        )
```

### 2.2 Handling Tables, Images, and Structured Content

**Tables** are extracted as structured data AND as text representations. The text representation is used for chunking and embedding. The structured data is stored as metadata for retrieval context.

**Images** are not embedded directly. Two strategies:

1. **Caption extraction** — if the document provides alt text or captions, these are included as text content.
2. **OCR extraction** — for PDFs with scanned images, use `pytesseract` as a fallback path.

```python
class OCRFallbackParser:
    """Extract text from image-heavy PDFs using OCR."""
    def extract(self, pdf_path: Path) -> str:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        text_parts = []
        for img in images:
            text = pytesseract.image_to_string(img)
            if text.strip():
                text_parts.append(text)
        return "\n\n".join(text_parts)
```

**Structured content** (code blocks, lists, blockquotes) is preserved as-is in the extracted text. Markdown and HTML parsers maintain these structures.

### 2.3 Fallback Strategies

```
Parse Attempt
  │
  ├─ Primary parser succeeds  → Return ParsedDocument
  │
  ├─ Primary parser fails     → Try secondary parser (if available)
  │     ├─ Secondary succeeds → Return ParsedDocument with warning
  │     └─ Secondary fails    → Try OCR fallback (PDF only)
  │           ├─ OCR succeeds → Return ParsedDocument with warning
  │           └─ OCR fails    → Log error, return partial result or skip
  │
  └─ Unsupported format       → Log error, skip file
```

Every fallback emits a structured warning log with the file path, error type, and which fallback was used. This enables monitoring of data quality issues.

---

## 3. Semantic Chunking

### 3.1 Why Semantic Chunking Over Fixed-Size Chunking

Fixed-size chunking (e.g., every 512 tokens) has fundamental problems:

- **Splits mid-sentence or mid-paragraph**, destroying semantic coherence.
- **Ignores document structure** — a heading might end up in a different chunk than its content.
- **Produces retrieval noise** — chunks that contain unrelated fragments confuse the retriever.

Semantic chunking respects the natural boundaries of the document:

- Paragraphs
- Section headings
- List items
- Code blocks
- Table boundaries

This produces chunks that are internally coherent and retrievable as self-contained units of meaning.

### 3.2 Chunking Strategy

The chunker operates on the structured output of the parser:

```
ParsedDocument
  │
  ├─ Split into semantic blocks (paragraphs, headings, lists, etc.)
  │
  ├─ Assign section context to each block (nearest heading + level)
  │
  ├─ Merge small blocks up to target chunk size
  │
  ├─ Split oversized blocks at sentence boundaries
  │
  ├─ Apply overlap between adjacent chunks
  │
  └─ Validate: min size, max size, not empty
```

**Block detection rules:**

```python
import re

def split_into_blocks(text: str) -> list[str]:
    """Split text into semantic blocks."""
    # Split on double newlines (paragraph boundaries)
    blocks = re.split(r"\n\s*\n", text)
    # Filter empty blocks
    return [b.strip() for b in blocks if b.strip()]
```

### 3.3 Configurable Parameters

```python
from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    target_chunk_size: int = 512       # Target tokens per chunk
    max_chunk_size: int = 768          # Hard maximum before forced split
    min_chunk_size: int = 50           # Minimum tokens; merge if smaller
    overlap_tokens: int = 50           # Overlap between adjacent chunks
    respect_sentence_boundaries: bool = True
    merge_small_blocks: bool = True
```

### 3.4 Chunking Algorithm

```python
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    chunk_index: int
    token_count: int
    section_heading: str | None
    section_level: int | None

@dataclass
class ChunkingResult:
    chunks: list[Chunk]
    warnings: list[str]

def chunk_document(
    parsed: ParsedDocument,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> ChunkingResult:
    blocks = split_into_blocks(parsed.content)
    if not blocks:
        return ChunkingResult(chunks=[], warnings=["Empty document"])

    chunks: list[Chunk] = []
    warnings: list[str] = []
    current_section: str | None = None
    current_level: int | None = None
    pending_text = ""

    for block in blocks:
        # Detect headings and update section context
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading_match:
            current_level = len(heading_match.group(1))
            current_section = heading_match.group(2)

        # Merge small blocks
        candidate = (pending_text + "\n\n" + block).strip() if pending_text else block
        candidate_tokens = token_counter(candidate)

        if candidate_tokens < config.min_chunk_size:
            pending_text = candidate
            continue

        # If block is too large, split at sentence boundaries
        if candidate_tokens > config.max_chunk_size:
            sub_chunks = _split_at_sentences(candidate, config, token_counter)
            for sc in sub_chunks:
                chunks.append(Chunk(
                    text=sc,
                    chunk_index=len(chunks),
                    token_count=token_counter(sc),
                    section_heading=current_section,
                    section_level=current_level,
                ))
            pending_text = ""
            continue

        # Normal case: emit chunk
        chunks.append(Chunk(
            text=candidate,
            chunk_index=len(chunks),
            token_count=candidate_tokens,
            section_heading=current_section,
            section_level=current_level,
        ))
        pending_text = ""

    # Handle remaining pending text
    if pending_text:
        if token_counter(pending_text) < config.min_chunk_size and chunks:
            # Merge with last chunk
            last = chunks[-1]
            last.text = last.text + "\n\n" + pending_text
            last.token_count = token_counter(last.text)
        else:
            chunks.append(Chunk(
                text=pending_text,
                chunk_index=len(chunks),
                token_count=token_counter(pending_text),
                section_heading=current_section,
                section_level=current_level,
            ))

    # Apply overlap (add trailing context from previous chunk)
    if config.overlap_tokens > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, config, token_counter)

    return ChunkingResult(chunks=chunks, warnings=warnings)


def _split_at_sentences(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> list[str]:
    """Split oversized text at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if token_counter(candidate) > config.max_chunk_size and current:
            result.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def _apply_overlap(
    chunks: list[Chunk],
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Add trailing overlap from each chunk to the next."""
    for i in range(1, len(chunks)):
        prev_tokens = chunks[i - 1].text.split()
        overlap_text = " ".join(prev_tokens[-config.overlap_tokens:])
        chunks[i].text = overlap_text + " " + chunks[i].text
        chunks[i].token_count = token_counter(chunks[i].text)
    return chunks
```

### 3.5 Metadata Propagation

Every chunk inherits metadata from its parent document and its position within the document. See [Metadata Extraction](#4-metadata-extraction) for the full schema.

---

## 4. Metadata Extraction

### 4.1 Automatic Metadata

Extracted automatically from the file and parsing results:

| Field          | Type     | Source                              |
| -------------- | -------- | ----------------------------------- |
| `document_id`  | `str`    | SHA-256 of file content             |
| `chunk_id`     | `str`    | SHA-256 of `document_id + chunk_index` |
| `source`       | `str`    | Original file path                  |
| `title`        | `str`    | Document title (from metadata/first heading) |
| `document_type`| `str`    | File extension / MIME type          |
| `file_hash`    | `str`    | SHA-256 of file content             |
| `content_type` | `str`    | MIME type                           |
| `file_size`    | `int`    | File size in bytes                  |
| `created_at`   | `datetime` | Ingestion timestamp               |
| `parsed_at`    | `datetime` | Parse completion timestamp        |

### 4.2 Structural Metadata

Extracted from document structure:

| Field              | Type     | Source                              |
| ------------------ | -------- | ----------------------------------- |
| `chunk_index`      | `int`    | Position within document chunks     |
| `total_chunks`     | `int`    | Total chunks in document            |
| `section_heading`  | `str`    | Nearest preceding heading           |
| `section_level`    | `int`    | Heading level (1-6)                 |
| `page_number`      | `int`    | Page number (PDF, DOCX only)        |
| `heading_path`     | `list[str]` | Full heading hierarchy            |
| `token_count`      | `int`    | Number of tokens in chunk           |
| `char_count`       | `int`    | Number of characters in chunk       |

### 4.3 Custom Metadata

Users can attach custom metadata at ingestion time:

```python
@dataclass
class IngestionRequest:
    file_path: Path
    custom_metadata: dict | None = None  # e.g., {"category": "legal", "tags": ["contract", "2024"]}
    source_id: str | None = None         # External ID to link back to
    collection: str | None = None        # Target collection/index name
```

Custom metadata is merged into the chunk metadata and indexed as keyword fields in OpenSearch for filtering.

### 4.4 Metadata Storage Schema

**PostgreSQL schema for document tracking:**

```sql
CREATE TABLE documents (
    document_id     TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    file_hash       TEXT NOT NULL UNIQUE,
    title           TEXT,
    document_type   TEXT NOT NULL,
    file_size       INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL,
    custom_metadata JSONB,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parsed_at       TIMESTAMPTZ,
    indexed_at      TIMESTAMPTZ
);

CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(document_id),
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    section_heading TEXT,
    section_level   INTEGER,
    heading_path    TEXT[],
    page_number     INTEGER,
    token_count     INTEGER NOT NULL,
    char_count      INTEGER NOT NULL,
    embedding_model TEXT,
    embedding_dim   INTEGER,
    custom_metadata JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);
```

**OpenSearch document mapping (per chunk):**

```json
{
  "mappings": {
    "properties": {
      "chunk_id":       { "type": "keyword" },
      "document_id":    { "type": "keyword" },
      "content":        { "type": "text" },
      "section_heading": { "type": "text" },
      "document_type":  { "type": "keyword" },
      "source":         { "type": "keyword" },
      "title":          { "type": "text" },
      "page_number":    { "type": "integer" },
      "chunk_index":    { "type": "integer" },
      "token_count":    { "type": "integer" },
      "created_at":     { "type": "date" },
      "embedding":      { "type": "dense_vector", "dims": 384, "index": true },
      "custom_metadata": { "type": "object", "enabled": true }
    }
  }
}
```

---

## 5. Embedding Generation

### 5.1 Model Selection Strategy

The embedding model is configurable. The pipeline supports three backends:

| Backend           | Model Example                    | Dimensions | Use Case                        |
| ----------------- | -------------------------------- | ---------- | ------------------------------- |
| sentence-transformers | `all-MiniLM-L6-v2`            | 384        | Local, fast, default            |
| sentence-transformers | `multi-qa-MiniLM-L6-cos-v1`  | 384        | Multilingual, retrieval-tuned   |
| OpenAI            | `text-embedding-3-small`         | 1536       | High quality, API-based         |
| OpenAI            | `text-embedding-3-large`         | 3072       | Highest quality, API-based      |

**Selection criteria:**

- **Latency**: Local models for low-latency pipelines.
- **Quality**: API models for highest retrieval quality.
- **Cost**: Local models have zero marginal cost.
- **Language**: Multilingual models for non-English content.

### 5.2 Embedding Interface

```python
from abc import ABC, abstractmethod

class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
```

**Sentence-transformers implementation:**

```python
from sentence_transformers import SentenceTransformer

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
```

**OpenAI implementation:**

```python
import openai

class OpenAIBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        # OpenAI dimensions depend on model
        return {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}[self._model_name]

    @property
    def model_name(self) -> str:
        return self._model_name
```

### 5.3 Batch Embedding

Embedding is done in batches to maximize throughput and respect API rate limits:

```python
from itertools import islice

def embed_chunks(
    chunks: list[Chunk],
    backend: EmbeddingBackend,
    batch_size: int = 64,
) -> list[tuple[Chunk, list[float]]]:
    """Embed all chunks in batches."""
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = backend.embed(texts)
        for chunk, embedding in zip(batch, embeddings):
            results.append((chunk, embedding))
    return results
```

**Batch size tuning:**

- Local models: batch size 64-256 depending on GPU memory.
- API models: batch size 32-2048 depending on rate limits and payload size.
- OpenAI limit: 2048 texts per request, 150K tokens total per request.

### 5.4 Embedding Caching

To avoid re-computing embeddings for unchanged chunks, use a content-addressed cache:

```python
import json
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str, model: str) -> str:
        import hashlib
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        key = self._key(text, model)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        return None

    def put(self, text: str, model: str, embedding: list[float]):
        key = self._key(text, model)
        cache_path = self.cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps(embedding))
```

Cache is checked before embedding. On cache hit, the stored embedding is used directly. On cache miss, the embedding is computed and stored.

### 5.5 Model Configuration

```yaml
# config/embedding.yaml
embedding:
  backend: sentence-transformers
  model: all-MiniLM-L6-v2
  batch_size: 64
  normalize: true
  cache_dir: .cache/embeddings
  fallback_model: null  # Optional: second model if primary fails
```

Model switching is supported: changing the `model` field invalidates the cache for the old model (cache keys include model name). The `embedding_model` field in chunk metadata tracks which model produced each embedding.

---

## 6. Indexing Pipeline

### 6.1 Vector Store Indexing

The pipeline supports multiple vector store backends:

| Backend     | Use Case                        | Persistence     |
| ----------- | ------------------------------- | --------------- |
| FAISS       | High-performance local search   | File-based      |
| Chroma      | Development / lightweight       | File-based      |
| Qdrant      | Production with filtering       | Server-based    |
| pgvector    | PostgreSQL-native, simple setup | Database-based  |
| OpenSearch  | Hybrid vector + BM25 (primary) | Server-based    |

**For this project, OpenSearch is the primary vector store.**

### 6.2 BM25 Index Construction

BM25 indexing is handled by OpenSearch's built-in analyzer. The `content` field is indexed with standard text analysis:

```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "content_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "stop", "snowball"]
        }
      }
    }
  }
}
```

For local/offline BM25 without OpenSearch:

```python
from rank_bm25 import BM25Okapi
import jieba

class BM25Index:
    def __init__(self):
        self.corpus: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.index: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def add_documents(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids.extend(chunk_ids)
        self.corpus.extend(texts)
        self.tokenized_corpus.extend([list(jieba.cut(t)) for t in texts])
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokenized_query = list(jieba.cut(query))
        scores = self.index.get_scores(tokenized_query)
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

### 6.3 Dual-Index Maintenance

Both vector and BM25 indexes are maintained in parallel:

```
Chunk
  │
  ├─ → OpenSearch index (vector field)    → Dense vector retrieval
  ├─ → OpenSearch index (text fields)     → BM25 retrieval
  └─ → PostgreSQL (metadata + content)    → Application metadata
```

Hybrid search combines results from both indexes using RRF (Reciprocal Rank Fusion) or a learned combiner at query time. The indexing pipeline does not handle ranking — it only ensures both indexes are populated.

### 6.4 Incremental Indexing vs. Full Re-index

**Incremental indexing** (default):

- Only process new or modified files.
- Check `file_hash` in `processed_files` table before parsing.
- Update chunks in place if document content changed.
- Delete orphaned chunks when a document is removed.

```python
def incremental_index(file_path: Path, config: PipelineConfig):
    validation = validate_file(file_path)
    if not validation.valid:
        raise ValidationError(validation.error)

    existing = db.get_document_by_hash(validation.file_hash)
    if existing:
        logger.info(f"Skipping unchanged file: {file_path}")
        return

    # Process and index
    parsed = parse(file_path, config.parser)
    chunks = chunk(parsed, config.chunking)
    embeddings = embed(chunks, config.embedding)
    index_chunks(embeddings, config.index)
    db.upsert_document(document_id, file_path, validation.file_hash, chunk_count=len(chunks))
```

**Full re-index:**

- Reprocesses all documents from scratch.
- Used when changing embedding models, chunking strategy, or parser configuration.
- Builds a new index, then swaps atomically.

```python
def full_reindex(directory: Path, config: PipelineConfig):
    # Build new index in a temporary collection
    temp_index = f"{config.index.name}_reindex_{int(time.time())}"
    for file_path in discover_files(directory):
        parsed = parse(file_path, config.parser)
        chunks = chunk(parsed, config.chunking)
        embeddings = embed(chunks, config.embedding)
        index_chunks(embeddings, config.index.with_name(temp_index))

    # Atomic swap
    alias_swap(config.index.name, temp_index)
    drop_index(old_index_name)
```

### 6.5 Index Persistence and Versioning

```yaml
# config/indexing.yaml
index:
  name: rag_documents
  vector_field: embedding
  text_field: content
  id_field: chunk_id
  versioning:
    enabled: true
    max_versions: 3  # Keep last 3 index versions for rollback
  persistence:
    backup_enabled: true
    backup_dir: .backups/indices
```

---

## 7. Pipeline Orchestration

### 7.1 End-to-End Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  INGEST  │───▶│  PARSE   │───▶│  CHUNK   │───▶│  EMBED   │───▶│  INDEX   │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │               │
       ▼               ▼               ▼               ▼               ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Validate │    │ Extract  │    │ Semantic │    │ Batch    │    │ OpenSearch│
  │ Dedupe   │    │ Tables   │    │ Split    │    │ Cache    │    │ + BM25   │
  │ Hash     │    │ OCR      │    │ Metadata │    │ Model    │    │ + Vector │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

  STATUS TRACKING:
  ─────────────────
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Pipeline Run ID: abc-123                                               │
  │                                                                         │
  │ [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 42% complete       │
  │                                                                         │
  │ Ingested:  12/30 files                                                 │
  │ Parsed:    10/30 files                                                 │
  │ Chunked:   247 chunks                                                  │
  │ Embedded:  200/247 chunks                                              │
  │ Indexed:   150/247 chunks                                              │
  │                                                                         │
  │ Errors: 1 (file.pdf - unsupported encoding)                            │
  │ Warnings: 3 (OCR fallback used for scan_*.pdf)                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Error Handling and Retry Logic

Each pipeline stage has distinct failure modes and recovery strategies:

```
Stage          │ Failure Mode              │ Strategy
───────────────┼───────────────────────────┼──────────────────────────────
Ingestion      │ File not found            │ Log error, skip, continue
               │ Permission denied         │ Log error, skip, continue
               │ Duplicate file            │ Skip (idempotent)
───────────────┼───────────────────────────┼──────────────────────────────
Parsing        │ Corrupted file            │ Try fallback parser, log warning
               │ Encoding error            │ Try latin-1 fallback, log warning
               │ Password-protected PDF    │ Log error, skip file
───────────────┼───────────────────────────┼──────────────────────────────
Chunking       │ Empty document            │ Log warning, skip
               │ Exceeds max chunks        │ Log warning, truncate or split
───────────────┼───────────────────────────┼──────────────────────────────
Embedding      │ API rate limit            │ Exponential backoff, retry 3x
               │ API timeout               │ Retry 3x with backoff
               │ Model not loaded          │ Log error, fail pipeline
───────────────┼───────────────────────────┼──────────────────────────────
Indexing       │ OpenSearch down           │ Retry with backoff, alert
               │ Mapping conflict          │ Log error, fail pipeline
               │ Document too large        │ Log error, skip chunk
```

**Retry implementation:**

```python
import time
import functools

def retry(max_attempts: int = 3, backoff_base: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    wait = backoff_base * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            raise last_error
        return wrapper
    return decorator

@retry(max_attempts=3, backoff_base=1.0)
def embed_batch(texts: list[str], backend: EmbeddingBackend) -> list[list[float]]:
    return backend.embed(texts)
```

### 7.3 Progress Tracking and Observability

**Structured logging at every stage:**

```python
import logging

logger = logging.getLogger("rag_pipeline")

def log_stage_start(stage: str, run_id: str, **context):
    logger.info(
        "Pipeline stage started",
        extra={
            "event": "stage_start",
            "stage": stage,
            "run_id": run_id,
            **context,
        },
    )

def log_stage_complete(stage: str, run_id: str, duration_ms: float, **context):
    logger.info(
        "Pipeline stage completed",
        extra={
            "event": "stage_complete",
            "stage": stage,
            "run_id": run_id,
            "duration_ms": duration_ms,
            **context,
        },
    )

def log_error(stage: str, run_id: str, error: Exception, **context):
    logger.error(
        "Pipeline stage failed",
        extra={
            "event": "stage_error",
            "stage": stage,
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context,
        },
        exc_info=True,
    )
```

**Prometheus metrics:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
chunks_embedded = Counter("rag_chunks_embedded_total", "Total chunks embedded", ["model"])
chunks_indexed = Counter("rag_chunks_indexed_total", "Total chunks indexed")
pipeline_errors = Counter("rag_pipeline_errors_total", "Total pipeline errors", ["stage", "error_type"])
files_processed = Counter("rag_files_processed_total", "Total files processed", ["status"])

# Histograms
embed_duration = Histogram("rag_embed_duration_seconds", "Embedding duration", ["model"], buckets=[0.1, 0.5, 1, 2, 5, 10])
parse_duration = Histogram("rag_parse_duration_seconds", "Parse duration", ["format"], buckets=[0.01, 0.05, 0.1, 0.5, 1, 5])

# Gauges
pipeline_progress = Gauge("rag_pipeline_progress", "Pipeline progress percentage", ["run_id"])
chunks_pending = Gauge("rag_chunks_pending", "Chunks pending embedding")
```

**Grafana dashboard panels:**

- Pipeline throughput (files/hour, chunks/hour)
- Error rate by stage
- Embedding latency distribution
- Queue depth (pending chunks)
- Index size over time
- Deduplication rate

### 7.4 Configuration Management

All pipeline behavior is driven by YAML configuration:

```yaml
# config/pipeline.yaml

pipeline:
  name: rag-ingestion
  log_level: INFO
  max_concurrent_files: 4

ingestion:
  supported_formats:
    - .pdf
    - .docx
    - .txt
    - .md
    - .html
    - .csv
  max_file_size_mb: 100
  deduplication: true

parsing:
  pdf_backend: pdfplumber
  ocr_fallback: false
  ocr_language: eng
  html_strip_tags:
    - script
    - style
    - nav
    - footer

chunking:
  target_chunk_size: 512
  max_chunk_size: 768
  min_chunk_size: 50
  overlap_tokens: 50
  respect_sentence_boundaries: true

embedding:
  backend: sentence-transformers
  model: all-MiniLM-L6-v2
  batch_size: 64
  normalize: true
  cache_enabled: true
  cache_dir: .cache/embeddings

indexing:
  vector_store: opensearch
  opensearch:
    host: localhost
    port: 9200
    index_name: rag_documents
    number_of_shards: 1
    number_of_replicas: 0
  bm25_enabled: true
  incremental: true

observability:
  prometheus_enabled: true
  prometheus_port: 9090
  log_format: json
  tracing_enabled: false
```

**Loading configuration:**

```python
import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    ingestion: dict = field(default_factory=dict)
    parsing: dict = field(default_factory=dict)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: dict = field(default_factory=dict)
    indexing: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)

def load_config(config_path: Path) -> PipelineConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(
        ingestion=raw.get("ingestion", {}),
        parsing=raw.get("parsing", {}),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding=raw.get("embedding", {}),
        indexing=raw.get("indexing", {}),
        observability=raw.get("observability", {}),
    )
```

### 7.5 Pipeline Runner

```python
import time
import uuid

def run_pipeline(
    input_path: Path,
    config: PipelineConfig,
    run_id: str | None = None,
) -> PipelineResult:
    """Execute the full ingestion pipeline."""
    run_id = run_id or str(uuid.uuid4())
    start_time = time.time()
    results = []

    logger.info(f"Pipeline run started: {run_id}")

    # Discover files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(discover_files(input_path))

    logger.info(f"Found {len(files)} files to process")

    for i, file_path in enumerate(files):
        file_start = time.time()
        try:
            # Stage 1: Validate
            validation = validate_file(file_path)
            if not validation.valid:
                results.append(FileResult(file=file_path, status="skipped", error=validation.error))
                continue

            # Stage 2: Parse
            parsed = parse_document(file_path, config.parsing)

            # Stage 3: Chunk
            chunking_result = chunk_document(parsed, config.chunking, count_tokens)

            # Stage 4: Embed
            embeddings = embed_chunks(chunking_result.chunks, get_embedding_backend(config.embedding))

            # Stage 5: Index
            index_chunks(embeddings, config.indexing)

            results.append(FileResult(
                file=file_path,
                status="success",
                chunk_count=len(chunking_result.chunks),
                duration_ms=(time.time() - file_start) * 1000,
            ))

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            results.append(FileResult(file=file_path, status="error", error=str(e)))

        # Update progress
        progress = ((i + 1) / len(files)) * 100
        pipeline_progress.labels(run_id=run_id).set(progress)

    total_duration = time.time() - start_time
    logger.info(f"Pipeline run completed: {run_id} in {total_duration:.1f}s")

    return PipelineResult(
        run_id=run_id,
        total_files=len(files),
        successful=sum(1 for r in results if r.status == "success"),
        failed=sum(1 for r in results if r.status == "error"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        total_chunks=sum(r.chunk_count or 0 for r in results),
        duration_seconds=total_duration,
        results=results,
    )
```

---

## Appendix: Token Counting

A lightweight token counter using tiktoken (for OpenAI models) or a simple whitespace splitter for local models:

```python
def count_tokens(text: str, method: str = "whitespace") -> int:
    if method == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        return len(text.split())
```

The token counting method should match the embedding model's tokenizer to ensure chunk size limits are accurate.
