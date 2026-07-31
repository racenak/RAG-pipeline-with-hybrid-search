# Sample Document

## Introduction

This is a sample markdown document for testing the RAG pipeline.

It preserves heading structure during parsing.

## How It Works

The pipeline processes documents in several stages:

1. **Validation** — check file type and size
2. **Parsing** — extract text content
3. **Cleaning** — normalize whitespace
4. **Chunking** — split into semantic units

## Configuration

All settings are in `config/defaults.yaml`:

```yaml
chunking:
  strategy: semantic
  target_size: 512
```

## Conclusion

This document tests heading extraction and markdown structure preservation.
