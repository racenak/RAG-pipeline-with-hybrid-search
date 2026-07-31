#!/usr/bin/env python3
"""CLI for document ingestion.

Usage:
    python scripts/ingest.py file path/to/doc.pdf
    python scripts/ingest.py url https://docs.example.com
    python scripts/ingest.py dir ./docs --recursive
    python scripts/ingest.py dir ./docs --no-recursive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _get_storage() -> object | None:
    """Create S3Storage client if available, else None."""
    try:
        from rag_pipeline.config import get_settings
        from rag_pipeline.data.storage import S3Storage

        settings = get_settings()
        return S3Storage(
            endpoint_url=settings.s3.endpoint_url,
            access_key=settings.s3.access_key,
            secret_key=settings.s3.secret_key,
            bucket=settings.s3.bucket,
        )
    except Exception:
        return None


def _collect_directory(directory, storage, recursive):  # noqa: ANN001
    """Collect all ingestion results from a directory, catching per-file errors."""
    from rag_pipeline.data.ingestion import discover_files
    from rag_pipeline.data.models import IngestionResult
    from rag_pipeline.data.parsers import get_parser
    from rag_pipeline.data.validation import validate_file

    results = []
    for path in discover_files(directory, recursive):
        try:
            v = validate_file(path)
            if not v.valid:
                results.append(IngestionResult(
                    document_id="", source=str(path), source_type="file",
                    success=False, error=v.error,
                ))
                continue

            parser = get_parser(path)
            parser.parse(path)

            fid = None
            if storage is not None:
                fid = storage.upload_file(path)

            results.append(IngestionResult(
                document_id=v.file_hash or "",
                source=str(path), source_type="file",
                success=True, seaweedfs_fid=fid,
            ))
        except Exception as e:
            results.append(IngestionResult(
                document_id="", source=str(path), source_type="file",
                success=False, error=str(e),
            ))

    return results


def cmd_file(args: argparse.Namespace) -> None:
    """Ingest a single file."""
    from rag_pipeline.data import ingest_file

    storage = None if args.no_storage else _get_storage()
    try:
        try:
            result = ingest_file(Path(args.path), storage=storage)
        except Exception as e:
            if storage:
                print(f"⚠️  S3 failed ({e}), retrying without storage...")
                result = ingest_file(Path(args.path), storage=None)
            else:
                raise

        if result.success:
            print(f"✅ {result.source}")
            print(f"   document_id: {result.document_id}")
            if result.seaweedfs_fid:
                print(f"   s3_key: {result.seaweedfs_fid}")
        else:
            print(f"❌ {result.source}: {result.error}")
            sys.exit(1)
    finally:
        if storage:
            storage.close()


def cmd_url(args: argparse.Namespace) -> None:
    """Ingest a URL."""
    from rag_pipeline.data import ingest_url
    from rag_pipeline.data.fetchers import MockURLFetcher

    storage = None if args.no_storage else _get_storage()
    fetcher = MockURLFetcher()
    try:
        try:
            result = ingest_url(args.url, fetcher, storage=storage)
        except Exception as e:
            if storage:
                print(f"⚠️  S3 failed ({e}), retrying without storage...")
                result = ingest_url(args.url, fetcher, storage=None)
            else:
                raise

        if result.success:
            print(f"✅ {result.source}")
            print(f"   document_id: {result.document_id}")
            if result.seaweedfs_fid:
                print(f"   s3_key: {result.seaweedfs_fid}")
        else:
            print(f"❌ {result.source}: {result.error}")
            sys.exit(1)
    finally:
        if storage:
            storage.close()


def cmd_dir(args: argparse.Namespace) -> None:
    """Ingest all files in a directory."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)

    storage = None if args.no_storage else _get_storage()
    try:
        results = _collect_directory(directory, storage, args.recursive)

        failed = [r for r in results if not r.success]
        if storage and failed and len(failed) == len(results):
            print("⚠️  All uploads failed — retrying without storage...")
            storage.close()
            storage = None
            results = _collect_directory(directory, None, args.recursive)

        for r in results:
            if r.success:
                fid = f" → {r.seaweedfs_fid}" if r.seaweedfs_fid else ""
                print(f"✅ {r.source}{fid}")
            else:
                print(f"❌ {r.source}: {r.error}")

        total = len(results)
        successful = sum(1 for r in results if r.success)
        print(f"\n{'='*50}")
        print(f"Total: {total} | Successful: {successful} | Failed: {total - successful}")
        if total == 0:
            print("No supported files found.")
    finally:
        if storage:
            storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Document Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Skip S3 storage (parse only)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser("file", help="Ingest a single file")
    file_parser.add_argument("path", help="Path to file")
    file_parser.set_defaults(func=cmd_file)

    url_parser = subparsers.add_parser("url", help="Ingest a URL via Firecrawl")
    url_parser.add_argument("url", help="URL to ingest")
    url_parser.set_defaults(func=cmd_url)

    dir_parser = subparsers.add_parser("dir", help="Ingest all files in a directory")
    dir_parser.add_argument("directory", help="Directory path")
    dir_parser.add_argument(
        "--recursive", action="store_true", default=True, help="Scan recursively (default)"
    )
    dir_parser.add_argument(
        "--no-recursive", action="store_false", dest="recursive", help="Don't scan subdirs"
    )
    dir_parser.set_defaults(func=cmd_dir)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
