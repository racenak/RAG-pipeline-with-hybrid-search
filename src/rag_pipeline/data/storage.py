"""S3-compatible file storage (SeaweedFS or any S3 provider)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "rag-documents"


class S3Storage:
    """S3-compatible storage client."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8333",
        access_key: str = "anything",
        secret_key: str = "anything",
        bucket: str = DEFAULT_BUCKET,
    ) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            self._client.create_bucket(Bucket=self.bucket)
            logger.info("Created bucket: %s", self.bucket)

    def upload_file(self, file_path: Path, key: str | None = None) -> str:
        """Upload a local file. Returns the S3 key."""
        key = key or f"files/{file_path.name}"
        self._client.upload_file(str(file_path), self.bucket, key)
        logger.info("Uploaded %s → s3://%s/%s", file_path.name, self.bucket, key)
        return key

    def upload_bytes(self, data: bytes, key: str) -> str:
        """Upload raw bytes. Returns the S3 key."""
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        logger.info("Uploaded %d bytes → s3://%s/%s", len(data), self.bucket, key)
        return key

    def download_file(self, key: str) -> bytes:
        """Download file content by key."""
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def delete_file(self, key: str) -> None:
        """Delete a file by key."""
        self._client.delete_object(Bucket=self.bucket, Key=key)
        logger.info("Deleted s3://%s/%s", self.bucket, key)

    def close(self) -> None:
        """Noop — boto3 manages connections internally."""
