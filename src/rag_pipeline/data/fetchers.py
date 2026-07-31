"""URL ingestion via Firecrawl — single page and site crawling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FetchedContent:
    """Content fetched from a URL via Firecrawl."""

    content: str  # Markdown from Firecrawl
    metadata: dict[str, Any] = field(default_factory=dict)
    source_url: str = ""
    success: bool = False
    error: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


class URLFetcher:
    """Fetch web content using Firecrawl API.

    Requires FIRECRAWL_API_KEY environment variable.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")
        from firecrawl import FirecrawlApp

        self.app = FirecrawlApp(api_key=api_key)

    def fetch_url(self, url: str) -> FetchedContent:
        """Scrape a single URL → markdown content."""
        try:
            result = self.app.scrape_url(url)
            return FetchedContent(
                content=result.get("markdown", ""),
                metadata=result.get("metadata", {}),
                source_url=url,
                success=result.get("success", True),
            )
        except Exception as e:
            return FetchedContent(
                content="",
                source_url=url,
                success=False,
                error=str(e),
            )

    def crawl_site(self, url: str, limit: int = 100) -> list[FetchedContent]:
        """Crawl a site starting from url, up to limit pages."""
        try:
            crawl_result = self.app.crawl_url(
                url,
                params={"limit": limit},
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
        except Exception as e:
            return [
                FetchedContent(
                    content="",
                    source_url=url,
                    success=False,
                    error=str(e),
                )
            ]


class MockURLFetcher(URLFetcher):
    """Mock fetcher for testing — no API key required."""

    def __init__(self) -> None:
        self.app = None  # type: ignore[assignment]

    def fetch_url(self, url: str) -> FetchedContent:
        return FetchedContent(
            content=f"# Mock Content\n\nContent fetched from {url}",
            metadata={"title": f"Mock Page: {url}"},
            source_url=url,
            success=True,
        )

    def crawl_site(self, url: str, limit: int = 100) -> list[FetchedContent]:  # noqa: ARG002
        return [self.fetch_url(url)]
