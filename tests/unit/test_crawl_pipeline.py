"""Tests for crawl + ingest pipeline."""

from unittest.mock import MagicMock

from rag_pipeline.pipeline import (
    CrawlResult,
    _crawl_group_from_url,
    _slug_from_url,
    crawl_and_ingest,
)


class TestSlugFromUrl:
    def test_basic(self):
        assert _slug_from_url("https://docs.prefect.io/v3/concepts/tasks") == "tasks"

    def test_index(self):
        assert _slug_from_url("https://docs.prefect.io/v3/concepts/") == "concepts"

    def test_trailing_slash(self):
        assert _slug_from_url("https://example.com/docs/intro/") == "intro"

    def test_deep_path(self):
        assert _slug_from_url("https://a.com/a/b/c/page.html") == "page-html"


class TestCrawlGroupFromUrl:
    def test_basic(self):
        group = _crawl_group_from_url("https://docs.prefect.io/v3/concepts")
        assert "prefect" in group
        assert "v3" in group
        assert "concepts" in group

    def test_no_path(self):
        group = _crawl_group_from_url("https://docs.example.com")
        assert "docs-example-com" in group

    def test_trailing_index(self):
        group = _crawl_group_from_url("https://docs.prefect.io/v3/concepts/index")
        assert "index" not in group.split("-")[-1:]


class TestCrawlResult:
    def test_defaults(self):
        r = CrawlResult(source_url="http://x.com", crawl_group="x")
        assert r.pages_crawled == 0
        assert r.total_chunks == 0
        assert r.errors == []


class TestCrawlAndIngest:
    def _mock_fetcher(self, pages=3) -> MagicMock:
        from rag_pipeline.data.fetchers import FetchedContent

        fetcher = MagicMock()
        fetcher.crawl_site.return_value = [
            FetchedContent(
                content=f"# Page {i}\n\nContent about topic {i}. " * 20,
                metadata={"title": f"Page {i}"},
                source_url=f"https://docs.example.com/page{i}",
                success=True,
            )
            for i in range(pages)
        ]
        return fetcher

    def test_crawl_and_ingest(self):
        from rag_pipeline.data.fetchers import FetchedContent

        fetcher = MagicMock()
        fetcher.crawl_site.return_value = [
            FetchedContent(
                content="# Tasks\n\nTasks are the core unit of work in Prefect. " * 20,
                metadata={"title": "Tasks"},
                source_url="https://docs.prefect.io/v3/concepts/tasks",
                success=True,
            )
        ]

        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_chunks.side_effect = lambda c: c
        os_client = MagicMock()
        pg_client = MagicMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "rag_pipeline.data.fetchers.URLFetcher", return_value=fetcher
        ):
            result = crawl_and_ingest(
                url="https://docs.prefect.io/v3/concepts",
                limit=10,
                storage=storage,
                embedder=embedder,
                opensearch_client=os_client,
                postgres_client=pg_client,
                api_key="test-key",
            )

        assert result.pages_crawled == 1
        assert result.pages_succeeded == 1
        assert result.total_chunks > 0
        assert result.indexed_in_opensearch is True
        assert result.registered_in_postgres is True
        assert len(result.errors) == 0

    def test_crawl_with_failed_pages(self):
        from rag_pipeline.data.fetchers import FetchedContent

        fetcher = MagicMock()
        fetcher.crawl_site.return_value = [
            FetchedContent(
                content="# Good Page\n\nSome content. " * 20,
                metadata={},
                source_url="https://docs.prefect.io/good",
                success=True,
            ),
            FetchedContent(
                content="",
                source_url="https://docs.prefect.io/bad",
                success=False,
                error="404 Not Found",
            ),
        ]

        embedder = MagicMock()
        embedder.embed_chunks.side_effect = lambda c: c
        os_client = MagicMock()
        pg_client = MagicMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "rag_pipeline.data.fetchers.URLFetcher", return_value=fetcher
        ):
            result = crawl_and_ingest(
                url="https://docs.prefect.io/v3/concepts",
                embedder=embedder,
                opensearch_client=os_client,
                postgres_client=pg_client,
                api_key="test-key",
            )

        assert result.pages_crawled == 2
        assert result.pages_succeeded == 1
        assert len(result.errors) == 1
        assert "404" in result.errors[0]

    def test_crawl_s3_key_structure(self):
        from rag_pipeline.data.fetchers import FetchedContent

        fetcher = MagicMock()
        fetcher.crawl_site.return_value = [
            FetchedContent(
                content="# Intro\n\nWelcome. " * 20,
                metadata={},
                source_url="https://docs.prefect.io/v3/concepts/intro",
                success=True,
            )
        ]

        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_chunks.side_effect = lambda c: c
        os_client = MagicMock()
        pg_client = MagicMock()

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "rag_pipeline.data.fetchers.URLFetcher", return_value=fetcher
        ):
            crawl_and_ingest(
                url="https://docs.prefect.io/v3/concepts",
                storage=storage,
                embedder=embedder,
                opensearch_client=os_client,
                postgres_client=pg_client,
                api_key="test-key",
            )

        # Check S3 key structure
        storage.upload_bytes.assert_called_once()
        call_kwargs = storage.upload_bytes.call_args
        key = call_kwargs[1]["key"]
        assert key.startswith("crawl/")
        assert "prefect" in key
        assert key.endswith(".md")
