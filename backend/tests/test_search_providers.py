"""
Test: Search Providers (search/*.py)

Mocks external API calls and verifies each provider returns standardized
result dicts with required keys.
"""

from unittest.mock import patch, AsyncMock, MagicMock
import pytest


REQUIRED_KEYS = {"url", "title", "content", "domain", "source_type"}


class TestTavilySearchProvider:

    @pytest.mark.asyncio
    async def test_search_returns_standardized_results(self):
        from app.search.tavily_search import TavilySearchProvider

        mock_raw = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Test Article",
                    "content": "Article content.",
                    "raw_content": "Full raw text.",
                    "published_date": "2024-01-01",
                },
            ]
        }

        provider = TavilySearchProvider()
        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.raw_results_async = AsyncMock(return_value=mock_raw)
            results = await provider.search("test query", num_results=3)

        assert len(results) == 1
        assert REQUIRED_KEYS.issubset(results[0].keys())
        assert results[0]["source_type"] == "tavily"
        assert results[0]["domain"] == "example.com"

    @pytest.mark.asyncio
    async def test_search_handles_api_error_gracefully(self):
        from app.search.tavily_search import TavilySearchProvider

        provider = TavilySearchProvider()
        with patch.object(provider, "_get_client") as mock_client:
            mock_client.return_value.raw_results_async = AsyncMock(
                side_effect=Exception("API rate limit")
            )
            results = await provider.search("test query")

        assert results == []


class TestSerperSearchProvider:

    @pytest.mark.asyncio
    async def test_search_returns_standardized_results(self):
        from app.search.serper_search import SerperSearchProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "link": "https://example.com/page",
                    "title": "Serper Result",
                    "snippet": "Search snippet text.",
                    "date": "2024-02-01",
                },
            ]
        }

        provider = SerperSearchProvider()
        with patch("app.search.serper_search.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            results = await provider.search("test query", num_results=3)

        assert len(results) == 1
        assert REQUIRED_KEYS.issubset(results[0].keys())
        assert results[0]["source_type"] == "serper"


class TestArxivSearchProvider:

    @pytest.mark.asyncio
    async def test_search_returns_standardized_results(self):
        from app.search.arxiv_search import ArxivSearchProvider

        mock_paper = MagicMock()
        mock_paper.entry_id = "https://arxiv.org/abs/2401.99999"
        mock_paper.title = "Arxiv Paper Title"
        mock_paper.summary = "Paper abstract text."
        mock_paper.published = MagicMock()
        mock_paper.published.strftime = MagicMock(return_value="2024-01-15")

        # Mock the `arxiv` module that gets imported inside the function
        mock_arxiv_module = MagicMock()
        mock_arxiv_module.Search.return_value = MagicMock()
        mock_arxiv_module.SortCriterion.Relevance = "relevance"
        mock_arxiv_module.Client.return_value = MagicMock()

        provider = ArxivSearchProvider()
        with patch.dict("sys.modules", {"arxiv": mock_arxiv_module}):
            with patch("app.search.arxiv_search.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = [mock_paper]
                results = await provider.search("transformer attention", num_results=2)

        assert len(results) == 1
        assert REQUIRED_KEYS.issubset(results[0].keys())
        assert results[0]["source_type"] == "arxiv"
        assert "arxiv.org" in results[0]["url"]


class TestWikipediaSearchProvider:

    @pytest.mark.asyncio
    async def test_search_returns_standardized_results(self):
        from app.search.wikipedia_search import WikipediaSearchProvider

        provider = WikipediaSearchProvider()

        mock_results = [
            {
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "title": "Machine learning",
                "content": "Machine learning is a subset of AI.",
                "raw_content": "Full article text...",
                "domain": "wikipedia.org",
                "publish_date": "",
                "source_type": "wikipedia",
            },
        ]

        # Mock the `wikipediaapi` import and asyncio.to_thread
        mock_wikiapi = MagicMock()
        mock_wikiapi.Wikipedia.return_value = MagicMock()
        with patch.dict("sys.modules", {"wikipediaapi": mock_wikiapi}):
            with patch("app.search.wikipedia_search.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = mock_results
                results = await provider.search("machine learning", num_results=2)

        assert len(results) == 1
        assert REQUIRED_KEYS.issubset(results[0].keys())
        assert results[0]["source_type"] == "wikipedia"


class TestNewsSearchProvider:

    @pytest.mark.asyncio
    async def test_search_returns_standardized_results(self):
        from app.search.news_search import NewsSearchProvider

        mock_raw_results = {
            "articles": [
                {
                    "url": "https://news.com/ai-story",
                    "title": "AI News Title",
                    "description": "Short description.",
                    "content": "Full article content.",
                    "publishedAt": "2024-03-01T10:00:00Z",
                    "source": {"name": "TechNews"},
                },
            ]
        }

        provider = NewsSearchProvider()
        # Mock the `newsapi` import and asyncio.to_thread
        mock_newsapi = MagicMock()
        mock_newsapi.NewsApiClient.return_value = MagicMock()
        with patch.dict("sys.modules", {"newsapi": mock_newsapi}):
            with patch("app.search.news_search.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = mock_raw_results
                results = await provider.search("artificial intelligence", num_results=3)

        assert len(results) == 1
        assert REQUIRED_KEYS.issubset(results[0].keys())
        assert results[0]["source_type"] == "newsapi"
