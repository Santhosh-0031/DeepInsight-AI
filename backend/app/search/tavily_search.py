"""
Tavily Search Provider — Deep web, curated results.
Extracted from the original utils.py and wrapped in a standardized provider interface.
"""

import asyncio
from typing import List, Dict, Any
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilySearchProvider:
    """Search provider using the Tavily API for deep web, curated search results."""

    def __init__(self):
        self._client = None

    def _get_client(self) -> TavilySearchAPIWrapper:
        if self._client is None:
            import os
            if not os.getenv("TAVILY_API_KEY"):
                raise ValueError("TAVILY_API_KEY environment variable is not set. Tavily search requires a valid API key.")
            self._client = TavilySearchAPIWrapper()
        return self._client

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Tavily API.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of standardized search result dicts with keys:
            url, title, content, raw_content, domain, publish_date, source_type
        """
        try:
            raw_results = await self._get_client().raw_results_async(
                query=query,
                max_results=num_results,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
            )

            results = []
            for item in raw_results.get("results", []):
                url = item.get("url", "")
                results.append({
                    "url": url,
                    "title": item.get("title", "Untitled"),
                    "content": item.get("content", ""),
                    "raw_content": "",
                    "domain": self._extract_domain(url),
                    "publish_date": item.get("published_date", ""),
                    "source_type": "tavily",
                })
            return results

        except Exception as e:
            print(f"[TavilySearch] Error searching for '{query}': {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
