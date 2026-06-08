"""
NewsAPI Search Provider — Recent news articles (last 30 days).
Uses the newsapi-python library.
"""

import os
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta


class NewsSearchProvider:
    """Search provider using NewsAPI for recent news articles (last 30 days)."""

    def __init__(self):
        self._api_key = os.getenv("NEWSAPI_KEY")

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search recent news articles via NewsAPI.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of standardized search result dicts.
        """
        if not self._api_key:
            print("[NewsSearch] NEWSAPI_KEY not set, skipping.")
            return []

        try:
            from newsapi import NewsApiClient

            newsapi = NewsApiClient(api_key=self._api_key)

            # Search for articles from the last 30 days
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            # Run sync newsapi call in thread pool
            raw_results = await asyncio.to_thread(
                lambda: newsapi.get_everything(
                    q=query,
                    from_param=from_date,
                    sort_by="relevancy",
                    page_size=num_results,
                    language="en",
                )
            )

            results = []
            for article in raw_results.get("articles", [])[:num_results]:
                url = article.get("url", "")
                results.append({
                    "url": url,
                    "title": article.get("title", "Untitled"),
                    "content": article.get("description", "") or article.get("content", ""),
                    "raw_content": article.get("content", ""),
                    "domain": self._extract_domain(url),
                    "publish_date": article.get("publishedAt", "")[:10],  # YYYY-MM-DD
                    "source_type": "newsapi",
                })
            return results

        except ImportError:
            print("[NewsSearch] 'newsapi-python' package not installed. Run: pip install newsapi-python")
            return []
        except Exception as e:
            print(f"[NewsSearch] Error searching for '{query}': {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
