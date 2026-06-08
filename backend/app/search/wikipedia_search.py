"""
Wikipedia Search Provider — Background context and definitions.
Uses the wikipedia-api Python library for free access.
"""

from typing import List, Dict, Any
import asyncio


class WikipediaSearchProvider:
    """Search provider using Wikipedia API for background context and definitions."""

    def __init__(self, language: str = "en"):
        self._language = language

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for relevant articles.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of standardized search result dicts.
        """
        try:
            import wikipediaapi

            wiki = wikipediaapi.Wikipedia(
                user_agent="DeepResearchAgent/2.0 (https://github.com/abhigupta/deep_research_agent)",
                language=self._language,
            )

            # Wikipedia API doesn't have a direct search — we use the page lookup
            # For better coverage, we also try the opensearch endpoint
            results = await asyncio.to_thread(self._search_sync, wiki, query, num_results)
            return results

        except ImportError:
            print("[WikipediaSearch] 'wikipedia-api' package not installed. Run: pip install wikipedia-api")
            return []
        except Exception as e:
            print(f"[WikipediaSearch] Error searching for '{query}': {e}")
            return []

    def _search_sync(self, wiki, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Synchronous search logic run in thread pool."""
        import urllib.request
        import urllib.parse
        import json

        results = []

        # Use MediaWiki opensearch API for actual search
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://{self._language}.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={num_results}&format=json"

            # Must set User-Agent explicitly on the Request object — urlopen(url) sends
            # "Python-urllib/3.x" which Wikipedia blocks with a 403.
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "DeepResearchAgent/2.0 (https://github.com/abhigupta/deep_research_agent)"
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if len(data) >= 4:
                titles = data[1]
                descriptions = data[2]
                urls = data[3]

                for i in range(min(len(titles), num_results)):
                    # Fetch the full page content for each result
                    page = wiki.page(titles[i])
                    content = page.summary if page.exists() else descriptions[i] if i < len(descriptions) else ""

                    results.append({
                        "url": urls[i] if i < len(urls) else "",
                        "title": titles[i],
                        "content": content[:2000],  # Truncate to avoid massive content
                        "raw_content": content,
                        "domain": "wikipedia.org",
                        "publish_date": "",  # Wikipedia doesn't expose publication dates easily
                        "source_type": "wikipedia",
                    })
        except Exception as e:
            print(f"[WikipediaSearch] OpenSearch fallback error: {e}")

        return results