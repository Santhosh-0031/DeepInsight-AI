"""
Serper Search Provider — Google search results via Serper API.
Uses httpx for async HTTP calls.
"""

import os
from dotenv import load_dotenv, find_dotenv # Add this
from typing import List, Dict, Any
import httpx


SERPER_API_URL = "https://google.serper.dev/search"


class SerperSearchProvider:
    """Search provider using the Serper API for Google search results."""

    def __init__(self):
        # Force search for .env in current or parent directories
        load_dotenv(find_dotenv(), override=True) 
        self._api_key = os.getenv("SERPER_API_KEY")
        
        # Debugging: Print only the first 4 chars to confirm it's there
        if self._api_key:
            print(f"[SerperSearch] Key loaded: {self._api_key[:4]}****")
        else:
            print("[SerperSearch] CRITICAL: Key is still None after load_dotenv")

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Serper (Google) API.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of standardized search result dicts.
        """
        if not self._api_key:
            print("[SerperSearch] SERPER_API_KEY not set, skipping.")
            return []

        try:
            headers = {
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": query,
                "num": num_results,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(SERPER_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("organic", [])[:num_results]:
                url = item.get("link", "")
                results.append({
                    "url": url,
                    "title": item.get("title", "Untitled"),
                    "content": item.get("snippet", ""),
                    "raw_content": "",  # Serper returns snippets, not full content
                    "domain": self._extract_domain(url),
                    "publish_date": item.get("date", ""),
                    "source_type": "serper",
                })
            return results

        except Exception as e:
            print(f"[SerperSearch] Error searching for '{query}': {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
