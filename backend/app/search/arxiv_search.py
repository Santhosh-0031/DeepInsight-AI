"""
ArXiv Search Provider — Academic papers and research.
Uses the arxiv Python library for free API access.

Rate-limit protection:
  - Global semaphore (Semaphore(1)) serialises all concurrent ArXiv calls
    so parallel section searches never hammer the API simultaneously.
  - Client configured with num_retries=1 and delay_seconds=1 so a 429
    fails fast (2s) instead of burning 12s on 4 retries × 3s sleep.
"""

from typing import List, Dict, Any
import asyncio

# Global semaphore — serialises ALL concurrent ArXiv queries across sections.
# ArXiv's free API rate-limits aggressively when hit in parallel; one-at-a-time
# prevents HTTP 429 storms and eliminates the 24s of wasted retry sleeps.
_arxiv_semaphore = asyncio.Semaphore(1)


class ArxivSearchProvider:
    """Search provider using the ArXiv API for academic papers and research."""

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search ArXiv for academic papers.

        Args:
            query: Search query string.
            num_results: Maximum number of results to return.

        Returns:
            List of standardized search result dicts.
        """
        try:
            import arxiv

            # Acquire the global semaphore before touching ArXiv.
            # This queues concurrent section searches instead of letting them
            # collide and trigger rate-limit retries.
            async with _arxiv_semaphore:
                # num_retries=1 → only 1 retry on failure (was 4 by default).
                # delay_seconds=1 → 1s between retries (was 3s by default).
                # net worst-case wait: 2s instead of 12s per query.
                client = arxiv.Client(num_retries=1, delay_seconds=1)
                search = arxiv.Search(
                    query=query,
                    max_results=num_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                # Run the sync arxiv search in a thread pool to avoid blocking
                results_list = await asyncio.to_thread(
                    lambda: list(client.results(search))
                )

            results = []
            for paper in results_list:
                results.append({
                    "url": paper.entry_id,
                    "title": paper.title,
                    "content": paper.summary,
                    "raw_content": "",
                    "domain": "arxiv.org",
                    "publish_date": paper.published.strftime("%Y-%m-%d") if paper.published else "",
                    "source_type": "arxiv",
                })
            return results

        except ImportError:
            print("[ArxivSearch] 'arxiv' package not installed. Run: pip install arxiv")
            return []
        except Exception as e:
            print(f"[ArxivSearch] Error searching for '{query}': {e}")
            return []
