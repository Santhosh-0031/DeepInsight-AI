import httpx
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class JinaReader:
    """
    Utility to fetch cleaned markdown content from URLs using r.jina.ai.
    This is used for 'Selective Deep Extraction' of top-ranked sources.
    """
    
    def __init__(self, timeout: int = 15):
        self.base_url = "https://r.jina.ai/"
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(3)  # Limit concurrent deep extractions

    async def fetch_markdown(self, url: str) -> Optional[str]:
        """
        Fetch the markdown representation of a webpage using Jina Reader.
        """
        async with self._semaphore:
            try:
                target_url = f"{self.base_url}{url}"
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        target_url, 
                        timeout=self.timeout,
                        follow_redirects=True
                    )
                    if response.status_code == 200:
                        return response.text
                    else:
                        logger.warning(f"[JinaReader] Failed to fetch {url}: HTTP {response.status_code}")
                        return None
            except Exception as e:
                logger.error(f"[JinaReader] Error fetching {url}: {e}")
                return None
