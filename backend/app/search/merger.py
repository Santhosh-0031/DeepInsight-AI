"""
Result Merger + Credibility Ranker.
Pools results from all search providers, deduplicates, and ranks by credibility.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from ..embeddings import embed_text, embed_texts, cosine_similarity


# Domain authority tiers for credibility scoring
HIGH_CREDIBILITY_DOMAINS = {
    "arxiv.org", "wikipedia.org", "nature.com", "science.org",
    "ieee.org", "acm.org", "nih.gov", "ncbi.nlm.nih.gov",
    "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
}

MEDIUM_CREDIBILITY_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "bloomberg.com",
    "techcrunch.com", "wired.com", "arstechnica.com",
    "github.com", "stackoverflow.com", "medium.com",
}


@dataclass
class SourceMetadata:
    """Metadata for a single source result with credibility scoring."""

    url: str = ""
    title: str = ""
    domain: str = ""
    credibility_score: float = 0.5  # 0.0 – 1.0 (domain authority)
    recency_score: float = 0.5     # 0.0 – 1.0 (how recent the content is)
    relevance_score: float = 0.5   # 0.0 – 1.0 (cosine similarity to HyDE doc)
    corroboration: int = 0         # how many other sources say the same thing
    final_score: float = 0.0       # weighted combination
    publish_date: str = ""
    content: str = ""
    raw_content: str = ""
    source_type: str = ""          # tavily, serper, arxiv, wikipedia, newsapi
    # Optional embedding vector for corroboration calculation
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "credibility_score": self.credibility_score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "corroboration": self.corroboration,
            "final_score": self.final_score,
            "publish_date": self.publish_date,
            "content": self.content,
            "source_type": self.source_type,
        }


class ResultMerger:
    """Merges, deduplicates, and ranks search results from multiple providers."""

    # Weights for final score calculation
    CREDIBILITY_WEIGHT = 0.3
    RECENCY_WEIGHT = 0.15
    RELEVANCE_WEIGHT = 0.4
    CORROBORATION_WEIGHT = 0.15

    async def merge_and_rank(
        self,
        all_results: List[Dict[str, Any]],
        hyde_document: str = "",
        hyde_embedding: Optional[List[float]] = None,
        top_k: int = 15,
    ) -> List[SourceMetadata]:
        """
        Merge results from multiple providers, deduplicate, score, and rank.

        Args:
            all_results: Flat list of search result dicts from all providers.
            hyde_document: HyDE hypothetical document for relevance scoring.
            hyde_embedding: Pre-computed HyDE embedding (P4 — skips API call if supplied).
            top_k: Number of top results to return.

        Returns:
            List of SourceMetadata objects, sorted by final_score descending.
        """
        if not all_results:
            return []

        # Step 1: Deduplicate by URL
        unique_results = self._deduplicate_by_url(all_results)

        # Step 2: Embed documents. If a pre-computed HyDE embedding is supplied,
        # skip the embed_text() API call (P4 optimization).
        contents_to_embed = [res.get("content", "") for res in unique_results]
        if hyde_embedding is not None:
            # Reuse the caller-supplied vector; only embed documents
            doc_embeddings = await embed_texts(contents_to_embed)
            hyde_emb = hyde_embedding
        else:
            # Fallback: compute HyDE embedding here (slower path)
            hyde_embedding_task = embed_text(hyde_document)
            doc_embeddings_task = embed_texts(contents_to_embed)
            hyde_emb, doc_embeddings = await asyncio.gather(hyde_embedding_task, doc_embeddings_task)

        # Step 3: Convert to SourceMetadata and score each
        scored_results = []
        for i, result in enumerate(unique_results):
            metadata = SourceMetadata(
                url=result.get("url", ""),
                title=result.get("title", "Untitled"),
                domain=result.get("domain", ""),
                content=result.get("content", ""),
                raw_content=result.get("raw_content", ""),
                publish_date=result.get("publish_date", ""),
                source_type=result.get("source_type", "unknown"),
                embedding=doc_embeddings[i]
            )

            # Score credibility based on domain
            metadata.credibility_score = self._score_credibility(metadata.domain)

            # Score recency based on publish date
            metadata.recency_score = self._score_recency(metadata.publish_date)

            # Relevance scoring using embedding cosine similarity
            metadata.relevance_score = self._score_relevance_semantic(
                metadata.embedding, hyde_emb
            )

            scored_results.append(metadata)

        # Step 4: Calculate corroboration (how many sources cover similar content)
        self._calculate_corroboration(scored_results)

        # Step 4: Calculate final weighted score
        for result in scored_results:
            result.final_score = (
                self.CREDIBILITY_WEIGHT * result.credibility_score
                + self.RECENCY_WEIGHT * result.recency_score
                + self.RELEVANCE_WEIGHT * result.relevance_score
                + self.CORROBORATION_WEIGHT * min(result.corroboration / 3.0, 1.0)
            )

        # Step 5: Sort by final score and return top-K
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        return scored_results[:top_k]

    def _deduplicate_by_url(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique = []
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(result)
        return unique

    def _score_credibility(self, domain: str) -> float:
        """Score domain credibility based on known authority tiers."""
        domain_lower = domain.lower()

        # Check for .gov and .edu domains
        if domain_lower.endswith(".gov"):
            return 0.95
        if domain_lower.endswith(".edu"):
            return 0.9

        # Check against known high/medium credibility domains
        for high_domain in HIGH_CREDIBILITY_DOMAINS:
            if high_domain in domain_lower:
                return 0.9
        for med_domain in MEDIUM_CREDIBILITY_DOMAINS:
            if med_domain in domain_lower:
                return 0.7

        # Default credibility for unknown domains
        return 0.5

    def _score_recency(self, publish_date: str) -> float:
        """Score recency — more recent content scores higher."""
        if not publish_date:
            return 0.3  # Unknown date gets a low-ish score

        try:
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                try:
                    dt = datetime.strptime(publish_date[:len(fmt.replace('%', 'X'))], fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.3

            days_old = (datetime.now() - dt).days
            if days_old <= 30:
                return 1.0
            elif days_old <= 90:
                return 0.85
            elif days_old <= 365:
                return 0.6
            elif days_old <= 730:
                return 0.4
            else:
                return 0.2

        except Exception:
            return 0.3

    def _score_relevance_semantic(self, content_emb: Optional[List[float]], hyde_emb: Optional[List[float]]) -> float:
        """
        Relevance scoring using embedding cosine similarity.
        """
        if not hyde_emb or not content_emb:
            return 0.5

        # Extract value between 0.0 and 1.0 safely from cosine similarity (-1.0 to 1.0)
        sim = cosine_similarity(content_emb, hyde_emb)
        # Shift domain [-1, 1] to [0, 1] loosely, or just bound it:
        score = max(sim, 0.1)
        return min(score, 1.0)

    def _calculate_corroboration(self, results: List[SourceMetadata]) -> None:
        """
        Calculate corroboration — how many other sources cover similar content.
        Uses semantic embedding similarity matching.
        """
        for i, result_a in enumerate(results):
            count = 0
            if not result_a.embedding:
                result_a.corroboration = count
                continue
                
            for j, result_b in enumerate(results):
                if i == j or not result_b.embedding:
                    continue
                
                # If content embeddings share significant similarity
                sim = cosine_similarity(result_a.embedding, result_b.embedding)
                if sim >= 0.75:  # High cosine similarity indicates corroboration
                    count += 1
            result_a.corroboration = count


def format_ranked_results(sources: List[SourceMetadata], max_tokens: int = 10000) -> str:
    """
    Format ranked sources into a string for LLM consumption.

    Args:
        sources: List of SourceMetadata objects, already ranked.
        max_tokens: Approximate max character limit (rough estimate).

    Returns:
        Formatted string with ranked source content.
    """
    if not sources:
        return "No search results found."

    formatted = "Content from ranked web search results:\n\n"
    char_count = len(formatted)

    for i, source in enumerate(sources, 1):
        entry = (
            f"Source {i} [{source.source_type.upper()}] (Score: {source.final_score:.2f}):\n"
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Content: {source.content}\n\n"
        )

        if char_count + len(entry) > max_tokens:
            break

        formatted += entry
        char_count += len(entry)

    return formatted.strip()
