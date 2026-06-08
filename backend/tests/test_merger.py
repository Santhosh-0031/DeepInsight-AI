"""
Test: Result Merger + Ranker (search/merger.py)

Verifies deduplication, scoring, ranking, and formatting logic.
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.search.merger import ResultMerger, SourceMetadata, format_ranked_results


class TestSourceMetadata:

    def test_to_dict_round_trip(self):
        src = SourceMetadata(
            url="https://example.com/page",
            title="Test Page",
            domain="example.com",
            content="Page content here.",
            source_type="tavily",
            credibility_score=0.8,
            relevance_score=0.7,
            final_score=0.75,
        )
        d = src.to_dict()
        assert d["url"] == "https://example.com/page"
        assert d["title"] == "Test Page"
        assert d["domain"] == "example.com"
        assert d["credibility_score"] == 0.8
        assert d["final_score"] == 0.75

    def test_default_scores(self):
        src = SourceMetadata()
        assert src.credibility_score == 0.5
        assert src.recency_score == 0.5
        assert src.relevance_score == 0.5
        assert src.final_score == 0.0


class TestResultMerger:

    def setup_method(self):
        self.merger = ResultMerger()

    @pytest.fixture(autouse=True)
    def mock_embeddings(self):
        with patch("app.search.merger.embed_text", new_callable=AsyncMock) as mock_embed_text, \
             patch("app.search.merger.embed_texts", new_callable=AsyncMock) as mock_embed_texts:
            mock_embed_text.return_value = [0.1] * 1536
            # Always return a list of valid embeddings
            mock_embed_texts.side_effect = lambda texts: [[0.1]*1536]*len(texts)
            yield

    @pytest.mark.asyncio
    async def test_deduplication_by_url(self, sample_search_results):
        """Duplicate URLs should be removed."""
        ranked = await self.merger.merge_and_rank(sample_search_results, top_k=10)
        urls = [s.url for s in ranked]
        assert len(urls) == len(set(urls)), "Duplicate URLs found after merge_and_rank"

    @pytest.mark.asyncio
    async def test_dedup_reduces_count(self, sample_search_results):
        """Input has 5 items with 1 duplicate URL → output should have 4."""
        ranked = await self.merger.merge_and_rank(sample_search_results, top_k=10)
        assert len(ranked) == 4

    @pytest.mark.asyncio
    async def test_results_sorted_by_final_score_descending(self, sample_search_results):
        ranked = await self.merger.merge_and_rank(sample_search_results, top_k=10)
        scores = [s.final_score for s in ranked]
        assert scores == sorted(scores, reverse=True), "Results not sorted by final_score"

    @pytest.mark.asyncio
    async def test_top_k_limits_output(self, sample_search_results):
        ranked = await self.merger.merge_and_rank(sample_search_results, top_k=2)
        assert len(ranked) <= 2

    @pytest.mark.asyncio
    async def test_empty_input(self):
        ranked = await self.merger.merge_and_rank([], top_k=10)
        assert ranked == []

    def test_credibility_scoring_known_domains(self):
        high = self.merger._score_credibility("nature.com")
        low = self.merger._score_credibility("randomsite.xyz")
        assert high > low, "Known authority domain should score higher"

    def test_credibility_scoring_academic(self):
        score = self.merger._score_credibility("arxiv.org")
        assert score >= 0.8, "Academic domains should score high"

    def test_relevance_scoring_with_hyde(self):
        """Higher cosine similarity should yield higher relevance."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        high_rel = self.merger._score_relevance_semantic(emb1, emb2)
        low_rel  = self.merger._score_relevance_semantic(emb1, emb3)
        assert high_rel > low_rel

    def test_recency_scoring_recent_date(self):
        recent = self.merger._score_recency("2025-01-01")
        old = self.merger._score_recency("2020-01-01")
        assert recent >= old, "Recent dates should score higher"

    def test_recency_scoring_empty_date(self):
        score = self.merger._score_recency("")
        assert 0.0 <= score <= 1.0

    def test_corroboration_calculation(self):
        sources = [
            SourceMetadata(title="Deep Learning Survey", content="deep learning approaches", embedding=[1.0, 0.0, 0.0]),
            SourceMetadata(title="Deep Learning Review", content="deep learning methods and techniques", embedding=[1.0, 0.0, 0.0]),
            SourceMetadata(title="Cooking Recipes", content="pasta and pizza recipes", embedding=[0.0, 1.0, 0.0]),
        ]
        self.merger._calculate_corroboration(sources)
        # The two deep learning sources should corroborate each other since embeddings match exactly
        assert sources[0].corroboration >= 1 or sources[1].corroboration >= 1


class TestFormatRankedResults:

    def test_format_empty_list(self):
        result = format_ranked_results([], max_tokens=5000)
        assert result == "No search results found."

    def test_format_includes_source_content(self):
        sources = [
            SourceMetadata(
                url="https://example.com",
                title="Test",
                content="Important finding about AI.",
                final_score=0.9,
            )
        ]
        result = format_ranked_results(sources, max_tokens=5000)
        assert "Important finding" in result
        assert "example.com" in result or "Test" in result

    def test_format_respects_max_tokens(self):
        sources = [
            SourceMetadata(
                url=f"https://example.com/{i}",
                title=f"Source {i}",
                content="A" * 5000,  # Each source has lots of content
                final_score=0.9 - i * 0.1,
            )
            for i in range(10)
        ]
        result = format_ranked_results(sources, max_tokens=1000)
        # Result should be truncated near the max_tokens limit
        assert len(result) < 50000  # Should not include all 50K chars
