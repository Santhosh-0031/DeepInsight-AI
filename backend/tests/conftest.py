"""
Shared test fixtures and configuration for the backend test suite.
"""

import os
import sys
import pytest

# Ensure the backend/app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set minimal env vars so modules can import without crashing
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-real")
os.environ.setdefault("SERPER_API_KEY", "test-key-not-real")
os.environ.setdefault("NEWSAPI_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_MODEL_CHEAP", "google/gemini-2.0-flash-exp:free")
os.environ.setdefault("LLM_MODEL_MID", "anthropic/claude-3.5-haiku")
os.environ.setdefault("LLM_MODEL_PREMIUM", "anthropic/claude-3.5-sonnet")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def sample_search_results():
    """Standard search result dicts as returned by any provider."""
    return [
        {
            "url": "https://arxiv.org/abs/2401.12345",
            "title": "Deep Learning for NLP: A Survey",
            "content": "This paper surveys deep learning approaches in NLP.",
            "raw_content": "Full text of the paper...",
            "domain": "arxiv.org",
            "publish_date": "2024-01-15",
            "source_type": "arxiv",
        },
        {
            "url": "https://www.nature.com/articles/s41586-024-07890",
            "title": "Transformer Architectures: Recent Advances",
            "content": "Transformer models have revolutionized NLP and beyond.",
            "raw_content": "Detailed article content...",
            "domain": "nature.com",
            "publish_date": "2024-06-01",
            "source_type": "tavily",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Deep_learning",
            "title": "Deep learning - Wikipedia",
            "content": "Deep learning is part of machine learning methods.",
            "raw_content": "Full Wikipedia article...",
            "domain": "en.wikipedia.org",
            "publish_date": "",
            "source_type": "wikipedia",
        },
        {
            "url": "https://arxiv.org/abs/2401.12345",  # duplicate URL
            "title": "Deep Learning Survey (duplicate)",
            "content": "Duplicate entry for testing dedup.",
            "raw_content": "",
            "domain": "arxiv.org",
            "publish_date": "2024-01-15",
            "source_type": "serper",
        },
        {
            "url": "https://techcrunch.com/2024/03/ai-advances",
            "title": "AI Advances in 2024",
            "content": "Recent AI advances cover transformers and large language models.",
            "raw_content": "",
            "domain": "techcrunch.com",
            "publish_date": "2024-03-10",
            "source_type": "news",
        },
    ]


@pytest.fixture
def sample_section():
    """A sample Section object for testing."""
    from app.state import Section
    return Section(
        name="Deep Learning Overview",
        description="An overview of deep learning techniques and applications.",
        plan="Cover CNNs, RNNs, Transformers, and recent advances.",
        research=True,
        content="",
    )
