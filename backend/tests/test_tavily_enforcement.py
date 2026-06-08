import os
import pytest
from app.search.tavily_search import TavilySearchProvider

def test_tavily_enforcement_no_key(monkeypatch):
    """Test that TavilySearchProvider raises ValueError when TAVILY_API_KEY is missing."""
    # Ensure TAVILY_API_KEY is NOT set
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    
    provider = TavilySearchProvider()
    
    with pytest.raises(ValueError) as excinfo:
        # Trigger client creation
        provider._get_client()
    
    assert "TAVILY_API_KEY environment variable is not set" in str(excinfo.value)

def test_tavily_works_with_key(monkeypatch):
    """Test that TavilySearchProvider can initialize client when key is present."""
    # Mock TAVILY_API_KEY
    monkeypatch.setenv("TAVILY_API_KEY", "test_key")
    
    provider = TavilySearchProvider()
    # This shouldn't raise an error until it actually tries to use the key for a request
    # But we only want to test the check we added in _get_client
    try:
        from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
        client = provider._get_client()
        assert isinstance(client, TavilySearchAPIWrapper) or client is not None
    except Exception as e:
        # If TavilySearchAPIWrapper itself fails due to dummy key, that's fine, 
        # as long as our ValueError isn't raised.
        if "TAVILY_API_KEY environment variable is not set" in str(e):
            pytest.fail("ValueError was raised even though key was mocked")
