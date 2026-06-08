import os
import asyncio
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

from app.search.tavily_search import TavilySearchProvider
from app.search.serper_search import SerperSearchProvider
from app.search.arxiv_search import ArxivSearchProvider
from app.search.wikipedia_search import WikipediaSearchProvider
from app.search.news_search import NewsSearchProvider


async def test_provider(provider_name: str, provider_instance, query: str = "Artificial Intelligence"):
    """Run a basic search to verify the provider API key and connection."""
    print(f"\n--- Testing {provider_name} ---")
    try:
        results = await provider_instance.search(query, num_results=1)
        if isinstance(results, Exception):
            print(f"❌ {provider_name} FAILED: {results}")
        else:
            if results:
                print(f"✅ {provider_name} PASSED! Found {len(results)} result(s).")
                first = results[0]
                title = first.get('title', 'Unknown Title') if isinstance(first, dict) else getattr(first, 'title', 'Unknown Title')
                print(f"   Sample: {title[:60]}...")
            else:
                print(f"⚠️ {provider_name} PASSED but returned no results for query '{query}'.")
    except Exception as e:
         print(f"❌ {provider_name} FAILED with Exception: {e}")


async def main():
    print("========================================")
    print("      Search API Keys Verifier          ")
    print("========================================")
    
    # Check env vars first
    tavily_key = os.getenv("TAVILY_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    news_key = os.getenv("NEWSAPI_KEY")
    
    print("\n[Environment Variables]")
    print(f"TAVILY_API_KEY  : {'✅ Set' if tavily_key else '❌ MISSING'}")
    print(f"SERPER_API_KEY  : {'✅ Set' if serper_key else '❌ MISSING'}")
    print(f"NEWSAPI_KEY     : {'✅ Set' if news_key else '❌ MISSING'}")
    print("Note: Wikipedia and ArXiv do not require API keys.\n")

    # Initialize providers
    tavily = TavilySearchProvider()
    serper = SerperSearchProvider()
    arxiv = ArxivSearchProvider()
    wiki = WikipediaSearchProvider()
    news = NewsSearchProvider()
    
    # Run tests sequentially
    await test_provider("Tavily", tavily)
    await test_provider("Serper (Google)", serper)
    await test_provider("News API", news)
    await test_provider("ArXiv", arxiv, query="quantum computing")
    await test_provider("Wikipedia", wiki)
    
    print("\n========================================")
    print("            Verification Complete       ")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
