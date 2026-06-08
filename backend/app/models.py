"""
Hierarchical LLM Model Factory.
Provides 3 tiers of LLMs via OpenRouter for cost-optimized routing.

Tier allocation:
  - CHEAP  (Gemini Flash 2.0): Planning, query rewriting, reflection, fact-check
  - MID    (Claude Haiku 3.5):  Section writing, follow-up chat
  - PREMIUM (Claude Sonnet 3.5): Final synthesis (ONE call only)
"""

import os
from langchain_openai import ChatOpenAI


# Singleton instances for each tier
_cheap_llm = None
_mid_llm = None
_premium_llm = None


def _create_llm(model_env_var: str, temperature: float = 0, timeout: int = 60) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for OpenRouter."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv(model_env_var),
        temperature=temperature,
        max_retries=2,
        timeout=timeout,
        default_headers={
            "HTTP-Referer": "https://github.com/abhigupta/deep_research_agent",
            "X-Title": "Deep Research Agent v2",
        },
        extra_body={
            "transforms": ["middle-out"],  # OpenRouter auto-compression
        },
    )


def get_cheap_llm() -> ChatOpenAI:
    """
    Get the cheap-tier LLM.
    Used for: query analysis, HyDE, report planning, query rewriting,
    critic/reflection, fact-checking.
    Cost: ~$0.0002–$0.0005 per call.
    """
    global _cheap_llm
    if _cheap_llm is None:
        _cheap_llm = _create_llm("LLM_MODEL_CHEAP", temperature=0)
    return _cheap_llm


def get_mid_llm() -> ChatOpenAI:
    """
    Get the mid-tier LLM.
    Used for: section writing, follow-up chat.
    Cost: ~$0.001 per call.
    """
    global _mid_llm
    if _mid_llm is None:
        _mid_llm = _create_llm("LLM_MODEL_MID", temperature=0.3)
    return _mid_llm


def get_premium_llm() -> ChatOpenAI:
    """
    Get the premium-tier LLM.
    Used for: final synthesis writer — ONE call per query.
    Cost: ~$0.015 per call.
    """
    global _premium_llm
    if _premium_llm is None:
        _premium_llm = _create_llm("LLM_MODEL_PREMIUM", temperature=0.6, timeout=120)
    return _premium_llm
