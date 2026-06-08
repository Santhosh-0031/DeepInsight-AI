import os
import pytest
from dotenv import load_dotenv, find_dotenv

# Ensure env vars are loaded
load_dotenv(find_dotenv(), override=True)

class TestModelTiers:
    """Verify each LLM tier initializes with correct config for the 2026 stack."""

    def setup_method(self):
        """Reset singletons before each test."""
        import app.models as models
        models._cheap_llm = None
        models._mid_llm = None
        models._premium_llm = None

    def test_cheap_llm_initializes(self):
        """Test GPT-5 Nano (OpenAI Class)."""
        from app.models import get_cheap_llm
        llm = get_cheap_llm()
        assert llm is not None
        # OpenAI uses .temperature and .model
        assert llm.temperature == pytest.approx(0.0, abs=1e-6)
        
        actual_model = getattr(llm, "model", getattr(llm, "model_name", None))
        assert actual_model == os.getenv("LLM_MODEL_CHEAP")

    def test_mid_llm_initializes(self):
        """Test Gemini 2.5 Flash (Google Class)."""
        from app.models import get_mid_llm
        llm = get_mid_llm()
        assert llm is not None
        # #! FIX: LangChain Google models often store model as 'model' or 'model_name'
        # and use 'temperature'
        assert llm.temperature == pytest.approx(0.3, abs=1e-6)
        
        actual_model = getattr(llm, "model", getattr(llm, "model_name", None))
        assert actual_model == os.getenv("LLM_MODEL_MID")

    def test_premium_llm_initializes(self):
        """Test Claude 3.5 Sonnet (Anthropic Class)."""
        from app.models import get_premium_llm
        llm = get_premium_llm()
        assert llm is not None
        # Anthropic standardizes to .temperature and .model in 2026
        assert llm.temperature == pytest.approx(0.6, abs=1e-6)
        
        actual_model = getattr(llm, "model", getattr(llm, "model_name", None))
        assert actual_model == os.getenv("LLM_MODEL_PREMIUM")

    def test_singletons(self):
        """Verify singleton pattern works across all tiers."""
        from app.models import get_cheap_llm, get_mid_llm, get_premium_llm
        assert get_cheap_llm() is get_cheap_llm()
        assert get_mid_llm() is get_mid_llm()
        assert get_premium_llm() is get_premium_llm()

    def test_different_tiers_are_distinct(self):
        from app.models import get_cheap_llm, get_mid_llm, get_premium_llm
        cheap = get_cheap_llm()
        mid = get_mid_llm()
        premium = get_premium_llm()
        assert cheap is not mid
        assert mid is not premium
        assert cheap is not premium