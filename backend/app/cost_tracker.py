"""
Cost Tracker — Phase 11.

Tracks estimated LLM costs per research run based on model tier and token counts.
Aggregates total cost and formats metadata for LangSmith and SSE streaming.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


# Per-model pricing (USD per 1K tokens) — OpenRouter rates as of 2025
MODEL_PRICING = {
    "cheap": {
        "label": "GPT-5 Nano",
        "input_per_1k": 0.00005,
        "output_per_1k": 0.0004,
    },
    "mid": {
        "label": "Gemini 2.5 Flash",
        "input_per_1k": 0.0003,
        "output_per_1k": 0.0025,
    },
    "premium": {
        "label": "Gemini 3 Flash Preview",
        "input_per_1k": 0.0005,
        "output_per_1k": 0.0030,
    },
}


@dataclass
class CallRecord:
    """Record of a single LLM call."""
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    node_name: str = ""


class CostTracker:
    """
    Accumulates LLM call costs across a single research pipeline run.

    Usage:
        tracker = CostTracker()
        tracker.track_call("cheap", input_tokens=500, output_tokens=200, node_name="query_analyzer")
        tracker.track_call("premium", input_tokens=8000, output_tokens=4000, node_name="final_synthesis")
        summary = tracker.get_summary()
    """

    def __init__(self):
        self._calls: List[CallRecord] = []

    def track_call(
        self,
        tier: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        node_name: str = "",
    ) -> float:
        """
        Record a single LLM call and return its cost.

        Args:
            tier: Model tier — 'cheap', 'mid', or 'premium'.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            node_name: Name of the graph node that made the call.

        Returns:
            Cost of this call in USD.
        """
        pricing = MODEL_PRICING.get(tier, MODEL_PRICING["cheap"])
        cost = (
            (input_tokens / 1000) * pricing["input_per_1k"]
            + (output_tokens / 1000) * pricing["output_per_1k"]
        )

        record = CallRecord(
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            node_name=node_name,
        )
        self._calls.append(record)
        return cost

    def get_summary(self) -> Dict[str, Any]:
        """
        Get an aggregated cost summary for the entire run.

        Returns:
            Dict with total_cost, calls_by_tier, total_tokens, call_count.
        """
        total_cost = sum(c.cost_usd for c in self._calls)
        total_input = sum(c.input_tokens for c in self._calls)
        total_output = sum(c.output_tokens for c in self._calls)

        calls_by_tier: Dict[str, Dict[str, Any]] = {}
        for tier_key in MODEL_PRICING:
            tier_calls = [c for c in self._calls if c.tier == tier_key]
            if tier_calls:
                calls_by_tier[tier_key] = {
                    "label": MODEL_PRICING[tier_key]["label"],
                    "call_count": len(tier_calls),
                    "input_tokens": sum(c.input_tokens for c in tier_calls),
                    "output_tokens": sum(c.output_tokens for c in tier_calls),
                    "cost_usd": round(sum(c.cost_usd for c in tier_calls), 6),
                }

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "call_count": len(self._calls),
            "calls_by_tier": calls_by_tier,
        }

    def to_langsmith_metadata(self) -> Dict[str, Any]:
        """Format cost data for LangSmith run metadata."""
        summary = self.get_summary()
        return {
            "cost_estimate_usd": summary["total_cost_usd"],
            "total_tokens": summary["total_tokens"],
            "llm_call_count": summary["call_count"],
            "cost_by_tier": {
                tier: data["cost_usd"]
                for tier, data in summary["calls_by_tier"].items()
            },
        }

    @staticmethod
    def estimate_run_cost(section_count: int) -> float:
        """
        Quick static estimate for a full research run without tracking individual calls.

        Based on typical token usage patterns:
          - Cheap: ~500 input + 300 output per call × (2 + 4 × sections)
          - Mid: ~2000 input + 1500 output per section
          - Premium: ~10000 input + 5000 output × 1
        """
        cheap_calls = 2 + section_count * 4
        cheap_cost = cheap_calls * (
            (500 / 1000) * MODEL_PRICING["cheap"]["input_per_1k"]
            + (300 / 1000) * MODEL_PRICING["cheap"]["output_per_1k"]
        )

        mid_cost = section_count * (
            (2000 / 1000) * MODEL_PRICING["mid"]["input_per_1k"]
            + (1500 / 1000) * MODEL_PRICING["mid"]["output_per_1k"]
        )

        premium_cost = (
            (10000 / 1000) * MODEL_PRICING["premium"]["input_per_1k"]
            + (5000 / 1000) * MODEL_PRICING["premium"]["output_per_1k"]
        )

        return round(cheap_cost + mid_cost + premium_cost, 4)
