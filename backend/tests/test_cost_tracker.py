"""
Test: Cost Tracker (cost_tracker.py)

Verifies per-call cost tracking, aggregation, LangSmith metadata,
and static run estimation.
"""

import pytest
from app.cost_tracker import CostTracker, MODEL_PRICING


class TestCostTracker:

    def test_track_single_call(self):
        tracker = CostTracker()
        cost = tracker.track_call("mid", input_tokens=1000, output_tokens=500, node_name="write_section")
        
        expected = (1000 / 1000) * MODEL_PRICING["mid"]["input_per_1k"] + (500 / 1000) * MODEL_PRICING["mid"]["output_per_1k"]
        assert abs(cost - expected) < 1e-8

    def test_track_cheap_call(self):
        tracker = CostTracker()
        cost = tracker.track_call("cheap", input_tokens=5000, output_tokens=3000)
        
        expected = (5000 / 1000) * MODEL_PRICING["cheap"]["input_per_1k"] + (3000 / 1000) * MODEL_PRICING["cheap"]["output_per_1k"]
        assert abs(cost - expected) < 1e-8

    def test_track_premium_call(self):
        tracker = CostTracker()
        cost = tracker.track_call("premium", input_tokens=10000, output_tokens=5000)
        
        expected = (10000 / 1000) * MODEL_PRICING["premium"]["input_per_1k"] + (5000 / 1000) * MODEL_PRICING["premium"]["output_per_1k"]
        assert abs(cost - expected) < 1e-8

    def test_get_summary_empty(self):
        tracker = CostTracker()
        summary = tracker.get_summary()
        assert summary["total_cost_usd"] == 0
        assert summary["call_count"] == 0
        assert summary["total_tokens"] == 0

    def test_get_summary_multiple_calls(self):
        tracker = CostTracker()
        tracker.track_call("cheap", 500, 300, "query_analyzer")
        tracker.track_call("mid", 2000, 1500, "write_section")
        tracker.track_call("premium", 10000, 5000, "synthesis")

        summary = tracker.get_summary()
        assert summary["call_count"] == 3
        assert summary["total_input_tokens"] == 500 + 2000 + 10000
        assert summary["total_output_tokens"] == 300 + 1500 + 5000
        assert summary["total_tokens"] == 19300
        assert summary["total_cost_usd"] > 0

    def test_get_summary_calls_by_tier(self):
        tracker = CostTracker()
        tracker.track_call("cheap", 100, 50)
        tracker.track_call("cheap", 200, 100)
        tracker.track_call("mid", 1000, 500)

        summary = tracker.get_summary()
        assert "cheap" in summary["calls_by_tier"]
        assert summary["calls_by_tier"]["cheap"]["call_count"] == 2
        assert "mid" in summary["calls_by_tier"]
        assert summary["calls_by_tier"]["mid"]["call_count"] == 1
        assert "premium" not in summary["calls_by_tier"]

    def test_to_langsmith_metadata(self):
        tracker = CostTracker()
        tracker.track_call("mid", 1000, 500, "test")
        meta = tracker.to_langsmith_metadata()

        assert "cost_estimate_usd" in meta
        assert "total_tokens" in meta
        assert "llm_call_count" in meta
        assert meta["llm_call_count"] == 1
        assert meta["total_tokens"] == 1500

    def test_unknown_tier_defaults_to_cheap(self):
        tracker = CostTracker()
        cost = tracker.track_call("unknown_tier", 1000, 500)
        
        # Unknown falls back to cheap pricing
        expected = (1000 / 1000) * MODEL_PRICING["cheap"]["input_per_1k"] + (500 / 1000) * MODEL_PRICING["cheap"]["output_per_1k"]
        assert abs(cost - expected) < 1e-8


class TestStaticEstimate:

    def test_estimate_run_cost_basic(self):
        cost = CostTracker.estimate_run_cost(section_count=5)
        assert cost > 0
        assert isinstance(cost, float)

    def test_more_sections_cost_more(self):
        cost_3 = CostTracker.estimate_run_cost(3)
        cost_8 = CostTracker.estimate_run_cost(8)
        assert cost_8 > cost_3

    def test_estimate_zero_sections(self):
        cost = CostTracker.estimate_run_cost(0)
        # Should still have premium cost (1 synthesis call)
        assert cost > 0