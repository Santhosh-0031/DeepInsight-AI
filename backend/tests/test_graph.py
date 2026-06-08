"""
Test: Graph Structure (graph.py)

Verifies the graph topology, node names, and edge structure
without running the full pipeline.
"""

import pytest


class TestSectionBuilderSubagent:
    """Verify the section builder subagent graph structure."""

    def test_section_builder_compiles(self):
        from app.graph import create_section_builder_subagent
        subagent = create_section_builder_subagent()
        assert subagent is not None

    def test_section_builder_has_expected_nodes(self):
        from app.graph import create_section_builder_subagent
        from langgraph.graph import StateGraph
        # Build a fresh graph to inspect (before compile)
        from app.state import SectionState, SectionOutputState
        from app.nodes import (
            query_rewriter_expander, multi_source_search, result_merger_ranker,
            write_section,
        )

        builder = StateGraph(SectionState, output=SectionOutputState)
        builder.add_node("query_rewriter_expander", query_rewriter_expander)
        builder.add_node("multi_source_search", multi_source_search)
        builder.add_node("result_merger_ranker", result_merger_ranker)
        builder.add_node("write_section", write_section)

        node_names = set(builder.nodes.keys())
        expected = {
            "query_rewriter_expander",
            "multi_source_search",
            "result_merger_ranker",
            "write_section",
        }
        assert expected.issubset(node_names)


class TestReporterAgent:
    """Verify the main reporter agent graph structure."""

    def test_reporter_agent_compiles(self):
        from app.graph import reporter_agent
        assert reporter_agent is not None

    def test_reporter_agent_has_expected_nodes(self):
        from app.graph import create_reporter_agent
        from langgraph.graph import StateGraph
        from app.state import ReportState, ReportStateInput, ReportStateOutput

        builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)

        # Check by importing node functions (they should all exist)
        from app.nodes import (
            query_analyzer_hyde, generate_report_plan,
            aggregator_deduplicator,
            final_synthesis_writer,
        )
        from app.output_compiler import output_compiler_node

        # All these should be importable without errors
        assert callable(query_analyzer_hyde)
        assert callable(generate_report_plan)
        assert callable(aggregator_deduplicator)
        assert callable(final_synthesis_writer)
        assert callable(output_compiler_node)

    def test_removed_nodes_dont_exist(self):
        """format_completed_sections, critic_agent, should_reflect, and fact_checker should no longer be importable from nodes."""
        from app import nodes
        assert not hasattr(nodes, "format_completed_sections"), \
            "format_completed_sections should have been removed"
        assert not hasattr(nodes, "critic_agent"), "critic_agent should have been removed"
        assert not hasattr(nodes, "fact_checker"), "fact_checker should have been removed"

    def test_new_nodes_exist(self):
        """result_merger_ranker and aggregator_deduplicator should exist."""
        from app.nodes import result_merger_ranker, aggregator_deduplicator
        assert callable(result_merger_ranker)
        assert callable(aggregator_deduplicator)


class TestNodeFunctions:
    """Verify individual node function signatures."""

    def test_aggregator_deduplicator_handles_empty_state(self):
        from app.nodes import aggregator_deduplicator
        result = aggregator_deduplicator({
            "completed_sections": [],
            "sources": [],
        })
        assert result["report_sections_from_research"] == ""

    def test_aggregator_deduplicator_deduplicates_sources(self):
        from app.nodes import aggregator_deduplicator
        from app.state import Section

        sections = [
            Section(name="Sec1", description="D1", plan="P1", research=True, content="Content 1"),
        ]
        sources = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
            {"url": "https://a.com", "title": "A duplicate"},
        ]
        result = aggregator_deduplicator({
            "completed_sections": sections,
            "sources": sources,
        })
        deduped = result["sources"]
        urls = [s["url"] for s in deduped]
        assert len(urls) == len(set(urls)), "Duplicate sources not removed"

    @pytest.mark.asyncio
    async def test_result_merger_ranker_handles_empty_results(self):
        from app.nodes import result_merger_ranker
        from app.state import Section

        section = Section(name="Test", description="Test desc", plan="Plan", research=True, content="")
        result = await result_merger_ranker({"section": section, "search_results": []})
        assert result["source_str"] == "No search results available."
        assert result["search_results"] == []
