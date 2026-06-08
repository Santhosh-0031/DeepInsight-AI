from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import ReportState, ReportStateInput, ReportStateOutput, SectionState, SectionOutputState
from .nodes import (
    query_analyzer_hyde,
    route_after_hyde,
    generate_report_plan,
    query_rewriter_expander,
    multi_source_search,
    result_merger_ranker,
    write_section,
    parallelize_section_writing,
    aggregator_deduplicator,
    final_synthesis_writer,
)
from .output_compiler import output_compiler_node


def create_section_builder_subagent():
    """
    Create the section builder subagent with a reflection loop.

    Flow:
        START → query_rewriter_expander → multi_source_search → result_merger_ranker
              → write_section → END
    """
    section_builder = StateGraph(SectionState, output=SectionOutputState)

    section_builder.add_node("query_rewriter_expander", query_rewriter_expander)
    section_builder.add_node("multi_source_search", multi_source_search)
    section_builder.add_node("result_merger_ranker", result_merger_ranker)
    section_builder.add_node("write_section", write_section)

    section_builder.add_edge(START, "query_rewriter_expander")
    section_builder.add_edge("query_rewriter_expander", "multi_source_search")
    section_builder.add_edge("multi_source_search", "result_merger_ranker")
    section_builder.add_edge("result_merger_ranker", "write_section")
    section_builder.add_edge("write_section", END)

    return section_builder.compile()


def create_reporter_agent():
    """
    Create the main reporter agent graph — v2.0 final structure.

    Flow:
        START → query_analyzer_hyde
             → [cache hit?] → output_compiler (short-circuit)
             → [cache miss] → generate_report_plan
             → [fan-out] section_builder (×N sections in parallel)
             → aggregator_deduplicator
             → final_synthesis_writer (Claude Sonnet — 1 call)
             → output_compiler → END
    """
    section_builder_subagent = create_section_builder_subagent()

    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)

    builder.add_node("query_analyzer_hyde", query_analyzer_hyde)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("aggregator_deduplicator", aggregator_deduplicator)
    builder.add_node("final_synthesis_writer", final_synthesis_writer)
    builder.add_node("output_compiler", output_compiler_node)

    # START → HyDE analyzer
    builder.add_edge(START, "query_analyzer_hyde")

    # HyDE → cache check: short-circuit if cached
    builder.add_conditional_edges(
        "query_analyzer_hyde",
        route_after_hyde,
        {
            "generate_report_plan": "generate_report_plan",
            "compile_final_report": "output_compiler",
        }
    )

    # Normal flow
    builder.add_conditional_edges("generate_report_plan",
                                  parallelize_section_writing,
                                  ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "aggregator_deduplicator")
    builder.add_edge("aggregator_deduplicator", "final_synthesis_writer")
    builder.add_edge("final_synthesis_writer", "output_compiler")
    builder.add_edge("output_compiler", END)

    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["generate_report_plan"]
    )


reporter_agent = create_reporter_agent()
