from langchain_openai import ChatOpenAI
import os
import asyncio
import json
import re
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send

from .state import (
    ReportState,
    SectionState,
    Section,
    Sections,
    Queries,
    SearchQuery,
    QueryAnalysisAndHyDE,
    SearchRoute,
    SearchRoutes,
)
from .models import get_cheap_llm, get_mid_llm, get_premium_llm
from .search import (
    TavilySearchProvider,
    SerperSearchProvider,
    ArxivSearchProvider,
    WikipediaSearchProvider,
    NewsSearchProvider,
)
from .search.merger import ResultMerger, format_ranked_results
from .search.jina import JinaReader
from .prompts import (
    DEFAULT_REPORT_STRUCTURE,
    REPORT_PLAN_QUERY_GENERATOR_PROMPT,
    REPORT_PLAN_SECTION_GENERATOR_PROMPT,
    REPORT_SECTION_QUERY_GENERATOR_PROMPT,
    SECTION_WRITER_PROMPT,
    FINAL_SECTION_WRITER_PROMPT,
    QUERY_ANALYZER_AND_HYDE_PROMPT,
    SEARCH_ROUTER_PROMPT,
)
from .cache.redis_cache import SemanticCache
from .embeddings import embed_text

# Singleton instances
_semantic_cache = SemanticCache()

# Singleton search providers
_tavily = TavilySearchProvider()
_serper = SerperSearchProvider()
_arxiv = ArxivSearchProvider()
_wikipedia = WikipediaSearchProvider()
_news = NewsSearchProvider()
_merger = ResultMerger()
_jina = JinaReader()

# Rate limiting semaphore for external search providers (prevents HTTP 429s)
_search_api_semaphore = asyncio.Semaphore(4)

async def _rate_limited_search(coro):
    async with _search_api_semaphore:
        return await coro

import warnings

# Suppress annoying Pydantic V2 serialization warnings from LangChain/OpenRouter
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# ============================================================
# Node: Query Analyzer + HyDE Generator — v2.0
# ============================================================


async def query_analyzer_hyde(state: ReportState):
    """
    First node in the pipeline. Analyzes query intent, generates a
    HyDE (Hypothetical Document Embedding) anchor, and checks Redis cache.

    If cache hit: short-circuits with cached report.
    If cache miss: stores HyDE document in state for downstream search.
    """

    topic = state["topic"]
    print("--- Query Analyzer + HyDE Generation ---")

    # Step 1: Check semantic cache
    try:
        cached = await _semantic_cache.check_cache(topic)
        if cached:
            print("--- Cache HIT — returning cached report ---")
            return {
                "cache_hit": True,
                "final_report": cached.get("content", ""),
            }
    except Exception as e:
        print(f"[Cache] Error checking cache: {e}")

    # Step 2: Fused Query Analysis and HyDE Generation
    structured_llm = get_mid_llm().with_structured_output(QueryAnalysisAndHyDE)
    prompt = QUERY_ANALYZER_AND_HYDE_PROMPT.format(topic=topic)

    try:
        result = await structured_llm.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content="Analyze this research topic and generate the HyDE document."
                ),
            ]
        )
        domain = result.domain
        hyde_document = result.hyde_document
        print(f"--- Fused Analysis & HyDE Complete ({len(hyde_document)} chars) ---")
    except Exception as e:
        print(f"[Query Analyzer] Failed structured output: {e}")
        domain = ""
        hyde_document = topic

    # Step 3: Pre-compute HyDE embedding ONCE so all sections reuse it (P4)
    hyde_embedding = None
    try:
        hyde_embedding = await embed_text(hyde_document)
        print(f"--- HyDE Embedding Pre-computed ({len(hyde_embedding) if hyde_embedding else 0} dims) ---")
    except Exception as e:
        print(f"[Query Analyzer] HyDE embedding failed: {e}")

    return {
        "hyde_document": hyde_document,
        "hyde_embedding": hyde_embedding,
        "cache_hit": False,
        "domain": domain,
    }


# ============================================================
# Conditional Edge: Route After HyDE (cache check)
# ============================================================


def route_after_hyde(state: ReportState) -> str:
    """
    Route based on cache hit:
    - cache_hit=True → skip to compile_final_report
    - cache_hit=False → proceed to generate_report_plan
    """
    if state.get("cache_hit", False):
        return "compile_final_report"
    return "generate_report_plan"


# ============================================================
# Node: Generate Report Plan
# ============================================================


async def generate_report_plan(state: ReportState):
    """Generate the overall plan for building the report."""

    topic = state["topic"]
    depth = state.get("depth", "deep")
    print(f"--- Generating Report Plan (Depth: {depth}) ---")

    report_structure = DEFAULT_REPORT_STRUCTURE
    number_of_queries = 3

    structured_llm = get_cheap_llm().with_structured_output(Queries)

    system_instructions_query = REPORT_PLAN_QUERY_GENERATOR_PROMPT.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
    )

    try:
        # Generate queries (P1: was .invoke → now .ainvoke)
        results = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions_query),
                HumanMessage(
                    content="Generate search queries that will help with planning the sections of the report."
                ),
            ]
        )

        # Convert SearchQuery objects to strings
        query_list = [
            query.search_query if isinstance(query, SearchQuery) else str(query)
            for query in results.queries
        ]

        # Multi-source search for planning context — P2: all queries run in PARALLEL
        async def _plan_search_one(query: str):
            """Run Tavily + Serper for a single planning query in parallel."""
            tasks = [
                _tavily.search(query, num_results=2),
                _serper.search(query, num_results=2),
            ]
            pair_results = await asyncio.gather(*tasks, return_exceptions=True)
            flat = []
            for r in pair_results:
                if isinstance(r, list):
                    flat.extend(r)
            return flat

        # Fire all 3 queries simultaneously instead of sequentially
        batch = await asyncio.gather(
            *[_plan_search_one(q) for q in query_list[:3]],
            return_exceptions=True,
        )

        all_results = []
        for group in batch:
            if isinstance(group, list):
                all_results.extend(group)

        # Cap results to prevent memory bloat
        if len(all_results) > 20:
            all_results = all_results[:20]

        if not all_results:
            print("Warning: No search results returned")
            search_context = "No search results available."
        else:
            ranked = await _merger.merge_and_rank(
                all_results, top_k=8
            )  # Reduced from 10
            search_context = format_ranked_results(
                ranked, max_tokens=6000
            )  # Reduced from 8000

        # Generate sections
        system_instructions_sections = REPORT_PLAN_SECTION_GENERATOR_PROMPT.format(
            topic=topic,
            report_organization=report_structure,
            search_context=search_context,
            depth=depth,
        )

        structured_llm = get_cheap_llm().with_structured_output(Sections)
        report_sections = await structured_llm.ainvoke(
            [
                SystemMessage(content=system_instructions_sections),
                HumanMessage(
                    content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields."
                ),
            ]
        )

        print("--- Generating Report Plan Completed ---")
        return {"sections": report_sections.sections}

    except Exception as e:
        print(f"Error in generate_report_plan: {e}")
        return {"sections": []}


# ============================================================
# Node: Query Rewriter + Expander (per section) — v2.0
# ============================================================


async def query_rewriter_expander(state: SectionState):
    """
    Generate diverse search queries for a section using HyDE context.
    Generates angle-based variants and selects the best ones.
    Replaces the old generate_queries() node.
    """

    section = state["section"]
    # Get HyDE document from parent state if available, else use section description
    hyde_context = (
        state.get("hyde_document", section.description) or section.description
    )
    depth = state.get("depth", "deep")

    print(f"--- Rewriting Queries for Section: {section.name} (Depth: {depth}) ---")

    number_of_queries = 3
    structured_llm = get_cheap_llm().with_structured_output(Queries)

    system_instructions = REPORT_SECTION_QUERY_GENERATOR_PROMPT.format(
        section_topic=section.description,
        hyde_context=hyde_context[:2000],  # Truncate to avoid token limits
        number_of_queries=number_of_queries,
    )

    user_instruction = (
        "Generate diverse search queries from multiple angles for this section topic."
    )
    search_queries = await structured_llm.ainvoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_instruction),
        ]
    )

    final_queries = search_queries.queries[:number_of_queries]

    print(
        f"--- Query Rewriting for Section: {section.name} Complete ({len(final_queries)} queries) ---"
    )

    return {"search_queries": final_queries}


# ============================================================
# Node: Multi-Source Search (Adaptive Routing) — v2.0
# ============================================================


async def multi_source_search(state: SectionState):
    """
    Route search queries to specific providers based on domain analysis.
    Collects raw results — ranking is done by the next node (result_merger_ranker).
    """

    search_queries = state["search_queries"]
    query_strings = [
        q.search_query if isinstance(q, SearchQuery) else str(q) for q in search_queries
    ]
    section_name = state["section"].name

    print(
        f'--- Adaptive Multi-Source Search for "{section_name}" ({len(query_strings)} queries) ---'
    )

    domain = state.get("domain", "").lower()
    hyde_context = state.get("hyde_document", section_name)
    depth = state.get("depth", "deep")

    # 1. Ask LLM to route the queries
    structured_llm = get_cheap_llm().with_structured_output(SearchRoutes)

    prompt = SEARCH_ROUTER_PROMPT.format(
        domain=domain, hyde_context=hyde_context[:2000]
    )

    # Send the raw strings attached
    queries_text = "\n".join([f"- {q}" for q in query_strings])
    user_msg = f"Determine the best search providers for these queries:\n{queries_text}"

    try:
        route_results = await structured_llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=user_msg)]
        )
        routes = route_results.routes
    except Exception as e:
        print(
            f"[MultiSourceSearch] LLM Routing failed ({e}), falling back to default routing"
        )
        # Fallback: Every query gets Tavily and Wikipedia
        routes = [
            SearchRoute(
                query=q,
                use_tavily=True,
                use_wikipedia=True,
                use_arxiv=False,
                use_news=False,
            )
            for q in query_strings
        ]

    # 2. Execute the routed searches with MEMORY LIMITS
    all_results = []
    MAX_RESULTS_PER_SECTION = (
        30  # Cap total results per section to prevent memory bloat
    )

    for route in routes:
        # If we've already collected enough results, skip remaining queries
        if len(all_results) >= MAX_RESULTS_PER_SECTION:
            break

        provider_tasks = []

        # Reduce result counts for quick depth
        tavily_results = 1 if depth == "quick" else 2

        if route.use_tavily:
            provider_tasks.append(
                _rate_limited_search(_tavily.search(route.query, num_results=tavily_results))
            )

        if route.use_wikipedia:
            provider_tasks.append(_rate_limited_search(_wikipedia.search(route.query, num_results=1)))

        if route.use_news:
            provider_tasks.append(_rate_limited_search(_news.search(route.query, num_results=1)))

        if route.use_arxiv:
            provider_tasks.append(_rate_limited_search(_arxiv.search(route.query, num_results=1)))

        # Fallback if LLM predicted NO providers:
        if not provider_tasks:
            provider_tasks.append(_rate_limited_search(_tavily.search(route.query, num_results=1)))

        results = await asyncio.gather(*provider_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(
                    f'[MultiSourceSearch] Provider error for query "{route.query}": {result}'
                )
                continue
            if isinstance(result, list):
                all_results.extend(result)
                # Keep only the cap
                if len(all_results) > MAX_RESULTS_PER_SECTION:
                    all_results = all_results[:MAX_RESULTS_PER_SECTION]

    print(
        f'--- Collected {len(all_results)} raw results (capped) for "{section_name}" ---'
    )

    # Store raw results in state — merging/ranking happens in result_merger_ranker
    raw_dicts = []
    for r in all_results:
        if hasattr(r, "to_dict"):
            raw_dicts.append(r.to_dict())
        elif isinstance(r, dict):
            raw_dicts.append(r)

    # Clear raw results from memory
    del all_results

    return {
        "search_results": raw_dicts,
    }


# ============================================================
# Node: Result Merger + Ranker (per section) — v2.0
# ============================================================


async def result_merger_ranker(state: SectionState):
    """
    Merge, deduplicate, and rank search results by credibility.
    Converts raw search results into formatted source context for the writer.
    """

    section_name = state["section"].name
    raw_results = state.get("search_results", [])

    print(
        f'--- Result Merger & Ranker for "{section_name}" ({len(raw_results)} raw results) ---'
    )

    if not raw_results:
        print(f'--- No results to rank for "{section_name}" ---')
        return {"source_str": "No search results available.", "search_results": []}

    # Merge, deduplicate, and rank using the ResultMerger
    # P4: use pre-computed HyDE embedding from state (avoids redundant embed_text API call)
    ranked_sources = await _merger.merge_and_rank(
        raw_results,
        hyde_document=state["section"].description,
        hyde_embedding=state.get("hyde_embedding", None),
        top_k=10,
    )

    # --- Selective Deep Extraction (Jina Reader) ---
    # Take the top 2 absolute best URLs and fetch their full markdown content.
    # This provides the writer with deep context without crashing memory with 30+ sites.
    top_2_sources = ranked_sources[:2]
    print(f"--- Jina Deep Extraction for Top 2 Sources in \"{section_name}\" ---")
    
    extraction_tasks = [
        _jina.fetch_markdown(s.url) for s in top_2_sources
    ]
    extracted_contents = await asyncio.gather(*extraction_tasks)
    
    # Inject extracted content back into the ranked sources
    for i, content in enumerate(extracted_contents):
        if content:
            # We append the deep content to the existing snippet to give the LLM both
            ranked_sources[i].content = f"--- FULL CONTENT START ---\n{content[:15000]}\n--- FULL CONTENT END ---\n\nORIGINAL SNIPPET: {ranked_sources[i].content}"
            print(f"    [Jina] Successfully extracted {len(content)} chars for: {ranked_sources[i].url}")

    # Format for LLM context
    source_str = format_ranked_results(ranked_sources, max_tokens=25000)

    # Convert to dicts for state storage
    search_results_dicts = [s.to_dict() for s in ranked_sources]

    print(f'--- Ranked {len(ranked_sources)} results for "{section_name}" ---')

    return {
        "source_str": source_str,
        "search_results": search_results_dicts,
    }


# ============================================================
# Node: Write Section (per section)
# ============================================================


async def write_section(state: SectionState):
    """Write a section of the report using the mid-tier LLM."""

    section = state["section"]
    source_str = state["source_str"]

    print(f"--- Writing Section: {section.name} ---")

    system_instructions = SECTION_WRITER_PROMPT.format(
        section_title=section.name,
        section_topic=section.description,
        context=source_str,
    )

    writer_llm = get_mid_llm().bind(
        presence_penalty=0.1,
    )

    user_instruction = "Generate a report section based on the provided sources."
    section_content = await writer_llm.ainvoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_instruction),
        ]
    )

    section.content = section_content.content

    print(f"--- Writing Section: {section.name} Completed ---")

    sources = state.get("search_results", [])

    return {"completed_sections": [section], "sources": sources}


# ============================================================
# Node: Parallelize Section Writing (fan-out)
# ============================================================


def parallelize_section_writing(state: ReportState):
    """Fan-out: kick off section builders in parallel for research sections."""

    hyde_doc = state.get("hyde_document", "")
    hyde_embedding = state.get("hyde_embedding", None)  # P4: pre-computed vector
    domain = state.get("domain", "")
    depth = state.get("depth", "deep")  # read directly from ReportState (bug fix)

    return [
        Send(
            "section_builder_with_web_search",
            {
                "section": s,
                "hyde_document": hyde_doc,
                "hyde_embedding": hyde_embedding,
                "domain": domain,
                "depth": depth,
            },
        )
        for s in state["sections"]
        if s.research
    ]


# ============================================================
# Utility: Format Sections
# ============================================================


def format_sections(sections: list[Section]) -> str:
    """Format a list of report sections into a single text string."""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research:
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str


# ============================================================
# Node: Aggregator + Deduplicator — v2.0
# ============================================================


def aggregator_deduplicator(state: ReportState):
    """
    Aggregate completed sections, deduplicate cross-section sources,
    and compile source metadata for the final report.

    Replaces the old format_completed_sections node.
    """

    print("--- Aggregator + Deduplicator ---")
    completed_sections = state.get("completed_sections", [])

    if not completed_sections:
        print("--- Aggregator: No sections to aggregate ---")
        return {"report_sections_from_research": ""}

    # Format sections as context for the synthesis writer
    completed_report_sections = format_sections(completed_sections)

    # Aggregate and deduplicate sources across all sections
    all_sources = state.get("sources", []) or []
    seen_urls = set()
    deduped_sources = []
    for src in all_sources:
        url = src.get("url", "") if isinstance(src, dict) else getattr(src, "url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped_sources.append(src)
        elif not url:
            deduped_sources.append(src)

    print(
        f"--- Aggregator Complete: {len(completed_sections)} sections, "
        f"{len(deduped_sources)} unique sources (from {len(all_sources)} total) ---"
    )

    return {
        "report_sections_from_research": completed_report_sections,
        "sources": deduped_sources,
    }


# ============================================================
# Node: Final Synthesis Writer — v2.0 (Premium LLM)
# ============================================================


async def final_synthesis_writer(state: ReportState):
    """
    The ONE premium LLM call. Uses Claude Sonnet 3.5 to synthesize
    all research sections into a cohesive, polished final report.

    Receives: approved sections, aggregated research context.
    Writes: Executive Summary, Introduction, full narrative, Conclusion.
    Replaces: write_final_sections + compile_final_report.
    """

    sections = state["sections"]
    completed_sections = {
        s.name: s.content for s in state.get("completed_sections", [])
    }
    report_content = state.get("report_sections_from_research", "")

    print("--- Final Synthesis Writer (Premium LLM) ---")

    # Build section content map
    research_sections = []
    non_research_sections = []
    for section in sections:
        if section.research:
            content = completed_sections.get(section.name, section.content or "")
            research_sections.append(f"## {section.name}\n{content}")
        else:
            non_research_sections.append(section)

    # Build the synthesis prompt
    from .prompts import FINAL_SYNTHESIS_PROMPT

    all_research_content = "\n\n".join(research_sections)

    non_research_names = [s.name for s in non_research_sections]
    topic = state.get("topic", "Research Report")

    system_instructions = FINAL_SYNTHESIS_PROMPT.format(
        topic=topic,
        research_sections=all_research_content[:80000],
        non_research_section_names=", ".join(non_research_names),
    )

    try:
        response = await get_premium_llm().ainvoke(
            [
                SystemMessage(content=system_instructions),
                HumanMessage(content="Synthesize the complete final report."),
            ]
        )

        final_report = response.content

        # Escape unescaped $ symbols for Markdown rendering
        sentinel = str(uuid.uuid4())
        final_report = final_report.replace("\\$", sentinel)
        final_report = final_report.replace("$", "\\$")
        final_report = final_report.replace(sentinel, "\\$")

        print(f"--- Final Synthesis Complete ({len(final_report)} chars) ---")

        return {"final_report": final_report}

    except Exception as e:
        print(f"Error in final_synthesis_writer: {e}")
        # Fallback: basic concatenation
        all_sections = "\n\n".join(
            completed_sections.get(s.name, s.content or "") for s in sections
        )
        return {"final_report": all_sections}
