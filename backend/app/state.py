from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
from typing import Annotated, List, Optional, Literal, Dict


# ============================================================
# Pydantic Models (Structured LLM Output Schemas)
# ============================================================

class Section(BaseModel):
    """A single section of the research report."""
    name: str = Field(
        description="Name for a particular section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web search for this section of the report."
    )
    content: str = Field(
        default="",
        description="The content for this section.",
    )
    # v2.0 additions
    key_questions: List[str] = Field(
        default_factory=list,
        description="Key questions this section should answer.",
    )
    search_angle: str = Field(
        default="",
        description="The angle or focus for search queries (e.g., 'quantitative data', 'case studies').",
    )
    priority: str = Field(
        default="medium",
        description="Priority level for this section: 'high', 'medium', or 'low'.",
    )


class Sections(BaseModel):
    """Collection of all report sections."""
    sections: List[Section] = Field(
        description="All the Sections of the overall report.",
    )


class SearchQuery(BaseModel):
    """A single web search query."""
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    """Collection of search queries."""
    queries: List[SearchQuery] = Field(
        description="List of web search queries. STRICT MAXIMUM of 3 queries.",
        max_length=3
    )


class SearchRoute(BaseModel):
    """Routing instructions for a single search query."""
    query: str = Field(description="The search query being routed.")
    use_tavily: bool = Field(description="True if general web search (Tavily) is needed. Good for broad topics, companies, or recent unstructured data.")
    use_wikipedia: bool = Field(description="True if Wikipedia is needed. Good for established concepts, history, and foundational factual overviews.")
    use_arxiv: bool = Field(description="True if ArXiv is needed. Good for computer science, physics, mathematics, and academic research papers.")
    use_news: bool = Field(description="True if News API is needed. Good for current events, politics, recent business developments, or very recent topics.")


class SearchRoutes(BaseModel):
    """Collection of search routes mapping 1-to-1 with queries."""
    routes: List[SearchRoute] = Field(
        description="List of search routes corresponding to each search query.",
    )


class SourceMetadata(BaseModel):
    """Metadata for a single source result with credibility scoring."""
    url: str = Field(default="", description="Source URL.")
    domain: str = Field(default="", description="Source domain.")
    title: str = Field(default="", description="Source title.")
    credibility_score: float = Field(default=0.5, description="Domain authority score (0.0–1.0).")
    recency_score: float = Field(default=0.5, description="How recent the content is (0.0–1.0).")
    relevance_score: float = Field(default=0.5, description="Cosine similarity to HyDE doc (0.0–1.0).")
    corroboration: int = Field(default=0, description="Number of other sources making the same claim.")
    final_score: float = Field(default=0.0, description="Weighted combination of all scores.")
    publish_date: str = Field(default="", description="Publication date.")
    content: str = Field(default="", description="Source content snippet.")
    source_type: str = Field(default="", description="Provider type: tavily, serper, arxiv, wikipedia, newsapi.")


class QueryAnalysisAndHyDE(BaseModel):
    """Structured output for the fused Query Analyzer and HyDE generator."""
    intent: str = Field(description="The core intent of the user.")
    scope: str = Field(description="The scope of the report (e.g., broad survey, focused deep-dive).")
    domain: str = Field(description="The domain or field of study (e.g., technology, business, science, news, medical).")
    output_format: str = Field(description="The ideal output format.")
    entities: str = Field(description="Comma-separated major entities or concepts.")
    time_sensitivity: str = Field(description="Time sensitivity (e.g., recent, historical).")
    hyde_document: str = Field(description="A 200-300 word hypothetical ideal answer to serve as a search anchor.")


# ============================================================
# LangGraph State Schemas
# ============================================================

class ReportStateInput(TypedDict):
    topic: str   # Report topic
    depth: str   # 'quick' or 'deep'


class ReportStateOutput(TypedDict):
    final_report: str  # Final report
    output_metadata: dict  # Output compiler results (paths, URLs, scores)


class ReportState(TypedDict):
    topic: str                                              # Report topic
    depth: str                                              # 'quick' or 'deep'
    sections: list[Section]                                 # List of report sections
    completed_sections: Annotated[list, operator.add]       # Send() API — accumulated
    report_sections_from_research: str                      # Formatted completed sections as context
    final_report: str                                       # Final compiled report
    # v2.0 additions
    hyde_document: str                                      # HyDE hypothetical ideal answer
    hyde_embedding: Optional[List[float]]                   # Pre-computed HyDE embedding vector (P4)
    sub_queries: list[str]                                  # Expanded sub-queries from HyDE
    search_results: dict                                    # Per-section search results
    sources: Annotated[list, operator.add]                  # Full source metadata with credibility
    cache_hit: bool                                         # Whether result was served from cache
    langsmith_run_id: str                                   # LangSmith trace ID
    output_metadata: dict                                   # Output compiler results (paths, URLs, scores)
    domain: str                                             # Extracted domain from query analyzer


class SectionState(TypedDict):
    section: Section                                        # Report section being worked on
    search_queries: list[SearchQuery]                       # Generated search queries
    source_str: str                                         # Formatted source content for LLM
    report_sections_from_research: str                      # Context from other completed sections
    completed_sections: list[Section]                       # Accumulated for Send() API
    # v2.0 additions
    search_results: list[dict]                              # Raw search results with metadata
    hyde_document: str                                      # HyDE context passed from parent
    hyde_embedding: Optional[List[float]]                   # Pre-computed HyDE embedding (P4)
    domain: str                                             # Extracted domain from query analyzer
    depth: str                                              # Quick or Deep depth variable


class SectionOutputState(TypedDict):
    completed_sections: list[Section]                       # Final output key for Send() API
    sources: list[dict]                                     # Aggregated sources from this section
