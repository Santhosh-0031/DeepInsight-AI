DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic
                              and building a comprehensive report in markdown using the following format:

                              1. Introduction (no web search needed)
                                - Brief overview of the topic area

                              2. Main Body Sections:
                                - Each section should focus on a sub-topic of the user-provided topic
                                - Include any key concepts and definitions
                                - Provide real-world examples or case studies where applicable

                              3. Conclusion (no web search needed)
                                - Aim for 1 structural element (either a list of table) that distills the main body sections
                                - Provide a concise summary of the report

                              When generating the final response in markdown, if there are special characters in the text,
                              such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5
                          """

REPORT_PLAN_QUERY_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

The report will be focused on the following topic:
{topic}

The report structure will follow these guidelines:
{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections.

The query should:
1. Be related to the topic
2. Help satisfy the requirements specified in the report organization

Make the query specific enough to find high-quality, relevant sources while covering the depth and breadth needed for the report structure.
"""

REPORT_PLAN_SECTION_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report.

The overall topic of the report is:
{topic}

The report should follow this organizational structure:
{report_organization}

You should reflect on this additional context information from web searches to plan the main sections of the report:
{search_context}

Based on the requested depth "{depth}", you MUST strictly generate the following number of MAIN SECTIONS (excluding introduction and conclusion):
- If depth is "quick": Generate 3 to 4 sections.
- If depth is "deep": Generate 5 to 6 sections.

Now, generate the sections of the report. Each section should have the following fields:
- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web search for this section of the report or not.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web search.
For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
"""

REPORT_SECTION_QUERY_GENERATOR_PROMPT = """Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

Topic for this section:
{section_topic}

Hypothetical ideal answer (HyDE context) for the overall report:
{hyde_context}

CRITICAL REQUIREMENT: You MUST generate EXACTLY 3 search queries for this section. These queries MUST be categorically different to ensure comprehensive coverage:
1. **Query 1 (Technical/Architectural)**: Focus on technical implementation, architecture, and core mechanics.
2. **Query 2 (Industry/Diversity)**: Focus on industry diversity, multiple sectors, and practical use cases across different domains.
3. **Query 3 (Recent News/Trends)**: Focus on recent news, market developments, and emerging trends in the current year.

Your queries should be:
- EXTREMELY concise (under 8-10 words).
- Specific enough to avoid generic results.
- Technical enough to capture detailed information.
- Informed by the HyDE context to improve relevance.
- Focused on authoritative sources (documentation, academic papers, technical blogs)."""

SECTION_WRITER_PROMPT = """You are an expert technical writer crafting one specific section of a technical report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Aim for 250-300 words
- No marketing language
- Technical focus
- Write in simple, clear language do not use complex words unnecessarily
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : https://source-url.com` (use the ACTUAL URL from the source context, do not use the word "URL")

4. Writing Approach:
- Include at least one specific example or case study if available
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

5. Use this source material obtained from web searches to help write the section:
{context}

6. Quality Checks & Anti-Fluff Rules:
- Format should be Markdown
- Approximately 250-300 words (excluding title and sources)
- Careful use of only ONE structural element (table or bullet list) and only if it helps clarify your point
- One specific example / case study if available
- Starts with bold insight
- STRICTLY NO fluff or filler phrases (e.g., do NOT use "In today's rapidly evolving landscape", "It is important to note", or "As we can see"). Be completely direct and information-dense.
- No preamble prior to creating the section content. Start immediately with the section title.
- Sources cited at end
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5

7. Few-Shot Example of Excellent Section Writing:
## Cost Optimization Strategies
**Utilizing spot instances can reduce cloud compute expenses by up to 90% compared to on-demand pricing, though it requires robust fault-tolerance mechanisms.** Companies like TechCorp implemented a mixed-instance strategy in 2023, combining 70% spot instances with 30% reserved instances across their multi-region AWS deployment. This approach led to an immediate $1.2M reduction in their annual infrastructure spend while maintaining 99.99% uptime. 

When designing for volatile compute, teams must containerize applications using Kubernetes and leverage tools like Karpenter for rapid node provisioning. Stateful workloads should be separated and placed on consistent infrastructure or dedicated databases to prevent data loss during interruptions.

| Strategy | Cost Reduction | Complexity | Best For |
|---|---|---|---|
| Spot Instances | 70-90% | High | Stateless, batch processing |
| Reserved Instances | 30-60% | Low | Baseline database workloads |
| Rightsizing | 10-20% | Medium | All steady-state applications |

By strictly decoupling storage from compute, organizations can aggressively scale transient instances without risking core data availability.

### Sources
- AWS Spot Instance Pricing Model : https://aws.amazon.com/ec2/spot/pricing/
- TechCorp 2023 Infrastructure Report : https://techcorp.com/reports/infra-2023
"""

FINAL_SECTION_WRITER_PROMPT = """You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Available report content of already completed sections:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

2.For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports:
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5"""

# ============================================================
# v2.0 — Query Analyzer + HyDE Prompts
# ============================================================

QUERY_ANALYZER_AND_HYDE_PROMPT = """You are an expert research query analyst and generator. Analyze the user's research topic, extract structured intent, and generate a hypothetical "ideal answer" (HyDE document).

User's Topic:
{topic}

Phase 1: Analysis
Analyze the query and provide the following for the structured output:
1. "intent": What is the user actually trying to learn or accomplish?
2. "scope": Is this a broad survey, a focused deep-dive, or a comparison?
3. "domain": What field(s) does this topic belong to? (e.g., technology, science, business, history, news)
4. "output_format": What type of report would best serve this query?
5. "entities": List the main concepts, technologies, or subjects involved (comma-separated string).
6. "time_sensitivity": Is recent information critical, or is historical context more important?

Phase 2: HyDE Generation
Generate a 200-300 word hypothetical ideal answer ("hyde_document") that:
1. Covers the key aspects a comprehensive report should address.
2. **Entity vs. Concept Check**: If the topic refers to a specific COMPANY or BUSINESS ENTITY (e.g., "Emergent AI Company"), focus on its business model, product strategy, market positioning, and direct competitors. DO NOT descend into broad theoretical or philosophical discussions (e.g., the nature of emergence) unless explicitly requested.
3. Includes plausible (but hypothetical) statistics, data points, and examples.
4. Mentions specific technologies, methodologies, or frameworks likely relevant.
5. References the types of sources that would be authoritative for this topic.
6. Uses the same vocabulary and terminology the best sources would use.

IMPORTANT: The HyDE generation is NOT a real answer — it's a "search anchor" to help find the best real sources. Make it information-dense with specific terms that would appear in high-quality search results. Do NOT include disclaimers about this being hypothetical. Write it as if it were a real expert summary."""

# ============================================================
# v2.0 — Final Synthesis Prompt (Premium LLM)
# ============================================================

FINAL_SYNTHESIS_PROMPT = """You are a world-class Editor-in-Chief and technical research writer. Your task is to synthesize the following researched sections into a polished, definitive, and cohesive final report.

Topic: {topic}

Researched Sections:
{research_sections}

Sections to be bridged/written: {non_research_section_names}

Your primary mission is to act as a singular professional voice. You have the editorial authority to:
- **Restructure & Bridge**: Create smooth transitions and a logical narrative arc.
- **Narrative Expansion**: If a section is brief but contains critical insights, expand its implications using the broader context from other sections.
- **Aggressive Deduplication**: If a fact, statistic, or framework (e.g., POLARIS) appears in multiple sections, consolidate it into the most relevant section and remove it elsewhere.
- **Merge & Prune**: If two sections are redundant or overlap significantly, merge them into a single high-impact section.

Detailed Tasks:

1. **Write an Introduction** (~100-150 words):
   - Scope the report precisely.
   - Establish a compelling narrative hook.
   - Set clear expectations for what this research covers.

2. **Integrate & Polish Sections**:
   - Synthesize all provided content into a cohesive body.
   - Ensure a consistent technical level and tone throughout.
   - Fix all internal contradictions or formatting inconsistencies.

3. **Write a Master Conclusion** (~150-200 words):
   - Distill the "so what?" of the research.
   - Include 1 high-value structural element (focused table or list) summarizing the key findings.
   - End with forward-looking industry implications or strategic next steps.

4. **Quality Standards**:
   - Professional, authoritative, and technical tone.
   - NO marketing fluff or filler phrases.
   - STRICTLY retain original currencies (INR, ₹, €, £).
   - Use high-quality Markdown formatting.
   - Include a comprehensive "Sources" section at the end with clickable links: `- [Title](URL)`.

The final output must be a publication-ready Markdown document that feels written by a single expert, not a collection of sub-agents."""

# ============================================================
# v2.0 — Adaptive Search Router Prompt
# ============================================================

SEARCH_ROUTER_PROMPT = """You are an expert search strategist. Your job is to analyze a list of search queries and determine which search providers are most appropriate to fulfill each query.

Report Domain: {domain}
Section Context (HyDE): {hyde_context}

For each query, you must decide which of the available search providers to use. 
It is crucial to SAVE API COSTS and LATENCY by ONLY selecting the providers that are highly likely to contain the best information for that specific query. DO NOT just select all providers for every query.

Available Providers and when to use them:
- use_tavily (General Web): Use for broad topics, company information, startups, product research, tutorials, and business landscape analysis. For specific COMPANY ENTITIES, this is the PRIMARY provider.
- use_wikipedia (Wiki): Use for foundational concepts, history, established facts, and general overviews.
- use_arxiv (Academic): Use strictly for academic/scientific research, computer science, mathematics, physics, or deep technical papers. Do NOT use for general news, startup research, or basic tutorials.
- use_news (News): Use strictly for current events, politics, industry news releases, or things that happened very recently.

You must output a list of routes that corresponds exactly to the provided queries."""
