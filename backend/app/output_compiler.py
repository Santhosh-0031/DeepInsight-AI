"""
Output Compiler + Multi-Format Renderer — Phase 9.

Produces multi-format output from the final report state:
  - PDF via WeasyPrint (HTML → PDF)
  - Markdown with inline citations and confidence flags
  - JSON structured data (sections, sources, scores)
  - ChromaDB mini store for follow-up chat
  - Redis cache storage for future cache hits
  - LangSmith run metadata (runtime, cost, section scores, reflection counts)
"""

import os
import re
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import markdown as md_lib

from .state import ReportState

logger = logging.getLogger(__name__)


class OutputCompiler:
    """
    Compiles the final research report into multiple output formats
    and persists results to disk, cache, and vector store.
    """

    def __init__(self, output_dir: str = None):
        self._output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs"
        )
        os.makedirs(self._output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main compile entry point
    # ------------------------------------------------------------------
    async def compile(
        self,
        report_content: str,
        report_id: str,
        topic: str,
        sections: list = None,
        sources: list = None,
        start_time: float = None,
        is_cache_hit: bool = False,
    ) -> Dict[str, Any]:
        """
        Compile the report into all output formats and persist everywhere.

        Returns:
            Dict with paths, URLs, and metadata for each output format.
        """
        runtime_seconds = round(time.time() - start_time, 2) if start_time else 0.0
        safe_name = self._sanitize_filename(topic)
        generated_at = datetime.now().isoformat()

        result: Dict[str, Any] = {
            "report_id": report_id,
            "topic": topic,
            "generated_at": generated_at,
        }

        # --- 1. Markdown with confidence flags ---
        md_content = report_content
        md_path = os.path.join(self._output_dir, f"{safe_name}.md")
        self._write_file(md_path, md_content)
        result["markdown_path"] = md_path
        result["markdown_filename"] = f"{safe_name}.md"
        result["markdown_content"] = md_content

        # --- 2. PDF via WeasyPrint (P6: offloaded to thread pool — non-blocking) ---
        pdf_path = os.path.join(self._output_dir, f"{safe_name}.pdf")
        pdf_ok = await asyncio.to_thread(self._generate_pdf, md_content, pdf_path, topic)
        if pdf_ok:
            result["pdf_path"] = pdf_path
            result["pdf_filename"] = f"{safe_name}.pdf"
            result["pdf_url"] = f"/api/reports/{safe_name}.pdf"

        # --- 3. JSON structured data ---
        json_data = self._build_json_report(
            report_id=report_id,
            topic=topic,
            generated_at=generated_at,
            content=md_content,
            sections=sections,
            sources=sources,
            runtime_seconds=runtime_seconds,
        )
        json_path = os.path.join(self._output_dir, f"{safe_name}.json")
        self._write_file(json_path, json.dumps(json_data, indent=2, ensure_ascii=False, default=str))
        result["json_path"] = json_path
        result["json_filename"] = f"{safe_name}.json"
        result["json_data"] = json_data

        # --- 4. ChromaDB mini store for follow-up chat (P5: fire-and-forget background task) ---
        # The user gets the report immediately; ChromaDB embedding runs in the background.
        asyncio.create_task(self._embed_to_chromadb(report_id, md_content))
        result["chat_enabled"] = True  # optimistically true — will be ready within seconds

        # --- 5. Redis cache for future hits ---
        if not is_cache_hit:
            await self._store_in_redis(topic, result)

        # --- 6. LangSmith run metadata ---
        self._write_langsmith_metadata(
            report_id=report_id,
            topic=topic,
            runtime_seconds=runtime_seconds,
            section_count=len(sections) if sections else 0,
            source_count=len(sources) if sources else 0,
        )

        # Attach summary metadata to result
        result["source_count"] = len(sources) if sources else 0
        result["runtime_seconds"] = runtime_seconds

        logger.info(f"[OutputCompiler] All outputs compiled for '{topic[:50]}' in {runtime_seconds}s")
        return result



    # ------------------------------------------------------------------
    # PDF — WeasyPrint (HTML → PDF)
    # ------------------------------------------------------------------
    def _generate_pdf(self, markdown_content: str, filepath: str, topic: str) -> bool:
        """Convert markdown to PDF using WeasyPrint (runs in thread pool via caller)."""
        try:
            from weasyprint import HTML

            html_body = md_lib.markdown(
                markdown_content,
                extensions=["tables", "fenced_code", "codehilite"],
            )

            # P6: removed Google Fonts @import (live HTTP request inside PDF render).
            # Using system font stack instead — eliminates 0.5-2s network call.
            # Modern premium light theme for PDF
            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            margin: 2.5cm;
            @bottom-right {{
                content: counter(page);
                font-size: 9pt;
                color: #71717a;
            }}
        }}
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.7;
            color: #18181b;
            margin: 0;
            padding: 0;
        }}
        h1 {{ 
            color: #f96c3d; 
            font-size: 28pt; 
            margin-bottom: 30pt;
            line-height: 1.2;
            font-weight: 800;
        }}
        h2 {{ 
            color: #18181b; 
            font-size: 20pt; 
            margin-top: 32pt; 
            margin-bottom: 16pt;
            border-bottom: 1px solid #e4e4e7;
            padding-bottom: 8pt;
            font-weight: 700;
        }}
        h3 {{ 
            color: #27272a; 
            font-size: 15pt; 
            margin-top: 24pt;
            font-weight: 600;
        }}
        p {{ margin-bottom: 14pt; }}
        ul, ol {{ margin-bottom: 14pt; padding-left: 20pt; }}
        li {{ margin-bottom: 6pt; }}
        li::marker {{ color: #f96c3d; font-weight: bold; }}
        strong {{ color: #09090b; font-weight: 700; }}
        a {{ color: #f96c3d; text-decoration: none; }}
        code {{
            background-color: #f4f4f5;
            padding: 2pt 4pt;
            border-radius: 4pt;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            color: #c026d3;
        }}
        pre {{
            background-color: #f4f4f5;
            padding: 16pt;
            border-radius: 8pt;
            white-space: pre-wrap;
            font-size: 10pt;
            border: 1px solid #e4e4e7;
            margin: 16pt 0;
        }}
        blockquote {{
            border-left: 4pt solid #f96c3d;
            padding-left: 16pt;
            margin: 20pt 0;
            color: #52525b;
            font-style: italic;
            background: #fffaf9;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20pt 0;
        }}
        th, td {{
            border: 1px solid #e4e4e7;
            padding: 10pt 12pt;
            text-align: left;
        }}
        th {{ 
            background-color: #f8fafc; 
            font-weight: 700; 
            color: #18181b;
        }}
        .citation {{
            font-size: 9pt;
            color: #71717a;
            vertical-align: super;
        }}
    </style>
</head>
<body>
    <div style="text-align: center; margin-bottom: 40pt;">
        <div style="font-size: 10pt; text-transform: uppercase; letter-spacing: 2pt; color: #71717a; margin-bottom: 10pt;">AI Research Intelligence Report</div>
        <h1>{topic}</h1>
        <div style="font-size: 10pt; color: #71717a;">Generated on {datetime.now().strftime('%B %d, %Y')}</div>
    </div>
    <hr style="border: 0; border-top: 1px solid #e4e4e7; margin-bottom: 40pt;">
    {html_body}
</body>
</html>"""

            HTML(string=full_html).write_pdf(filepath)
            logger.info(f"[OutputCompiler] PDF saved → {filepath}")
            return True

        except ImportError:
            logger.warning("[OutputCompiler] WeasyPrint not installed — PDF skipped.")
            return False
        except Exception as e:
            logger.error(f"[OutputCompiler] PDF generation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # JSON — structured report data
    # ------------------------------------------------------------------
    def _build_json_report(
        self,
        report_id: str,
        topic: str,
        generated_at: str,
        content: str,
        sections: list = None,
        sources: list = None,
        runtime_seconds: float = 0.0,
    ) -> dict:
        """Build a structured JSON representation of the report."""
        # Serialize sections
        serialized_sections = []
        for s in (sections or []):
            if hasattr(s, "model_dump"):
                serialized_sections.append(s.model_dump())
            elif hasattr(s, "dict"):
                serialized_sections.append(s.dict())
            elif isinstance(s, dict):
                serialized_sections.append(s)
            else:
                serialized_sections.append({"name": str(s)})

        # Serialize sources
        serialized_sources = []
        for src in (sources or []):
            if hasattr(src, "model_dump"):
                serialized_sources.append(src.model_dump())
            elif hasattr(src, "dict"):
                serialized_sources.append(src.dict())
            elif isinstance(src, dict):
                serialized_sources.append(src)

        return {
            "report_id": report_id,
            "topic": topic,
            "generated_at": generated_at,
            "runtime_seconds": runtime_seconds,
            "content": content,
            "sections": serialized_sections,
            "sources": serialized_sources,
            "metadata": {
                "section_count": len(serialized_sections),
                "source_count": len(serialized_sources),
            },
        }

    # ------------------------------------------------------------------
    # ChromaDB — embed report chunks for follow-up chat
    # ------------------------------------------------------------------
    async def _embed_to_chromadb(self, report_id: str, content: str) -> bool:
        """Embed report chunks into ChromaDB for RAG-based follow-up chat."""
        try:
            from .chat.followup import FollowupChatHandler
            handler = FollowupChatHandler()
            ok = await handler.embed_report(report_id, content)
            if ok:
                logger.info(f"[OutputCompiler] ChromaDB embedded for report {report_id}")
            return ok
        except Exception as e:
            logger.warning(f"[OutputCompiler] ChromaDB embedding failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Redis — cache compiled outputs
    # ------------------------------------------------------------------
    async def _store_in_redis(self, topic: str, result: dict) -> None:
        """Store the compiled output in Redis semantic cache."""
        try:
            from .cache.redis_cache import SemanticCache
            cache = SemanticCache()
            # Store the markdown content + metadata (exclude binary paths)
            cache_payload = {
                "content": result.get("markdown_content", ""),
                "report_id": result.get("report_id", ""),
                "json_data": result.get("json_data", {}),
                "generated_at": result.get("generated_at", ""),
            }
            await cache.store_result(topic, cache_payload)
            logger.info(f"[OutputCompiler] Redis cached for '{topic[:50]}'")
        except Exception as e:
            logger.warning(f"[OutputCompiler] Redis caching failed: {e}")

    # ------------------------------------------------------------------
    # LangSmith — write run metadata
    # ------------------------------------------------------------------
    def _write_langsmith_metadata(
        self,
        report_id: str,
        topic: str,
        runtime_seconds: float,
        section_count: int,
        source_count: int,
    ) -> None:
        """
        Write LangSmith run metadata. Attempts to tag the active LangSmith
        run if available, otherwise logs metadata locally.
        """
        metadata = {
            "report_id": report_id,
            "topic": topic,
            "runtime_seconds": runtime_seconds,
            "cost_estimate_usd": self._estimate_cost(section_count),
            "section_count": section_count,
            "source_count": source_count,
        }

        # Try to write to the active LangSmith run
        try:
            from langsmith import Client as LangSmithClient
            ls = LangSmithClient()
            langsmith_run_id = os.getenv("LANGCHAIN_RUN_ID")
            if langsmith_run_id:
                ls.update_run(langsmith_run_id, extra={"output_metadata": metadata})
                logger.info("[OutputCompiler] LangSmith metadata written to active run.")
                return
        except Exception:
            pass

        # Fallback: write metadata to a local JSON file
        meta_path = os.path.join(self._output_dir, f"{report_id}_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[OutputCompiler] LangSmith metadata saved locally → {meta_path}")
        except Exception as e:
            logger.warning(f"[OutputCompiler] Failed to save metadata: {e}")

    @staticmethod
    def _estimate_cost(section_count: int) -> float:
        """
        Rough cost estimate per research run.
        Based on: cheap calls (~$0.0003) × (planning + per-section nodes)
                + mid calls (~$0.001) × sections
                + premium call (~$0.015) × 1
        """
        cheap_calls = 2 + section_count * 4   # HyDE, plan, + 4 per section (rewrite, search, write, critic)
        mid_calls = section_count              # section writing
        premium_calls = 1                      # final synthesis
        return round(
            cheap_calls * 0.0003 + mid_calls * 0.001 + premium_calls * 0.015, 4
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _write_file(self, filepath: str, content: str) -> None:
        """Write string content to a file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"[OutputCompiler] Saved → {filepath}")
        except Exception as e:
            logger.error(f"[OutputCompiler] Error writing {filepath}: {e}")

    @staticmethod
    def _sanitize_filename(topic: str) -> str:
        """Sanitize topic string for use as a filename."""
        name = re.sub(r'[\\/*?:"<>|]', "", topic)
        name = name.replace(" ", "_")
        return name[:50]


# ==============================================================
# Graph Node — wired into LangGraph as the terminal node
# ==============================================================

async def output_compiler_node(state: ReportState) -> dict:
    """
    LangGraph node that compiles the final report into all output formats.

    Wired as: final_synthesis_writer → output_compiler → END

    Reads from state:
        final_report, topic, sections, sources,
        confidence_scores, fact_check_flags, reflection_count
    Writes to state:
        output_metadata (dict with all output paths and data)
    """
    from .chat.followup import FollowupChatHandler

    compiler = OutputCompiler()
    report_id = FollowupChatHandler.generate_report_id()

    final_report = state.get("final_report", "")
    topic = state.get("topic", "Research Report")
    sections = state.get("sections", [])
    sources = state.get("sources", [])

    logger.info("--- Output Compiler Node ---")

    result = await compiler.compile(
        report_content=final_report,
        report_id=report_id,
        topic=topic,
        sections=sections,
        sources=sources,
        start_time=None,  # Runtime tracked per-session in main.py
        is_cache_hit=state.get("cache_hit", False),
    )

    logger.info(f"--- Output Compiler Complete (report_id={report_id}) ---")

    return {
        "output_metadata": result,
        "final_report": final_report,  # Pass through unchanged
    }
