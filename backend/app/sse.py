"""
SSE (Server-Sent Events) Manager.
Manages real-time progress event streaming from LangGraph nodes to the frontend.
"""

import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SSEEvent:
    """A single SSE event to send to the client."""
    event_type: str        # e.g., "query_analyzed", "section_researching", "report_ready"
    message: str           # Human-readable progress message
    data: dict = field(default_factory=dict)  # Additional structured data
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SSEManager:
    """
    Manages SSE event streaming for a single research session.

    Each research request gets its own SSEManager instance.
    LangGraph nodes call emit() to push progress updates,
    and the FastAPI SSE endpoint consumes events via stream().
    """

    def __init__(self):
        self._queue: asyncio.Queue[Optional[SSEEvent]] = asyncio.Queue()
        self._closed = False

    async def emit(
        self,
        event_type: str,
        message: str,
        data: dict = None,
    ) -> None:
        """
        Emit an SSE event to the stream.

        Args:
            event_type: Type of event (e.g., 'query_analyzed', 'section_complete').
            message: Human-readable status message for the UI.
            data: Optional structured data (e.g., confidence scores, section names).
        """
        if self._closed:
            return

        event = SSEEvent(
            event_type=event_type,
            message=message,
            data=data or {},
        )
        await self._queue.put(event)

    async def close(self) -> None:
        """Signal end of stream."""
        self._closed = True
        await self._queue.put(None)  # Sentinel to stop iteration

    async def stream(self):
        """
        Async generator that yields SSE-formatted event strings.
        Used by the FastAPI SSE endpoint.

        Yields:
            dict: SSE event data formatted for sse-starlette's EventSourceResponse.
        """
        while True:
            event = await self._queue.get()

            if event is None:
                # Stream ended
                break

            yield {
                "event": event.event_type,
                "data": json.dumps({
                    "type": event.event_type,
                    "message": event.message,
                    "data": event.data,
                    "timestamp": event.timestamp,
                }),
            }


# Pre-defined event types for consistency
class EventTypes:
    """Standard SSE event types used across the pipeline."""
    QUERY_ANALYZING = "query_analyzing"
    QUERY_ANALYZED = "query_analyzed"
    CACHE_HIT = "cache_hit"
    PLAN_GENERATING = "plan_generating"
    PLAN_GENERATED = "plan_generated"
    SECTION_RESEARCHING = "section_researching"
    SECTION_WRITING = "section_writing"
    SECTION_COMPLETE = "section_complete"
    REFLECTION_LOOP = "reflection_loop"
    FACT_CHECKING = "fact_checking"
    FACT_CHECK_COMPLETE = "fact_check_complete"
    SYNTHESIS_WRITING = "synthesis_writing"
    COMPILING_OUTPUT = "compiling_output"
    REPORT_READY = "report_ready"
    PLAN_REVIEW_REQUIRED = "plan_review_required"
    ERROR = "error"
