"""
Follow-up Chat Handler.
RAG-based Q&A over generated reports using ChromaDB for retrieval.

Answer flow (single LLM call max):
  distance <= CONTEXT_THRESHOLD   → answer from context
  distance <= DOMAIN_THRESHOLD    → domain-relevant fallback from general knowledge
  distance >  DOMAIN_THRESHOLD    → politely decline (no LLM call)
"""

from ..embeddings import embed_texts , embed_text

import uuid
from typing import List, Dict, Any

try:
    import chromadb
except ImportError:
    chromadb = None

TOP_K = 7  # retrieve more candidates

ENTITY_BOOST_FACTOR = 0.75  

# Tune these two values based on your embedding model
CONTEXT_THRESHOLD = 0.70  # chunk is a direct answer source
DOMAIN_THRESHOLD = 0.90  # chunk is in the same subject area but not a direct answer


class FollowupChatHandler:
    """
    Handles follow-up questions about a generated report.

    Flow:
    1. After report generation, chunk report → embed → store in ChromaDB.
    2. User asks a question → embed → retrieve top-5 chunks → Claude Haiku answers.
    """

    def __init__(self):
        self._chroma_client = None

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract likely entity tokens from messy user input.
        Works even with lowercase and minor typos.
        """

        stopwords = {
        "what","why","when","where","which","who",
        "is","are","do","does","did","can","could",
        "should","would","explain","tell","about",
        "the","a","an","to","for","of"
        }

        entities = []

        for w in text.split():
            w_clean = w.strip(".,!?()[]{}\"'").lower()

            if (
                len(w_clean) >= 2      # catch AI, ML, etc.
                and w_clean not in stopwords
                and any(c.isalpha() for c in w_clean)
            ):
                entities.append(w_clean)

        return list(set(entities))

    def _get_chroma_client(self):
        """Lazy init ChromaDB client."""
        if self._chroma_client is None:
            if chromadb is None:
                print("[FollowupChat] chromadb package not installed. Chat disabled.")
                return None

            # Old Code:
            # self._chroma_client = chromadb.Client()

            # New Code: Use PersistentClient so data survives server restarts
            import os

            db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
            self._chroma_client = chromadb.PersistentClient(path=db_path)

        return self._chroma_client

    async def embed_report(self, report_id: str, report_content: str) -> bool:
        """
        Chunk and embed a report into a ChromaDB collection.
        Memory-optimized: limits chunks to prevent OOM.

        Args:
            report_id: Unique identifier for the report/research session.
            report_content: Full markdown report content.

        Returns:
            True if successfully embedded, False otherwise.
        """
        client = self._get_chroma_client()
        if client is None:
            return False

        try:
            collection = client.get_or_create_collection(
                name=f"report_{report_id}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

            chunks = self._chunk_report(report_content)

            # Memory optimization: Cap chunks to prevent OOM
            # ChromaDB can handle ~500 chunks on 512MB RAM
            MAX_CHUNKS = 500
            if len(chunks) > MAX_CHUNKS:
                print(
                    f"[FollowupChat] Report has {len(chunks)} chunks, capping at {MAX_CHUNKS}"
                )
                chunks = chunks[:MAX_CHUNKS]

            if not chunks:
                return False
            embeddings = await embed_texts(chunks)
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"chunk_index": i} for i in range(len(chunks))],
            )

            print(
                f"[FollowupChat] Embedded {len(chunks)} chunks for report {report_id}"
            )
            return True

        except Exception as e:
            print(f"[FollowupChat] Error embedding report: {e}")
            return False

    async def answer_question(
        self,
        report_id: str,
        question: str,
        llm=None,
    ) -> Dict[str, Any]:
        """
        Answer a follow-up question using RAG + distance-based routing.
        Maximum 1 LLM call per question.

        Args:
            report_id: Report identifier to search within.
            question: User's follow-up question.
            llm: LLM instance (Claude Haiku 3.5).

        Returns:
            Dict with 'answer', 'sources', and 'source_type' fields.
            source_type: "context" | "general_knowledge" | "declined" | "error"
        """
        client = self._get_chroma_client()
        if client is None:
            return self._make_response("Chat feature is not available.", [], "error")

        if llm is None:
            return self._make_response("LLM not configured for chat.", [], "error")

        try:
            collection = client.get_collection(name=f"report_{report_id}")
            query_embedding = await embed_text(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(TOP_K, collection.count()),
            )

            retrieved_chunks: List[str] = results.get("documents", [[]])[0]
            distances: List[float] = results.get("distances", [[]])[0]

            entities = self._extract_entities(question)
            # ── APPLY ENTITY BOOST ───────────────────────────
            boosted = []
            for chunk, dist in zip(retrieved_chunks, distances):
                boosted_dist = dist

                chunk_lower = chunk.lower()

                for ent in entities:
                    if ent in chunk_lower and dist <= DOMAIN_THRESHOLD:
                        boosted_dist = min(dist * ENTITY_BOOST_FACTOR, dist)
                        break

                boosted.append((chunk, boosted_dist))

            # Sort by boosted distance (lower = better)
            boosted.sort(key=lambda x: x[1])

            boosted_chunks = [c for c, _ in boosted]
            boosted_distances = [d for _, d in boosted]

            best_distance = boosted_distances[0] if boosted_distances else 1.0

            # ── Route based on distance ──────────────────────────────────────

            if best_distance <= CONTEXT_THRESHOLD:
                # Good match in report → answer from context
                return await self._answer_from_context(question, boosted_chunks, llm)

            elif best_distance <= DOMAIN_THRESHOLD:
                # Same domain but not covered in report → general knowledge fallback
                return await self._answer_from_general_knowledge(question, llm)

            else:
                # Completely unrelated → decline, zero LLM calls
                return self._make_response(
                    (
                        "Your question doesn't seem related to the topic of this report. "
                        "I can only answer questions relevant to the report's subject area."
                    ),
                    [],
                    "declined",
                )

        except Exception as e:
            print(f"[FollowupChat] Error answering question: {e}")
            return self._make_response(f"Error: {str(e)}", [], "error")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _answer_from_context(
        self,
        question: str,
        chunks: List[str],
        llm,
    ) -> Dict[str, Any]:
        """Answer strictly from retrieved context chunks."""
        from langchain_core.messages import HumanMessage, SystemMessage

        context = "\n\n---\n\n".join(chunks)

        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful research assistant. "
                        "Answer the user's question based ONLY on the following context "
                        "from a research report. Be concise and accurate. "
                        "If the context does not fully cover the question, say so.\n\n"
                        f"Context:\n{context}"
                    )
                ),
                HumanMessage(content=question),
            ]
        )

        return self._make_response(response.content, chunks, "context")

    async def _answer_from_general_knowledge(
        self,
        question: str,
        llm,
    ) -> Dict[str, Any]:
        """
        Answer from general knowledge when the topic is domain-relevant
        but not covered in the report. Notifies the user clearly.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful research assistant. "
                        "The user's question is related to the report's subject area "
                        "but was not covered in the report itself. "
                        "Answer from your general knowledge. Be concise and accurate."
                    )
                ),
                HumanMessage(content=question),
            ]
        )

        notice = (
            "> ⚠️ **Not found in the report** — answering from general knowledge.\n\n"
        )

        return self._make_response(notice + response.content, [], "general_knowledge")

    def _chunk_report(
        self,
        content: str,
        chunk_size: int = 400,  # Decreased from 800 for better granularity
        overlap: int = 100,  # Decreased from 150
    ) -> List[str]:
        """
        Sliding window chunking with overlap.
        Optimized: smaller chunks = more specific retrieval context.
        """
        text = content.replace("\r", "")
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    @staticmethod
    def _make_response(
        answer: str,
        sources: List[str],
        source_type: str,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "source_type": source_type,
        }

    @staticmethod
    def generate_report_id() -> str:
        """Generate a unique report ID."""
        return str(uuid.uuid4())[:8]
