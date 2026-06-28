"""
services/retriever.py

The "R" in RAG, written out manually — no RetrievalQAChain, no LangChain
retriever abstraction. Just: embed the query, ask Qdrant for nearest
neighbours, shape the result into a schema the rest of the app understands.

Kept deliberately tiny and dependency-light so it's obvious exactly what's
happening at retrieval time.
"""

from __future__ import annotations

from core.config import settings
from db.qdrant_store import QdrantStore
from models.schemas import SourceChunk
from services.embedder import Embedder


class Retriever:
    """Combines the embedder (query -> vector) and Qdrant (vector -> chunks)."""

    def __init__(self, embedder: Embedder, qdrant: QdrantStore) -> None:
        self._embedder = embedder
        self._qdrant = qdrant

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[SourceChunk]:
        """
        Args:
            query:       the user's natural-language question.
            top_k:       how many chunks to return (defaults to settings).
            document_id: optionally restrict retrieval to a single document.

        Returns:
            Chunks ranked by similarity score, highest first.
        """
        query_vector = self._embedder.embed_query(query)

        points = await self._qdrant.search(
            query_vector=query_vector,
            top_k=top_k or settings.retrieval_top_k,
            document_id=document_id,
        )

        return [
            SourceChunk(
                chunk_id=str(point.id),
                document_id=point.payload["document_id"],
                filename=point.payload.get("filename", ""),
                text=point.payload["text"],
                score=point.score,
            )
            for point in points
        ]
