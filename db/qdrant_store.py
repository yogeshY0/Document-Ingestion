"""
db/qdrant_store.py

Thin wrapper around Qdrant — the only vector store this project uses
(no FAISS/Chroma per the task constraints). Two responsibilities:

  1. ensure_collection: create the collection on first use, sized to match
     whichever embedding provider is active (384 dims for local, 1536 for
     OpenAI). Idempotent — safe to call on every ingest.
  2. upsert_chunks / search: write embedded chunks, and retrieve the
     top-k nearest chunks for a query vector at chat time.

We use Qdrant's AsyncQdrantClient so this plays nicely with FastAPI's async
endpoints without blocking the event loop.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models as qmodels

from core.config import settings
from models.schemas import ChunkRecord


class QdrantStore:
    """Stateless-ish wrapper; holds a single AsyncQdrantClient connection."""

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._collection = settings.qdrant_collection

    async def ensure_collection(self, vector_size: int) -> None:
        """
        Create the collection if it doesn't exist yet. Distance is Cosine,
        which pairs correctly with the L2-normalized vectors the local
        embedder produces (and is also the right metric for OpenAI embeddings).
        """
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    async def upsert_chunks(self, chunks: list[ChunkRecord], filename: str) -> None:
        """
        Write each chunk's embedding + metadata payload as a Qdrant point.

        `filename` is denormalized into every chunk's payload (rather than
        requiring a join back to Postgres) so the chat API can show the
        source filename next to a retrieved chunk with zero extra DB calls.
        """
        points = [
            qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "strategy_used": chunk.strategy_used.value,
                    "page_number": chunk.page_number,
                },
            )
            for chunk in chunks
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_id: str | None = None,
    ) -> list[qmodels.ScoredPoint]:
        """
        Return the top_k chunks most similar to query_vector.

        Args:
            query_vector: embedding of the user's query (same model/dim as
                           the stored chunk vectors — caller's responsibility).
            top_k:        how many results to return.
            document_id:  optional filter to restrict search to one document.
        """
        query_filter = None
        if document_id is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )

        result = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return result.points

    async def close(self) -> None:
        await self._client.close()
