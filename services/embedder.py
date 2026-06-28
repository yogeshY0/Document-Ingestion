"""
services/embedder.py

Responsibility: take a list of ChunkRecord (with empty embeddings)
                → fill in each chunk's embedding vector.

Two providers:
  "openai" → calls OpenAI API, returns 1536-dim vectors.
             Pro: very high quality. Con: costs money, needs API key, network call.

  "local"  → runs sentence-transformers on CPU, returns 384-dim vectors.
             Pro: free, offline, fast enough for most use cases.
             Con: lower quality than OpenAI (but still very good for RAG).

The provider is selected by settings.embedding_provider.
Both providers produce the same output shape: list[float].

Key concept — WHY batching matters:
  If you have 200 chunks and call the embed API 200 times, you pay 200× the
  network latency overhead. Batching sends all 200 in one call (or a few large
  calls). For OpenAI this also reduces API cost.
  Batch size 100 is a safe default — OpenAI's limit is 2048, local has no limit
  but RAM is the constraint.
"""

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from models.schemas import ChunkRecord
from core.config import settings


BATCH_SIZE = 100  # Process this many chunks per API call


class Embedder:
    """
    Wraps both embedding providers behind a single interface.
    The rest of the codebase never knows which provider is active.
    """

    def __init__(self) -> None:
        self._openai_client: OpenAI | None = None
        self._local_model: SentenceTransformer | None = None

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """
        Fill in the embedding field for every chunk.
        Modifies chunks IN PLACE and also returns the list.

        Why in-place modification?
          ChunkRecord is a Pydantic model. We update chunk.embedding directly.
          Returning the list too makes the calling code read more clearly:
            chunks = embedder.embed_chunks(chunks)
          versus silently relying on mutation.

        Args:
            chunks: List of ChunkRecord with embedding=[] (from the chunker).

        Returns:
            Same list, now with embedding vectors filled in.
        """
        provider = settings.embedding_provider

        if provider == "openai":
            self._embed_with_openai(chunks)
        else:
            self._embed_with_local(chunks)

        return chunks

    def embed_query(self, query_text: str) -> list[float]:
        """
        Embed a single query string (used at retrieval time in the RAG API).
        Returns a single vector.

        CRITICAL: must use the SAME model as was used to embed chunks.
        If chunks were embedded with OpenAI, the query must also use OpenAI.
        Mixing models → vectors live in different spaces → cosine similarity is meaningless.
        """
        provider = settings.embedding_provider

        if provider == "openai":
            client = self._get_openai_client()
            response = client.embeddings.create(
                input=[query_text],
                model=settings.openai_embedding_model,
            )
            return response.data[0].embedding
        else:
            model = self._get_local_model()
            vector = model.encode([query_text], show_progress_bar=False)[0]
            return vector.tolist()

    # ── OpenAI provider ────────────────────────────────────────────────────────

    def _embed_with_openai(self, chunks: list[ChunkRecord]) -> None:
        """
        Batch-embed all chunks using the OpenAI Embeddings API.

        Batching logic:
          We send BATCH_SIZE chunk texts per API call.
          OpenAI returns the embeddings in the same order we sent the texts.
          We map them back to the original chunk objects by index.
        """
        client = self._get_openai_client()
        texts = [chunk.text for chunk in chunks]

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]

            response = client.embeddings.create(
                input=batch_texts,
                model=settings.openai_embedding_model,
                # encoding_format="float" is the default — explicit for clarity
            )

            # response.data is a list of Embedding objects, same order as input
            for i, embedding_obj in enumerate(response.data):
                chunk_index = batch_start + i
                chunks[chunk_index].embedding = embedding_obj.embedding

    def _get_openai_client(self) -> OpenAI:
        """Lazy-initialise the OpenAI client."""
        if self._openai_client is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "embedding_provider is 'openai' but OPENAI_API_KEY is not set. "
                    "Either set the key or switch to embedding_provider='local'."
                )
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    # ── Local (sentence-transformers) provider ─────────────────────────────────

    def _embed_with_local(self, chunks: list[ChunkRecord]) -> None:
        """
        Batch-embed all chunks using a local sentence-transformers model.

        model.encode() natively supports batch processing — pass all texts
        at once and it handles batching internally.

        Output: numpy array of shape (num_chunks, 384).
        We convert each row to a Python list[float] for storage.
        """
        model = self._get_local_model()
        texts = [chunk.text for chunk in chunks]

        # encode() returns numpy ndarray shape (N, embedding_dim)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalise → cosine sim = dot product
                                        # (slightly faster search in Qdrant)
        )

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

    def _get_local_model(self) -> SentenceTransformer:
        """Lazy-load the local embedding model (downloaded on first use, ~90MB)."""
        if self._local_model is None:
            self._local_model = SentenceTransformer(settings.local_embedding_model)
        return self._local_model
