"""
services/chunker.py

Responsibility: take raw text → return a list of text chunks.

Two strategies:

  FIXED (character-based):
    Split at every N characters, with M characters of overlap.
    Simple, fast, deterministic.
    Problem: might cut mid-sentence. Overlap reduces this.

  SEMANTIC (sentence-boundary + similarity):
    1. Split text into sentences using NLTK.
    2. Embed each sentence (small, fast local model).
    3. Compute cosine similarity between adjacent sentences.
    4. When similarity DROPS (topic changes), start a new chunk.
    Slower, but produces chunks that each contain one coherent idea.
    Better retrieval quality — the retrieved chunk won't mix two topics.

The chunker is strategy-agnostic from the caller's perspective:
  chunks = chunker.chunk(text, strategy=ChunkStrategy.FIXED, ...)
"""

import re
import uuid
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

from models.schemas import ChunkStrategy, ChunkRecord, DocumentRecord
from core.config import settings


# Download NLTK sentence tokenizer data on first run.
# 'punkt_tab' is the updated name in NLTK 3.8+. Falls back gracefully.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class Chunker:
    """
    Handles both chunking strategies.

    The semantic chunker needs an embedding model to measure sentence similarity.
    We use a small local model here (all-MiniLM-L6-v2) — NOT the same model
    used for the final chunk embeddings. This is purely for intra-chunker
    similarity math. It's fast (~80ms per call on CPU).

    Why not reuse the main embedder?
      Circular dependency: embedder would import chunker, chunker would import embedder.
      Also, the semantic chunker uses a tiny model on purpose — we don't need
      high-quality embeddings to detect topic shifts, just relative similarity.
    """

    def __init__(self) -> None:
        # Lazy-load the local similarity model.
        # Only instantiated once per app lifetime (shared across requests).
        self._similarity_model: SentenceTransformer | None = None

    def chunk(
        self,
        document: DocumentRecord,
        strategy: ChunkStrategy,
        chunk_size: int = settings.default_chunk_size,
        chunk_overlap: int = settings.default_chunk_overlap,
    ) -> list[ChunkRecord]:
        """
        Main entry point. Selects strategy and returns list of ChunkRecord.

        Args:
            document:      Full DocumentRecord from the extractor.
            strategy:      Which strategy to use.
            chunk_size:    Used by FIXED strategy.
            chunk_overlap: Used by FIXED strategy.

        Returns:
            List of ChunkRecord — each has its own chunk_id UUID,
            the text, char count, and reference back to the document.
            NOTE: embedding is empty list [] at this stage — filled later
            by the embedder service.
        """
        if strategy == ChunkStrategy.FIXED:
            texts = self._fixed_chunk(document.raw_text, chunk_size, chunk_overlap)
        else:
            texts = self._semantic_chunk(document.raw_text)

        # Wrap each text string into a proper ChunkRecord
        return [
            ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                chunk_index=i,
                text=text,
                char_count=len(text),
                embedding=[],          # Will be filled by embedder.py
                strategy_used=strategy,
                page_number=None,      # Page tracking not implemented in this version
            )
            for i, text in enumerate(texts)
        ]

    # ── Strategy 1: Fixed-size chunking ───────────────────────────────────────

    def _fixed_chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Slide a window of `chunk_size` characters over the text,
        advancing by (chunk_size - overlap) each step.

        Example with chunk_size=20, overlap=5:
          text = "AAAAA BBBBB CCCCC DDDDD EEEEE"
          chunk 1: chars 0–19   = "AAAAA BBBBB CCCCC DD"
          chunk 2: chars 15–34  = "D DDDDD EEEEE"  ← 5 chars of overlap

        The overlap ensures sentences split across a boundary still appear
        fully in at least one chunk.

        Clean-up: we strip each chunk and discard whitespace-only chunks.
        """
        chunks: list[str] = []
        start = 0
        step = chunk_size - overlap  # How far to advance the window each time

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:  # Skip empty/whitespace-only slices
                chunks.append(chunk)

            start += step

        return chunks

    # ── Strategy 2: Semantic chunking ─────────────────────────────────────────

    def _semantic_chunk(self, text: str) -> list[str]:
        """
        Split text at natural topic boundaries by measuring sentence similarity.

        Algorithm:
          1. Tokenise into sentences using NLTK.
          2. Embed all sentences at once (batched — fast).
          3. For each adjacent pair of sentences, compute cosine similarity.
          4. If similarity < threshold → topic has shifted → start new chunk.
          5. Accumulate sentences into the current chunk until a split point.

        Why cosine similarity?
          We want to measure the ANGLE between vectors, not their magnitude.
          Two sentences about the same topic will point in a similar direction
          in embedding space, regardless of sentence length.
          cos(θ) → 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.

        Threshold (from settings.semantic_similarity_threshold, default 0.3):
          If cos_sim < 0.3 between sentence[i] and sentence[i+1], we cut here.
          0.3 is empirically a good default — sentences about different topics
          in a technical document typically score 0.1–0.25.
        """
        # Step 1: sentence tokenise
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= 1:
            # Edge case: single sentence or very short text
            return [text.strip()] if text.strip() else []

        # Step 2: embed all sentences in one batch
        model = self._get_similarity_model()
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
        # embeddings shape: (num_sentences, 384) — numpy array

        # Step 3 & 4: compute similarity between adjacent sentences, decide cuts
        chunks: list[str] = []
        current_chunk_sentences: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if sim < settings.semantic_similarity_threshold:
                # Topic shift detected — save the current chunk and start fresh
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk_sentences = [sentences[i]]
            else:
                # Still on the same topic — keep accumulating
                current_chunk_sentences.append(sentences[i])

        # Don't forget the last chunk (loop ends before saving it)
        if current_chunk_sentences:
            last_chunk = " ".join(current_chunk_sentences).strip()
            if last_chunk:
                chunks.append(last_chunk)

        return chunks

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two embedding vectors.

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

        Where:
          A · B   = dot product (sum of element-wise products)
          ||A||   = L2 norm of A (sqrt of sum of squares)

        Result is always in [-1, 1]. For embeddings from the same model,
        practically always in [0, 1] for positive semantic similarity.

        We clamp to [-1, 1] to handle any floating point drift.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0  # Zero vectors have undefined similarity — treat as unrelated

        similarity = dot_product / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))

    def _get_similarity_model(self) -> SentenceTransformer:
        """
        Lazy-load the similarity model (only when semantic chunking is first used).
        Reuses the same instance for all subsequent calls.

        Why lazy? Loading takes ~1-2 seconds. If the user only uses fixed chunking,
        we never pay that cost.
        """
        if self._similarity_model is None:
            self._similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._similarity_model
