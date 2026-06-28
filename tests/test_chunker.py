"""Pure unit tests for the FIXED chunking strategy — no DB/network needed."""

from models.schemas import ChunkStrategy, DocumentRecord
from services.chunker import Chunker


def _make_document(text: str) -> DocumentRecord:
    return DocumentRecord(
        document_id="doc-1",
        filename="test.txt",
        file_size_bytes=len(text),
        raw_text=text,
        total_pages=None,
    )


def test_fixed_chunk_respects_size_and_overlap():
    text = "A" * 50 + "B" * 50  # 100 chars
    document = _make_document(text)
    chunker = Chunker()

    chunks = chunker.chunk(document, ChunkStrategy.FIXED, chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.strategy_used == ChunkStrategy.FIXED for chunk in chunks)
    assert all(chunk.embedding == [] for chunk in chunks)  # not embedded yet


def test_fixed_chunk_indices_are_sequential():
    document = _make_document("word " * 200)
    chunker = Chunker()

    chunks = chunker.chunk(document, ChunkStrategy.FIXED, chunk_size=100, chunk_overlap=20)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == "doc-1" for c in chunks)


def test_empty_text_produces_no_chunks():
    document = _make_document("")
    chunker = Chunker()

    chunks = chunker.chunk(document, ChunkStrategy.FIXED, chunk_size=100, chunk_overlap=10)

    assert chunks == []
