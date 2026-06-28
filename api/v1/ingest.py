"""
api/v1/ingest.py

The single POST /ingest endpoint.

This file is deliberately thin — it:
  1. Validates the uploaded file.
  2. Calls the service layer in order (extract → chunk → embed → store).
  3. Returns a structured response.

No business logic lives here. Each step is in its own service.
This is the "orchestrator" pattern — the endpoint just coordinates.

FastAPI specifics:
  - UploadFile: FastAPI's wrapper around the uploaded file. Gives us
    .filename, .content_type, .read() (async).
  - Form(): tells FastAPI these params come from form data (not JSON body).
    Required because you can't mix UploadFile + JSON body in one request.
  - Depends(): dependency injection — FastAPI calls the function and passes
    the result to your endpoint. Used here for shared service instances.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends

from models.schemas import (
    ChunkStrategy,
    IngestResponse,
    ErrorResponse,
)
from services.extractor import TextExtractor, UnsupportedFileTypeError, ExtractionError
from services.chunker import Chunker
from services.embedder import Embedder
from db.qdrant_store import QdrantStore
from db.postgres import PostgresStore
from core.config import settings


router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


# ── Dependency providers ───────────────────────────────────────────────────────
# These are called by FastAPI's Depends() system.
# In a larger app you'd use a DI container. For this size, functions are fine.

def get_extractor() -> TextExtractor:
    return TextExtractor()

def get_chunker() -> Chunker:
    return Chunker()

def get_embedder() -> Embedder:
    return Embedder()

def get_qdrant() -> QdrantStore:
    return QdrantStore()

def get_postgres() -> PostgresStore:
    return PostgresStore()


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a .pdf or .txt file. The API extracts text, chunks it using "
        "the selected strategy, generates embeddings, and stores everything "
        "in Qdrant (vectors) and PostgreSQL (metadata)."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or parameters"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Internal processing error"},
    },
)
async def ingest_document(
    # ── File ──────────────────────────────────────────────────────────────────
    file: UploadFile = File(..., description="PDF or TXT file to ingest"),

    # ── Chunking parameters (Form fields, not JSON) ────────────────────────
    strategy: ChunkStrategy = Form(
        default=ChunkStrategy.FIXED,
        description="'fixed' for character-based, 'semantic' for topic-based",
    ),
    chunk_size: int = Form(
        default=500,
        ge=100,
        le=4000,
        description="Characters per chunk (fixed strategy only)",
    ),
    chunk_overlap: int = Form(
        default=50,
        ge=0,
        le=500,
        description="Overlap characters between chunks (fixed strategy only)",
    ),

    # ── Injected services ────────────────────────────────────────────────────
    extractor: TextExtractor = Depends(get_extractor),
    chunker: Chunker = Depends(get_chunker),
    embedder: Embedder = Depends(get_embedder),
    qdrant: QdrantStore = Depends(get_qdrant),
    postgres: PostgresStore = Depends(get_postgres),
) -> IngestResponse:
    """
    Full ingestion pipeline:
      read file → validate size → extract text → chunk → embed → store → respond
    """

    # ── Step 1: Read file bytes ────────────────────────────────────────────────
    file_bytes = await file.read()  # async read — doesn't block event loop

    # ── Step 2: Validate file size ─────────────────────────────────────────────
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(file_bytes) / 1024 / 1024:.1f}MB exceeds "
                   f"limit of {settings.max_file_size_mb}MB.",
        )

    # ── Step 3: Extract text ───────────────────────────────────────────────────
    try:
        document = extractor.extract(file_bytes, file.filename or "upload")
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # ── Step 4: Chunk the text ─────────────────────────────────────────────────
    chunks = chunker.chunk(
        document=document,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced no chunks after extraction. "
                   "The file may be empty or contain only images.",
        )

    # ── Step 5: Generate embeddings ────────────────────────────────────────────
    try:
        chunks = embedder.embed_chunks(chunks)
    except ValueError as e:
        # e.g., OpenAI key missing
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # ── Step 6: Store in Qdrant ────────────────────────────────────────────────
    # Ensure the collection exists with the right vector dimension
    vector_size = len(chunks[0].embedding)
    await qdrant.ensure_collection(vector_size=vector_size)
    await qdrant.upsert_chunks(chunks, filename=document.filename)

    # ── Step 7: Store metadata in PostgreSQL ───────────────────────────────────
    await postgres.save_document(
        document=document,
        chunks=chunks,
        strategy=strategy,
        embedding_provider=settings.embedding_provider,
    )

    # ── Step 8: Return response ────────────────────────────────────────────────
    return IngestResponse(
        document_id=document.document_id,
        filename=document.filename,
        file_size_bytes=document.file_size_bytes,
        strategy_used=strategy,
        chunks_created=len(chunks),
        embedding_provider=settings.embedding_provider,
        ingested_at=datetime.now(timezone.utc),
    )
