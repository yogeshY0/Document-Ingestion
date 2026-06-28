"""
models/schemas.py

All Pydantic models shared across the two APIs live here. Keeping schemas in
one module (rather than scattered per-service) means there's a single place
to check "what shape does this data have" and avoids circular imports between
services that both need the same type.

Roughly grouped into:
  1. Ingestion API models (DocumentRecord, ChunkRecord, IngestResponse)
  2. Conversational RAG API models (ChatMessage, ChatRequest/Response)
  3. Booking models (BookingDetails, BookingState, BookingRecord)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, EmailStr


# ════════════════════════════════════════════════════════════════════════════
# 1. Ingestion API
# ════════════════════════════════════════════════════════════════════════════

class ChunkStrategy(str, Enum):
    """The two selectable chunking strategies."""
    FIXED = "fixed"
    SEMANTIC = "semantic"


class DocumentRecord(BaseModel):
    """Result of text extraction — the full document, pre-chunking."""
    document_id: str
    filename: str
    file_size_bytes: int
    raw_text: str
    total_pages: int | None = None


class ChunkRecord(BaseModel):
    """A single chunk produced by the chunker, later filled with an embedding."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    char_count: int
    embedding: list[float] = Field(default_factory=list)
    strategy_used: ChunkStrategy
    page_number: int | None = None


class IngestResponse(BaseModel):
    """Response returned to the client after a successful ingestion."""
    document_id: str
    filename: str
    file_size_bytes: int
    strategy_used: ChunkStrategy
    chunks_created: int
    embedding_provider: str
    ingested_at: datetime


class ErrorResponse(BaseModel):
    detail: str


# ════════════════════════════════════════════════════════════════════════════
# 2. Conversational RAG API
# ════════════════════════════════════════════════════════════════════════════

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single turn stored in Redis chat memory."""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Stable id for this conversation")
    message: str = Field(..., min_length=1, description="The user's message")


class SourceChunk(BaseModel):
    """A retrieved chunk returned alongside the answer, for traceability."""
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    booking_state: "BookingState"


# ════════════════════════════════════════════════════════════════════════════
# 3. Interview booking
# ════════════════════════════════════════════════════════════════════════════

class BookingStatus(str, Enum):
    NONE = "none"                # no booking flow active
    IN_PROGRESS = "in_progress"  # collecting slots (name/email/date/time)
    CONFIRMED = "confirmed"      # all slots filled, validated, and persisted


class BookingDetails(BaseModel):
    """Slots collected across one or more turns. All optional — filled in
    incrementally as the user provides information."""
    name: str | None = None
    email: str | None = None
    date: str | None = None  # normalized to YYYY-MM-DD once confirmed
    time: str | None = None  # normalized to HH:MM (24h) once confirmed

    def is_complete(self) -> bool:
        return all([self.name, self.email, self.date, self.time])

    def missing_fields(self) -> list[str]:
        fields = {"name": self.name, "email": self.email, "date": self.date, "time": self.time}
        return [k for k, v in fields.items() if not v]


class BookingState(BaseModel):
    """Persisted in Redis per session_id; tracks where we are in the booking flow."""
    status: BookingStatus = BookingStatus.NONE
    details: BookingDetails = Field(default_factory=BookingDetails)


class BookingRecord(BaseModel):
    """A confirmed booking as stored in PostgreSQL."""
    booking_id: str
    session_id: str
    name: str
    email: EmailStr
    interview_date: str
    interview_time: str
    created_at: datetime
