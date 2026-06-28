"""
api/v1/chat.py

The Conversational RAG API. One main endpoint, POST /api/v1/chat, plus a
couple of small utility endpoints for inspecting/clearing session state and
listing confirmed bookings.

Per-request flow:
  1. Load this session's chat history + booking state from Redis.
  2. If a booking flow is already in progress, OR the user's message looks
     like a booking request, route to the booking-extraction path.
     Otherwise, route to the retrieval-augmented-generation path.
  3. Append the turn to Redis history.
  4. Return the answer (+ sources, + booking state) to the client.

No LangChain, no RetrievalQAChain — every step above is a plain function
call you can trace by reading this file top to bottom.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends

from db.postgres import PostgresStore
from db.qdrant_store import QdrantStore
from db.redis_store import RedisChatMemory
from models.schemas import (
    BookingRecord,
    BookingState,
    BookingStatus,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
)
from services.booking import BookingValidationError, validate_and_normalize
from services.embedder import Embedder
from services.llm import LLMService
from services.retriever import Retriever

router = APIRouter(prefix="/api/v1", tags=["Conversational RAG"])


# ── Dependency providers ──────────────────────────────────────────────────────
# Shared singletons (one connection pool / client each), reused across requests.

_redis = RedisChatMemory()
_postgres = PostgresStore()
_qdrant = QdrantStore()
_embedder = Embedder()
_retriever = Retriever(embedder=_embedder, qdrant=_qdrant)


def get_redis() -> RedisChatMemory:
    return _redis


def get_postgres() -> PostgresStore:
    return _postgres


def get_retriever() -> Retriever:
    return _retriever


def get_llm() -> LLMService:
    # Constructed per-request (cheap — it's a thin client wrapper) so a
    # missing OPENAI_API_KEY surfaces as a clean 500 on the request that
    # needs it, rather than crashing the whole app at import time.
    return LLMService()


# Cheap heuristic to decide whether a brand-new message is a booking request.
# Deliberately simple and deterministic — once we're in a booking flow,
# every subsequent turn is routed there regardless of these keywords.
_BOOKING_INTENT_RE = re.compile(
    r"\b(book|schedule|set up|arrange)\b.{0,30}\b(interview|call|meeting|appointment)\b",
    re.IGNORECASE,
)


def _looks_like_booking_request(message: str) -> bool:
    return bool(_BOOKING_INTENT_RE.search(message))


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Multi-turn conversational RAG, with interview-booking support",
)
async def chat(
    request: ChatRequest,
    redis: RedisChatMemory = Depends(get_redis),
    postgres: PostgresStore = Depends(get_postgres),
    retriever: Retriever = Depends(get_retriever),
    llm: LLMService = Depends(get_llm),
) -> ChatResponse:
    history = await redis.get_history(request.session_id)
    booking_state = await redis.get_booking_state(request.session_id)

    entering_booking_flow = booking_state.status == BookingStatus.NONE and _looks_like_booking_request(
        request.message
    )

    if booking_state.status == BookingStatus.IN_PROGRESS or entering_booking_flow:
        answer, booking_state = await _handle_booking_turn(
            request=request,
            history=history,
            booking_state=booking_state,
            llm=llm,
            postgres=postgres,
        )
        sources = []
    else:
        sources = await retriever.retrieve(request.message)
        answer = llm.generate_rag_answer(request.message, history, sources)

    await redis.append_message(request.session_id, ChatRole.USER, request.message)
    await redis.append_message(request.session_id, ChatRole.ASSISTANT, answer)
    await redis.set_booking_state(request.session_id, booking_state)

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        sources=sources,
        booking_state=booking_state,
    )


async def _handle_booking_turn(
    request: ChatRequest,
    history: list[ChatMessage],
    booking_state: BookingState,
    llm: LLMService,
    postgres: PostgresStore,
) -> tuple[str, BookingState]:
    """
    Advance the interview-booking flow by exactly one turn:
    extract whatever new fields the LLM found, and either confirm + persist
    (if complete and valid), ask for corrections (if complete but invalid),
    or ask for what's still missing (if incomplete).
    """
    reply_text, merged_details = llm.extract_booking_info(
        user_message=request.message,
        history=history,
        current_details=booking_state.details,
    )

    if not merged_details.is_complete():
        return reply_text, BookingState(status=BookingStatus.IN_PROGRESS, details=merged_details)

    try:
        normalized = validate_and_normalize(merged_details)
    except BookingValidationError as e:
        # Complete but invalid (e.g. bad email) — drop the bad field(s) so
        # the user is asked again, stay in the booking flow.
        correction_prompt = (
            f"{reply_text}\n\nA couple of things need fixing: {'; '.join(e.issues)}."
        )
        return correction_prompt, BookingState(status=BookingStatus.IN_PROGRESS, details=merged_details)

    booking = await postgres.save_booking(
        session_id=request.session_id,
        name=normalized.name,  # type: ignore[arg-type]
        email=normalized.email,  # type: ignore[arg-type]
        interview_date=normalized.date,  # type: ignore[arg-type]
        interview_time=normalized.time,  # type: ignore[arg-type]
    )

    confirmation = (
        f"{reply_text}\n\nYour interview is booked for {booking.interview_date} "
        f"at {booking.interview_time}. A confirmation reference is {booking.booking_id}."
    )
    return confirmation, BookingState(status=BookingStatus.CONFIRMED, details=normalized)


# ── Utility endpoints ─────────────────────────────────────────────────────────

@router.delete("/chat/{session_id}", summary="Clear a session's chat memory and booking state")
async def clear_session(session_id: str, redis: RedisChatMemory = Depends(get_redis)) -> dict[str, str]:
    await redis.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/bookings", response_model=list[BookingRecord], summary="List all confirmed bookings")
async def list_bookings(postgres: PostgresStore = Depends(get_postgres)) -> list[BookingRecord]:
    return await postgres.list_bookings()
