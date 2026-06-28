"""
db/redis_store.py

Redis is used for exactly one thing: short-lived, per-session conversational
state. Two kinds of state, two key prefixes:

  chat:history:{session_id}   -> a Redis LIST of JSON-encoded ChatMessage,
                                  used as the multi-turn memory fed into the
                                  LLM prompt.
  chat:booking:{session_id}   -> a Redis STRING holding the JSON-encoded
                                  BookingState (status + whatever slots have
                                  been collected so far), used to resume the
                                  interview-booking flow across turns.

Why Redis and not Postgres for this? It's ephemeral, high-churn, per-session
data — exactly Redis's sweet spot. TTL means abandoned sessions clean
themselves up automatically.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from core.config import settings
from models.schemas import BookingState, ChatMessage, ChatRole


class RedisChatMemory:
    """Owns a single Redis connection pool, reused across requests."""

    def __init__(self) -> None:
        self._client = aioredis.from_url(settings.redis_url, decode_responses=True)

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:history:{session_id}"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:booking:{session_id}"

    # ── Chat history ────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return all stored turns for this session, oldest first."""
        raw_messages = await self._client.lrange(self._history_key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(raw) for raw in raw_messages]

    async def append_message(self, session_id: str, role: ChatRole, content: str) -> None:
        """
        Push a new turn onto the history list, trim to the configured window,
        and refresh the TTL so active sessions never expire mid-conversation.
        """
        key = self._history_key(session_id)
        message = ChatMessage(role=role, content=content)

        await self._client.rpush(key, message.model_dump_json())
        # Keep only the most recent N messages — bounds prompt size & cost.
        await self._client.ltrim(key, -settings.chat_history_max_messages, -1)
        await self._client.expire(key, settings.redis_chat_ttl_seconds)

    # ── Booking flow state ───────────────────────────────────────────────────

    async def get_booking_state(self, session_id: str) -> BookingState:
        raw = await self._client.get(self._booking_key(session_id))
        if raw is None:
            return BookingState()
        return BookingState.model_validate_json(raw)

    async def set_booking_state(self, session_id: str, state: BookingState) -> None:
        await self._client.set(
            self._booking_key(session_id),
            state.model_dump_json(),
            ex=settings.redis_chat_ttl_seconds,
        )

    # ── Housekeeping ──────────────────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> None:
        await self._client.delete(self._history_key(session_id), self._booking_key(session_id))

    async def close(self) -> None:
        await self._client.aclose()
