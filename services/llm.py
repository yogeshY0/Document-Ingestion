"""
services/llm.py

Every call to the language model in this project goes through here, and
every prompt is assembled by hand — string formatting and a messages list,
nothing more. That's the "custom RAG, no RetrievalQAChain" requirement:
there is no chain object deciding what to retrieve, how to format context,
or when to call a tool. This module does it explicitly so it's auditable.

Two responsibilities:
  1. generate_rag_answer  — answer a question grounded in retrieved chunks.
  2. extract_booking_info — incrementally pull {name, email, date, time}
     out of free-form conversation using OpenAI tool calling, across
     however many turns it takes.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from core.config import settings
from models.schemas import BookingDetails, ChatMessage, ChatRole, SourceChunk

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided "
    "context excerpts from the user's uploaded documents. "
    "If the answer isn't in the context, say you don't have enough information "
    "in the ingested documents to answer — do not make things up. "
    "Be concise and cite which excerpt(s) you used when relevant."
)

BOOKING_SYSTEM_PROMPT = (
    "You are helping the user schedule an interview. You need exactly four "
    "pieces of information: their full name, email address, preferred date, "
    "and preferred time. The user may give these all at once or one at a "
    "time, in any order, across several messages.\n\n"
    "Call the `update_booking_details` tool on every turn with whatever new "
    "values you can confidently extract from the user's latest message "
    "(leave a field null if it wasn't mentioned this turn). Already-known "
    "values are listed below as CURRENT_DETAILS — don't re-ask for those.\n\n"
    "After the tool call, also write a short, friendly reply: if fields are "
    "still missing, ask specifically for those (and only those). If "
    "everything is now filled in, briefly confirm the details back to the "
    "user.\n\n"
    "CURRENT_DETAILS: {current_details}"
)

# Tool/function schema the model can call to report extracted booking fields.
UPDATE_BOOKING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_booking_details",
        "description": (
            "Report any interview-booking fields (name, email, date, time) "
            "found in the user's latest message. Use null for fields not "
            "mentioned in this turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"], "description": "Full name of the interviewee"},
                "email": {"type": ["string", "null"], "description": "Email address"},
                "date": {"type": ["string", "null"], "description": "Preferred interview date, in any format the user gave"},
                "time": {"type": ["string", "null"], "description": "Preferred interview time, in any format the user gave"},
            },
            "required": ["name", "email", "date", "time"],
        },
    },
}


class LLMService:
    """
    Wraps the chat-completion client. No retrieval logic lives here — that's
    Retriever's job.

    Supports two interchangeable backends, both via the same `openai` SDK,
    because Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint:
      - "openai": api.openai.com, needs OPENAI_API_KEY, costs money.
      - "ollama": a local Ollama server (free, no key, runs offline).
    Whichever is selected, the rest of this class is identical — same method
    signatures, same tool-calling logic.
    """

    def __init__(self) -> None:
        if settings.llm_provider == "ollama":
            # Ollama ignores the API key but the SDK requires a non-empty string.
            self._client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
            self._model = settings.ollama_model
        else:
            if not settings.openai_api_key:
                raise ValueError(
                    "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                    "Either set the key, or switch LLM_PROVIDER=ollama in .env "
                    "to use a free local model instead."
                )
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_chat_model

    # ── RAG answer generation ───────────────────────────────────────────────

    def generate_rag_answer(
        self,
        query: str,
        history: list[ChatMessage],
        context_chunks: list[SourceChunk],
    ) -> str:
        """
        Manually assembles: [system prompt + context] + [prior turns] + [query],
        then calls chat.completions.create. This whole method IS the "chain" —
        written out instead of hidden behind an abstraction.
        """
        context_block = self._format_context(context_chunks)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": f"{RAG_SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block}"}
        ]
        messages.extend(self._history_to_openai_messages(history))
        messages.append({"role": "user", "content": query})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _format_context(chunks: list[SourceChunk]) -> str:
        if not chunks:
            return "(no relevant context was found in the ingested documents)"
        return "\n\n".join(
            f"[Source {i + 1} — {chunk.filename}]\n{chunk.text}" for i, chunk in enumerate(chunks)
        )

    @staticmethod
    def _history_to_openai_messages(history: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role.value, "content": msg.content} for msg in history]

    # ── Booking-slot extraction ─────────────────────────────────────────────

    def extract_booking_info(
        self,
        user_message: str,
        history: list[ChatMessage],
        current_details: BookingDetails,
    ) -> tuple[str, BookingDetails]:
        """
        One LLM call that both (a) extracts any new booking fields from the
        user's latest message via tool calling, and (b) drafts the natural
        -language reply (asking for what's missing, or confirming).

        Returns:
            (assistant_reply_text, merged_booking_details)
        """
        system_prompt = BOOKING_SYSTEM_PROMPT.format(
            current_details=current_details.model_dump_json()
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history_to_openai_messages(history))
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=[UPDATE_BOOKING_TOOL],
            tool_choice="auto",
            temperature=0,
        )

        choice = response.choices[0].message
        merged = current_details.model_copy()

        if choice.tool_calls:
            for tool_call in choice.tool_calls:
                if tool_call.function.name == "update_booking_details":
                    extracted = json.loads(tool_call.function.arguments)
                    merged = self._merge_details(merged, extracted)

        reply_text = choice.content
        if not reply_text:
            # Model called the tool but produced no text — synthesize a
            # reasonable fallback so the user always gets a response.
            reply_text = self._fallback_reply(merged)

        return reply_text, merged

    @staticmethod
    def _merge_details(current: BookingDetails, extracted: dict[str, Any]) -> BookingDetails:
        """Only overwrite a field if the model actually extracted a non-null value."""
        updated = current.model_dump()
        for field in ("name", "email", "date", "time"):
            value = extracted.get(field)
            if value:
                updated[field] = value
        return BookingDetails(**updated)

    @staticmethod
    def _fallback_reply(details: BookingDetails) -> str:
        missing = details.missing_fields()
        if not missing:
            return "Thanks — I have everything I need to book your interview."
        return f"Could you also share your {', '.join(missing)}?"