"""
core/config.py
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Chat LLM ─────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "ollama"] = "openai"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "local"] = "local"
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "document_chunks"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_dsn: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_chat_ttl_seconds: int = 60 * 60 * 24
    chat_history_max_messages: int = 20

    # ── Ingestion ─────────────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    allowed_extensions: set[str] = {".pdf", ".txt"}
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50
    semantic_similarity_threshold: float = 0.3

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 4


settings = Settings()