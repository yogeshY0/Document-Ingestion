"""
db/postgres.py

SQL metadata store, using SQLAlchemy 2.0's async engine + asyncpg driver.

What lives here (NOT vectors — those are in Qdrant):
  - documents:        one row per ingested file (filename, size, strategy, ...)
  - chunk_metadata:    one row per chunk (for auditing / re-indexing without
                       needing to read Qdrant — chunk text is duplicated here
                       deliberately, it's cheap and avoids a round trip)
  - bookings:          confirmed interview bookings from the chat API

Async SQLAlchemy is used throughout so neither API ever blocks the event
loop on a DB call.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core.config import settings
from models.schemas import ChunkRecord, ChunkStrategy, DocumentRecord, BookingRecord


class Base(DeclarativeBase):
    pass


# ── ORM models ──────────────────────────────────────────────────────────────

class DocumentORM(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strategy_used: Mapped[str] = mapped_column(String(32))
    embedding_provider: Mapped[str] = mapped_column(String(32))
    chunks_count: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["ChunkMetadataORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkMetadataORM(Base):
    __tablename__ = "chunk_metadata"

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[DocumentORM] = relationship(back_populates="chunks")


class BookingORM(Base):
    __tablename__ = "bookings"

    booking_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(256))
    email: Mapped[str] = mapped_column(String(256))
    interview_date: Mapped[str] = mapped_column(String(32))
    interview_time: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


# ── Store ────────────────────────────────────────────────────────────────────

class PostgresStore:
    """
    Owns the async engine + session factory. Instantiated once (see
    Depends() providers) and reused — creating a new engine per request
    would exhaust connections.
    """

    def __init__(self) -> None:
        self._engine = create_async_engine(settings.postgres_dsn, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        strategy: ChunkStrategy,
        embedding_provider: str,
    ) -> None:
        """Persist document + per-chunk metadata in a single transaction."""
        async with self._session_factory() as session:
            session.add(
                DocumentORM(
                    document_id=document.document_id,
                    filename=document.filename,
                    file_size_bytes=document.file_size_bytes,
                    total_pages=document.total_pages,
                    strategy_used=strategy.value,
                    embedding_provider=embedding_provider,
                    chunks_count=len(chunks),
                    ingested_at=datetime.now(timezone.utc),
                )
            )
            session.add_all(
                [
                    ChunkMetadataORM(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        char_count=chunk.char_count,
                        page_number=chunk.page_number,
                        text=chunk.text,
                    )
                    for chunk in chunks
                ]
            )
            await session.commit()

    async def save_booking(
        self,
        session_id: str,
        name: str,
        email: str,
        interview_date: str,
        interview_time: str,
    ) -> BookingRecord:
        """Persist a confirmed interview booking and return it as a schema."""
        booking_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            session.add(
                BookingORM(
                    booking_id=booking_id,
                    session_id=session_id,
                    name=name,
                    email=email,
                    interview_date=interview_date,
                    interview_time=interview_time,
                    created_at=created_at,
                )
            )
            await session.commit()

        return BookingRecord(
            booking_id=booking_id,
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            created_at=created_at,
        )

    async def list_bookings(self) -> list[BookingRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(BookingORM))
            rows = result.scalars().all()
            return [
                BookingRecord(
                    booking_id=row.booking_id,
                    session_id=row.session_id,
                    name=row.name,
                    email=row.email,
                    interview_date=row.interview_date,
                    interview_time=row.interview_time,
                    created_at=row.created_at,
                )
                for row in rows
            ]
