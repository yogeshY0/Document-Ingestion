"""
main.py

FastAPI application entry point.

Two responsibilities:
  1. App setup: create the FastAPI instance, configure metadata, mount routers.
  2. Lifespan: run startup and shutdown logic (DB table creation, etc.)

Why lifespan instead of @app.on_event("startup")?
  @app.on_event is deprecated in FastAPI 0.93+.
  Lifespan is the modern replacement — it's a context manager that wraps
  the entire app lifetime. Code before `yield` = startup. Code after = shutdown.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.v1.ingest import router as ingest_router
from api.v1.chat import router as chat_router
from db.postgres import PostgresStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create DB tables if they don't exist.
    Shutdown: nothing to clean up (connection pools auto-close).
    """
    postgres = PostgresStore()
    await postgres.create_tables()
    print("✓ PostgreSQL tables ready")
    print("✓ App started — visit http://localhost:8000/docs for API docs")

    yield  # App runs here

    # Shutdown logic (if needed in the future) goes here
    print("App shutting down")


app = FastAPI(
    title="Document RAG Backend",
    description=(
        "Two APIs: (1) Document Ingestion — upload PDF/TXT, chunk, embed, "
        "store in Qdrant + PostgreSQL. (2) Conversational RAG — multi-turn "
        "chat over ingested documents with Redis memory and LLM-driven "
        "interview booking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount all routes
app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe. Returns 200 if the app is running."""
    return {"status": "ok"}
