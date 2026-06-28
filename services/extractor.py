"""
services/extractor.py

Responsibility: take raw file bytes → return clean text + metadata.

Two supported formats:
  PDF  →  extracted via PyMuPDF (fitz). Preserves page numbers.
  TXT  →  read directly. Handles encoding gracefully.

Why PyMuPDF (fitz)?
  - Fastest Python PDF library (C bindings under the hood).
  - Gives per-page text, so we can store which page each chunk came from.
  - Handles multi-column PDFs better than pdfplumber.
  - MIT licensed — safe for commercial use.

This service is PURE — it takes bytes, returns a DocumentRecord.
No DB calls, no embedding. Just extraction.
"""

import fitz  # PyMuPDF — the import name is 'fitz', not 'pymupdf'
import uuid
from pathlib import Path

from models.schemas import DocumentRecord
from core.config import settings


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension is not in settings.allowed_extensions."""
    pass


class ExtractionError(Exception):
    """Raised when text extraction fails (corrupt PDF, encoding issues, etc.)."""
    pass


class TextExtractor:
    """
    Stateless extractor. Every method is a pure function over its inputs.
    Instantiate once at app startup and reuse — no shared state.
    """

    def extract(self, file_bytes: bytes, filename: str) -> DocumentRecord:
        """
        Main entry point. Routes to the correct extractor based on extension.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Original filename (used to determine type and for metadata).

        Returns:
            DocumentRecord with full extracted text and metadata.

        Raises:
            UnsupportedFileTypeError: extension not in allowed list.
            ExtractionError: extraction failed for any reason.
        """
        self._validate_extension(filename)

        extension = Path(filename).suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf(file_bytes, filename)
        elif extension == ".txt":
            return self._extract_txt(file_bytes, filename)
        else:
            # Should never reach here because _validate_extension catches it,
            # but mypy/pyright would complain without this branch.
            raise UnsupportedFileTypeError(f"No extractor for: {extension}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_extension(self, filename: str) -> None:
        """Check the file extension against the allowlist in settings."""
        extension = Path(filename).suffix.lower()
        if extension not in settings.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"File type '{extension}' is not supported. "
                f"Allowed: {settings.allowed_extensions}"
            )

    def _extract_pdf(self, file_bytes: bytes, filename: str) -> DocumentRecord:
        """
        Extract text from a PDF using PyMuPDF.

        Strategy:
          - Open the PDF from bytes (no temp file needed).
          - Iterate each page, extract text.
          - Join all pages with double newlines.
          - Store page count for metadata.

        We store page-level text in a structured way initially, then flatten
        to full text for chunking. This lets us later preserve page numbers
        per chunk if needed.
        """
        try:
            # fitz.open() can open from bytes by specifying stream= and filetype=
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise ExtractionError(f"Could not open PDF '{filename}': {e}") from e

        if doc.page_count == 0:
            raise ExtractionError(f"PDF '{filename}' has no pages.")

        pages_text: list[str] = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            # get_text("text") → plain text, strips formatting.
            # get_text("blocks") → preserves layout blocks (useful for tables).
            # We use plain text — layout isn't important for RAG.
            page_text = page.get_text("text").strip()

            if page_text:  # Skip blank pages
                pages_text.append(page_text)

        doc.close()

        if not pages_text:
            raise ExtractionError(f"PDF '{filename}' contains no extractable text.")

        # Join pages with a separator that the chunker can see
        full_text = "\n\n".join(pages_text)

        return DocumentRecord(
            document_id=str(uuid.uuid4()),
            filename=filename,
            file_size_bytes=len(file_bytes),
            raw_text=full_text,
            total_pages=len(pages_text),
        )

    def _extract_txt(self, file_bytes: bytes, filename: str) -> DocumentRecord:
        """
        Extract text from a plain text file.

        Encoding strategy:
          1. Try UTF-8 (most common for modern files).
          2. Fall back to latin-1 (covers older Windows text files).
          latin-1 never fails because it maps every byte 0x00-0xFF to a character.
          This is a deliberate fallback, not lazy coding.
        """
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # latin-1 is the safe fallback — it decodes every possible byte
            text = file_bytes.decode("latin-1")

        text = text.strip()

        if not text:
            raise ExtractionError(f"Text file '{filename}' is empty.")

        return DocumentRecord(
            document_id=str(uuid.uuid4()),
            filename=filename,
            file_size_bytes=len(file_bytes),
            raw_text=text,
            total_pages=None,  # TXT files have no concept of pages
        )
