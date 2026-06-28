"""Pure unit tests for text extraction and booking validation."""

import pytest

from services.extractor import ExtractionError, TextExtractor
from services.booking import BookingValidationError, validate_and_normalize
from models.schemas import BookingDetails


def test_extract_txt_decodes_utf8():
    extractor = TextExtractor()
    document = extractor.extract(b"hello world", "notes.txt")

    assert document.raw_text == "hello world"
    assert document.filename == "notes.txt"
    assert document.total_pages is None


def test_extract_empty_txt_raises():
    extractor = TextExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract(b"   ", "empty.txt")


def test_extract_unsupported_extension_raises():
    from services.extractor import UnsupportedFileTypeError

    extractor = TextExtractor()
    with pytest.raises(UnsupportedFileTypeError):
        extractor.extract(b"data", "file.docx")


def test_booking_validation_normalizes_valid_details():
    details = BookingDetails(
        name="Alish Karki",
        email="alish@example.com",
        date="next Friday",
        time="3pm",
    )
    normalized = validate_and_normalize(details)

    assert normalized.email == "alish@example.com"
    assert len(normalized.date) == 10  # YYYY-MM-DD
    assert normalized.time is not None


def test_booking_validation_rejects_bad_email():
    details = BookingDetails(name="Alish", email="not-an-email", date="2026-07-01", time="10:00")

    with pytest.raises(BookingValidationError):
        validate_and_normalize(details)
