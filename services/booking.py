"""
services/booking.py

Validation lives separately from extraction (services/llm.py) on purpose:
the LLM is good at pulling loosely-formatted info out of natural language,
but it's not the right place to enforce "is this actually a valid email" or
"is this date in the future" — that's deterministic logic and belongs in
plain Python, not a prompt.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from dateutil import parser as dateutil_parser

from models.schemas import BookingDetails

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BookingValidationError(Exception):
    """Raised with a list of human-readable problems with the booking details."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def validate_and_normalize(details: BookingDetails) -> BookingDetails:
    """
    Takes a *complete* BookingDetails (all four fields present) and returns a
    normalized copy (date -> YYYY-MM-DD, time -> HH:MM 24h), or raises
    BookingValidationError describing every problem found.
    """
    issues: list[str] = []

    name = (details.name or "").strip()
    if len(name) < 2:
        issues.append("name looks too short — please provide your full name")

    email = (details.email or "").strip()
    if not _EMAIL_RE.match(email):
        issues.append(f"'{email}' doesn't look like a valid email address")

    normalized_date: str | None = None
    try:
        parsed_date = dateutil_parser.parse(details.date or "", fuzzy=True).date()
        if parsed_date < date.today():
            issues.append("the interview date can't be in the past")
        else:
            normalized_date = parsed_date.isoformat()
    except (ValueError, OverflowError):
        issues.append(f"couldn't understand the date '{details.date}'")

    normalized_time: str | None = None
    try:
        parsed_time = dateutil_parser.parse(details.time or "", fuzzy=True).time()
        normalized_time = parsed_time.strftime("%H:%M")
    except (ValueError, OverflowError):
        issues.append(f"couldn't understand the time '{details.time}'")

    if issues:
        raise BookingValidationError(issues)

    return BookingDetails(name=name, email=email, date=normalized_date, time=normalized_time)
