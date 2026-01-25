"""Utility functions for rate limit detection and parsing.

This module contains pure parsing functions without dependencies on Claude types,
avoiding circular imports when used by the adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ParsedRateLimit:
    """Parsed rate limit information from an error message.

    Attributes:
        is_rate_limit: Whether the message indicates a rate limit
        retry_after_seconds: Suggested wait time in seconds
        reset_time: Estimated time when the limit will reset
        message: Original error message
    """

    is_rate_limit: bool
    retry_after_seconds: int
    reset_time: datetime | None
    message: str


# Default configuration values
DEFAULT_RETRY_SECONDS = 60
MIN_RETRY_SECONDS = 10
MAX_RETRY_SECONDS = 300
MAX_TIME_BASED_RETRY_SECONDS = 86400  # 24 hours for time-based resets

# Phrases that indicate a rate limit error
_RATE_LIMIT_PHRASES = (
    "rate limit",
    "429",
    "usage limit",
    "token limit",
    "too many requests",
    "overloaded",
    "quota",
    "hit your limit",
    "you've hit your limit",
)

# Patterns for extracting retry-after seconds from messages
_RETRY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"retry.?after[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"wait[:\s]+(\d+)\s*(?:second|sec)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:second|sec)(?:s)?\s*(?:later|wait)", re.IGNORECASE),
    re.compile(r"try\s+again\s+in\s+(\d+)", re.IGNORECASE),
)

# Patterns for extracting time-based reset from messages (e.g., "resets 2pm")
# Matches formats like: "resets 2pm", "resets 2:30pm", "resets 14:00"
_TIME_RESET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 12-hour format: "resets 2pm", "resets 2:30pm", "resets 2:30 pm"
    re.compile(
        r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        re.IGNORECASE,
    ),
    # 24-hour format: "resets 14:00", "resets 14:30"
    re.compile(
        r"resets?\s+(\d{1,2}):(\d{2})(?!\s*(?:am|pm))",
        re.IGNORECASE,
    ),
)


def is_rate_limit_message(message: str) -> bool:
    """Check if a message indicates a rate limit error.

    Args:
        message: Error message to check

    Returns:
        True if the message indicates a rate limit
    """
    if not message:
        return False
    lower = message.lower()
    return any(phrase in lower for phrase in _RATE_LIMIT_PHRASES)


def extract_reset_time(message: str) -> datetime | None:
    """Extract a reset time from a message (e.g., 'resets 2pm').

    Args:
        message: The error message to parse

    Returns:
        datetime of reset time, or None if not found
    """
    if not message:
        return None

    now = datetime.now()

    for pattern in _TIME_RESET_PATTERNS:
        match = pattern.search(message)
        if match:
            try:
                groups = match.groups()

                # Check if this is 12-hour or 24-hour format
                if len(groups) >= 3 and groups[2]:  # 12-hour format with am/pm
                    hour = int(groups[0])
                    minute = int(groups[1]) if groups[1] else 0
                    am_pm = groups[2].lower()

                    # Convert to 24-hour format
                    if am_pm == "pm" and hour != 12:
                        hour += 12
                    elif am_pm == "am" and hour == 12:
                        hour = 0
                else:  # 24-hour format
                    hour = int(groups[0])
                    minute = int(groups[1]) if groups[1] else 0

                # Create reset time for today
                reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # If the time has already passed today, assume it's tomorrow
                if reset_time <= now:
                    reset_time += timedelta(days=1)

                return reset_time

            except (ValueError, IndexError):
                continue

    return None


def extract_retry_seconds(
    message: str,
    default: int = DEFAULT_RETRY_SECONDS,
    min_seconds: int = MIN_RETRY_SECONDS,
    max_seconds: int = MAX_RETRY_SECONDS,
) -> int:
    """Extract retry-after seconds from a message.

    Args:
        message: The error message to parse
        default: Default retry seconds if not extractable
        min_seconds: Minimum allowed retry seconds
        max_seconds: Maximum allowed retry seconds

    Returns:
        Number of seconds to wait before retrying
    """
    if not message:
        return default

    # First, try to extract seconds-based retry values
    for pattern in _RETRY_PATTERNS:
        match = pattern.search(message)
        if match:
            try:
                seconds = int(match.group(1))
                return max(min_seconds, min(seconds, max_seconds))
            except (ValueError, IndexError):
                continue

    # Try to extract time-based reset (e.g., "resets 2pm")
    reset_time = extract_reset_time(message)
    if reset_time:
        now = datetime.now()
        if reset_time > now:
            seconds = int((reset_time - now).total_seconds())
            # For time-based resets, allow longer waits (up to 24 hours)
            return max(min_seconds, min(seconds, MAX_TIME_BASED_RETRY_SECONDS))

    return default


def parse_rate_limit_message(
    message: str,
    default_retry: int = DEFAULT_RETRY_SECONDS,
    min_retry: int = MIN_RETRY_SECONDS,
    max_retry: int = MAX_RETRY_SECONDS,
) -> ParsedRateLimit:
    """Parse a rate limit message and extract all relevant information.

    This is the main entry point for parsing rate limit messages from error strings.

    Args:
        message: The error message to parse
        default_retry: Default retry seconds if not extractable
        min_retry: Minimum allowed retry seconds
        max_retry: Maximum allowed retry seconds

    Returns:
        ParsedRateLimit with all extracted information
    """
    if not message or not is_rate_limit_message(message):
        return ParsedRateLimit(
            is_rate_limit=False,
            retry_after_seconds=default_retry,
            reset_time=None,
            message=message or "",
        )

    # First try to extract a specific reset time (e.g., "resets 2pm")
    reset_time = extract_reset_time(message)
    if reset_time:
        retry_seconds = int((reset_time - datetime.now()).total_seconds())
        retry_seconds = max(min_retry, retry_seconds)
    else:
        # Fall back to extracting retry-after seconds
        retry_seconds = extract_retry_seconds(message, default_retry, min_retry, max_retry)
        reset_time = datetime.now() + timedelta(seconds=retry_seconds)

    return ParsedRateLimit(
        is_rate_limit=True,
        retry_after_seconds=retry_seconds,
        reset_time=reset_time,
        message=message,
    )
