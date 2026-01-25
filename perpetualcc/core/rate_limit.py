"""Rate limit detection and monitoring for Claude Code sessions."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

from perpetualcc.claude.types import ClaudeEvent, RateLimitEvent, ResultEvent
from perpetualcc.core.rate_limit_utils import (
    extract_reset_time,
    is_rate_limit_message,
)

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Type of rate limit encountered."""

    TOKEN_LIMIT = "token_limit"  # Context window or token limit
    REQUEST_LIMIT = "request_limit"  # Too many requests
    USAGE_LIMIT = "usage_limit"  # API usage quota exceeded
    OVERLOADED = "overloaded"  # Server overloaded
    UNKNOWN = "unknown"  # Could not determine type


@dataclass(frozen=True)
class RateLimitInfo:
    """Information about a detected rate limit.

    Attributes:
        detected_at: When the rate limit was detected
        limit_type: Type of rate limit (token, request, usage, etc.)
        retry_after_seconds: Suggested wait time in seconds
        reset_time: Estimated time when the limit will reset
        message: Original error message from the API
        raw_event: The original event that triggered detection
    """

    detected_at: datetime
    limit_type: RateLimitType
    retry_after_seconds: int
    reset_time: datetime | None
    message: str
    raw_event: ClaudeEvent | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the rate limit has expired (reset time has passed)."""
        if self.reset_time is None:
            return False
        return datetime.now() >= self.reset_time

    @property
    def remaining_seconds(self) -> int:
        """Get remaining seconds until reset (0 if expired)."""
        if self.reset_time is None:
            return self.retry_after_seconds
        remaining = (self.reset_time - datetime.now()).total_seconds()
        return max(0, int(remaining))


@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for rate limit detection and handling.

    Attributes:
        default_retry_seconds: Default wait time when retry-after not specified
        max_retry_seconds: Maximum wait time to prevent excessive delays
        min_retry_seconds: Minimum wait time to ensure proper backoff
        backoff_multiplier: Multiplier for exponential backoff on repeated limits
    """

    default_retry_seconds: int = 60
    max_retry_seconds: int = 300  # 5 minutes max
    min_retry_seconds: int = 10
    backoff_multiplier: float = 1.5


# Patterns for detecting rate limit types from error messages
_RATE_LIMIT_PATTERNS: tuple[tuple[re.Pattern[str], RateLimitType], ...] = (
    (re.compile(r"token.?limit", re.IGNORECASE), RateLimitType.TOKEN_LIMIT),
    (re.compile(r"context.?(?:window|length)", re.IGNORECASE), RateLimitType.TOKEN_LIMIT),
    (re.compile(r"too\s+many\s+requests", re.IGNORECASE), RateLimitType.REQUEST_LIMIT),
    (re.compile(r"429", re.IGNORECASE), RateLimitType.REQUEST_LIMIT),
    (re.compile(r"rate\s*limit", re.IGNORECASE), RateLimitType.REQUEST_LIMIT),
    (re.compile(r"usage\s*limit", re.IGNORECASE), RateLimitType.USAGE_LIMIT),
    (re.compile(r"quota", re.IGNORECASE), RateLimitType.USAGE_LIMIT),
    (re.compile(r"overloaded", re.IGNORECASE), RateLimitType.OVERLOADED),
    (re.compile(r"capacity", re.IGNORECASE), RateLimitType.OVERLOADED),
)

# Patterns for extracting retry-after seconds from messages
_RETRY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"retry.?after[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"wait[:\s]+(\d+)\s*(?:second|sec)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:second|sec)(?:s)?\s*(?:later|wait)", re.IGNORECASE),
    re.compile(r"try\s+again\s+in\s+(\d+)", re.IGNORECASE),
)


class RateLimitDetector:
    """Detects rate limits from Claude events and error messages.

    This class analyzes events and messages to determine if a rate limit
    has been hit and extracts relevant information for handling.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize the rate limit detector.

        Args:
            config: Optional configuration for detection behavior
        """
        self.config = config or RateLimitConfig()

    def detect(self, event: ClaudeEvent) -> RateLimitInfo | None:
        """Detect if an event indicates a rate limit.

        Args:
            event: A Claude event to analyze

        Returns:
            RateLimitInfo if a rate limit was detected, None otherwise
        """
        if isinstance(event, RateLimitEvent):
            return self._from_rate_limit_event(event)
        elif isinstance(event, ResultEvent):
            return self._from_result_event(event)
        return None

    def detect_from_message(self, message: str) -> RateLimitInfo | None:
        """Detect rate limit from an error message string.

        Args:
            message: Error message to analyze

        Returns:
            RateLimitInfo if a rate limit was detected, None otherwise
        """
        if not message:
            return None

        # Check if this looks like a rate limit message (use utility function)
        limit_type = self._detect_limit_type(message)
        if limit_type == RateLimitType.UNKNOWN and not is_rate_limit_message(message):
            return None

        # First try to extract a specific reset time (use utility function)
        reset_time = extract_reset_time(message)
        if reset_time:
            retry_seconds = int((reset_time - datetime.now()).total_seconds())
            retry_seconds = max(self.config.min_retry_seconds, retry_seconds)
        else:
            # Fall back to extracting retry-after seconds
            retry_seconds = self._extract_retry_seconds(message)
            reset_time = datetime.now() + timedelta(seconds=retry_seconds)

        return RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=limit_type,
            retry_after_seconds=retry_seconds,
            reset_time=reset_time,
            message=message,
            raw_event=None,
        )

    def _from_rate_limit_event(self, event: RateLimitEvent) -> RateLimitInfo:
        """Create RateLimitInfo from a RateLimitEvent."""
        limit_type = self._detect_limit_type(event.message)
        if event.retry_after > 0:
            retry_seconds = event.retry_after
        else:
            retry_seconds = self.config.default_retry_seconds
        retry_seconds = self._clamp_retry_seconds(retry_seconds)

        reset_time = event.reset_time
        if reset_time is None:
            reset_time = datetime.now() + timedelta(seconds=retry_seconds)

        return RateLimitInfo(
            detected_at=event.timestamp,
            limit_type=limit_type,
            retry_after_seconds=retry_seconds,
            reset_time=reset_time,
            message=event.message,
            raw_event=event,
        )

    def _from_result_event(self, event: ResultEvent) -> RateLimitInfo | None:
        """Create RateLimitInfo from a ResultEvent if it indicates rate limiting."""
        if not event.is_error or not event.result:
            return None

        message = event.result
        if not is_rate_limit_message(message):
            return None

        limit_type = self._detect_limit_type(message)

        # Use utility function for time extraction
        reset_time = extract_reset_time(message)
        if reset_time:
            retry_seconds = int((reset_time - datetime.now()).total_seconds())
            retry_seconds = max(self.config.min_retry_seconds, retry_seconds)
        else:
            retry_seconds = self._extract_retry_seconds(message)
            reset_time = datetime.now() + timedelta(seconds=retry_seconds)

        return RateLimitInfo(
            detected_at=event.timestamp,
            limit_type=limit_type,
            retry_after_seconds=retry_seconds,
            reset_time=reset_time,
            message=message,
            raw_event=event,
        )

    def _detect_limit_type(self, message: str) -> RateLimitType:
        """Detect the type of rate limit from the message."""
        for pattern, limit_type in _RATE_LIMIT_PATTERNS:
            if pattern.search(message):
                return limit_type
        return RateLimitType.UNKNOWN

    def _extract_retry_seconds(self, message: str) -> int:
        """Extract retry-after seconds from a message."""
        # First, try to extract seconds-based retry values
        for pattern in _RETRY_PATTERNS:
            match = pattern.search(message)
            if match:
                try:
                    seconds = int(match.group(1))
                    return self._clamp_retry_seconds(seconds)
                except (ValueError, IndexError):
                    continue

        # Try to extract time-based reset using utility function
        reset_time = extract_reset_time(message)
        if reset_time:
            now = datetime.now()
            if reset_time > now:
                seconds = int((reset_time - now).total_seconds())
                # For time-based resets, allow longer waits (up to 24 hours)
                return max(self.config.min_retry_seconds, min(seconds, 86400))

        return self.config.default_retry_seconds

    def _clamp_retry_seconds(self, seconds: int) -> int:
        """Clamp retry seconds to configured bounds."""
        return max(self.config.min_retry_seconds, min(seconds, self.config.max_retry_seconds))


# Type alias for progress callback
ProgressCallback = Callable[[int, int], None]


class RateLimitMonitor:
    """Monitors rate limits and provides waiting/countdown functionality.

    This class manages the state of detected rate limits and provides
    async methods for waiting until limits reset.
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        detector: RateLimitDetector | None = None,
    ):
        """Initialize the rate limit monitor.

        Args:
            config: Optional configuration for rate limit handling
            detector: Optional custom detector instance
        """
        self.config = config or RateLimitConfig()
        self.detector = detector or RateLimitDetector(self.config)
        self._current_limit: RateLimitInfo | None = None
        self._consecutive_limits: int = 0
        self._limit_history: list[RateLimitInfo] = []

    @property
    def current_limit(self) -> RateLimitInfo | None:
        """Get the current active rate limit, if any."""
        if self._current_limit and self._current_limit.is_expired:
            self._current_limit = None
        return self._current_limit

    @property
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return self.current_limit is not None

    @property
    def consecutive_limits(self) -> int:
        """Get the count of consecutive rate limits hit."""
        return self._consecutive_limits

    def detect(self, event: ClaudeEvent) -> RateLimitInfo | None:
        """Detect and track rate limits from events.

        Args:
            event: A Claude event to analyze

        Returns:
            RateLimitInfo if a rate limit was detected, None otherwise
        """
        info = self.detector.detect(event)
        if info:
            self._record_limit(info)
        return info

    def detect_from_message(self, message: str) -> RateLimitInfo | None:
        """Detect and track rate limits from error messages.

        Args:
            message: Error message to analyze

        Returns:
            RateLimitInfo if a rate limit was detected, None otherwise
        """
        info = self.detector.detect_from_message(message)
        if info:
            self._record_limit(info)
        return info

    def _record_limit(self, info: RateLimitInfo) -> None:
        """Record a detected rate limit."""
        self._current_limit = info
        self._consecutive_limits += 1
        self._limit_history.append(info)

        logger.warning(
            "Rate limit detected: type=%s, retry_after=%ds, message=%s",
            info.limit_type.value,
            info.retry_after_seconds,
            info.message[:100],
        )

    def clear(self) -> None:
        """Clear the current rate limit state after successful operation."""
        self._current_limit = None
        self._consecutive_limits = 0

    def get_adjusted_retry_seconds(self) -> int:
        """Get retry seconds with exponential backoff for consecutive limits.

        Returns:
            Adjusted retry seconds based on consecutive limit count
        """
        if not self._current_limit:
            return self.config.default_retry_seconds

        base = self._current_limit.retry_after_seconds
        if self._consecutive_limits <= 1:
            return base

        # Apply exponential backoff for consecutive limits
        multiplier = self.config.backoff_multiplier ** (self._consecutive_limits - 1)
        adjusted = int(base * multiplier)
        return min(adjusted, self.config.max_retry_seconds)

    async def wait_for_reset(
        self,
        info: RateLimitInfo | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Wait for the rate limit to reset.

        Args:
            info: Optional specific rate limit info to wait for.
                  If None, uses current limit.
            progress_callback: Optional callback for progress updates.
                             Called with (remaining_seconds, total_seconds).
        """
        target_info = info or self._current_limit
        if not target_info:
            return

        # Use adjusted retry seconds for backoff
        total_seconds = self.get_adjusted_retry_seconds()
        remaining = total_seconds

        logger.info(
            "Waiting %d seconds for rate limit reset (consecutive=%d)",
            total_seconds,
            self._consecutive_limits,
        )

        while remaining > 0:
            if progress_callback:
                progress_callback(remaining, total_seconds)

            # Sleep in 1-second intervals for responsive progress updates
            await asyncio.sleep(1)
            remaining -= 1

        # Final callback at 0
        if progress_callback:
            progress_callback(0, total_seconds)

        logger.info("Rate limit wait complete")

    def get_history(self, limit: int | None = None) -> list[RateLimitInfo]:
        """Get rate limit history.

        Args:
            limit: Optional maximum number of entries to return

        Returns:
            List of RateLimitInfo objects, most recent last
        """
        if limit:
            return list(self._limit_history[-limit:])
        return list(self._limit_history)

    def get_statistics(self) -> dict:
        """Get statistics about rate limit occurrences.

        Returns:
            Dictionary with rate limit statistics
        """
        if not self._limit_history:
            return {
                "total_limits": 0,
                "by_type": {},
                "current_limit": None,
                "consecutive_limits": self._consecutive_limits,
            }

        by_type: dict[str, int] = {}
        for info in self._limit_history:
            type_name = info.limit_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_limits": len(self._limit_history),
            "by_type": by_type,
            "current_limit": self._current_limit,
            "consecutive_limits": self._consecutive_limits,
        }
