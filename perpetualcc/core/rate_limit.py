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
        max_retry_seconds: Maximum wait time for request/overload limits
        max_token_limit_seconds: Maximum wait time for token limits (can be hours)
        min_retry_seconds: Minimum wait time to ensure proper backoff
        backoff_multiplier: Multiplier for exponential backoff on repeated limits
    """

    default_retry_seconds: int = 60
    max_retry_seconds: int = 300  # 5 minutes max for request limits
    max_token_limit_seconds: int = 86400  # 24 hours max for token limits
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

    def get_adjusted_retry_seconds(self, limit_info: RateLimitInfo | None = None) -> int:
        """Get retry seconds with exponential backoff for consecutive limits.

        For TOKEN_LIMIT types, respects the actual reset_time which may be hours away.
        For other types, applies capped backoff.

        Args:
            limit_info: Optional specific limit info. If None, uses current limit.

        Returns:
            Adjusted retry seconds based on limit type and consecutive count
        """
        target_limit = limit_info or self._current_limit
        if not target_limit:
            return self.config.default_retry_seconds

        # For token limits, respect the actual reset time (can be hours)
        if target_limit.limit_type == RateLimitType.TOKEN_LIMIT:
            if target_limit.reset_time:
                remaining = target_limit.remaining_seconds
                # Clamp to max_token_limit_seconds (24 hours)
                return min(remaining, self.config.max_token_limit_seconds)
            # Fallback: use retry_after_seconds without strict cap
            return min(target_limit.retry_after_seconds, self.config.max_token_limit_seconds)

        # For other limit types (request, usage, overload), apply capped backoff
        base = target_limit.retry_after_seconds
        if self._consecutive_limits <= 1:
            return min(base, self.config.max_retry_seconds)

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

        For TOKEN_LIMIT types with a specific reset_time, waits until that exact
        time arrives. This avoids any premature retries before the actual reset.

        For other types or when no reset_time is available, uses countdown-based
        waiting with efficient sleep intervals.

        Args:
            info: Optional specific rate limit info to wait for.
                  If None, uses current limit.
            progress_callback: Optional callback for progress updates.
                             Called with (remaining_seconds, total_seconds).
        """
        target_info = info or self._current_limit
        if not target_info:
            return

        # Determine if we should wait for a specific reset_time or use countdown
        use_reset_time = (
            target_info.reset_time is not None
            and target_info.limit_type == RateLimitType.TOKEN_LIMIT
        )

        if use_reset_time:
            await self._wait_until_reset_time(target_info, progress_callback)
        else:
            await self._wait_countdown(target_info, progress_callback)

        logger.info("Rate limit wait complete")

    async def _wait_until_reset_time(
        self,
        target_info: RateLimitInfo,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Wait until the exact reset_time arrives.

        This method continuously checks the actual reset_time rather than
        counting down a static duration, ensuring no premature retries.
        """
        reset_time = target_info.reset_time
        assert reset_time is not None  # Guaranteed by caller

        # Check if reset time has already passed
        now = datetime.now()
        if now >= reset_time:
            logger.info("Token limit reset time has already passed, proceeding immediately")
            if progress_callback:
                progress_callback(0, 0)
            return

        # Calculate initial total for progress reporting
        initial_remaining = max(0, int((reset_time - now).total_seconds()))
        total_seconds = initial_remaining

        # Cap at 24 hours
        if total_seconds > self.config.max_token_limit_seconds:
            total_seconds = self.config.max_token_limit_seconds
            # Adjust reset_time for capped wait
            reset_time = datetime.now() + timedelta(seconds=total_seconds)

        hours = total_seconds / 3600
        logger.info(
            "Token limit hit. Waiting until reset at %s (%.1f hours)",
            reset_time.strftime("%Y-%m-%d %H:%M:%S"),
            hours,
        )

        last_progress_log = total_seconds

        # Wait until reset_time arrives (not countdown-based)
        while datetime.now() < reset_time:
            remaining = max(0, int((reset_time - datetime.now()).total_seconds()))

            if progress_callback:
                progress_callback(remaining, total_seconds)

            # Smart sleep intervals based on remaining time
            if remaining > 3600:
                sleep_interval = 300  # 5 minutes
            elif remaining > 300:
                sleep_interval = 60  # 1 minute
            else:
                sleep_interval = 1  # 1 second

            # Don't sleep past reset_time
            sleep_interval = min(sleep_interval, remaining) if remaining > 0 else 1

            await asyncio.sleep(sleep_interval)

            # Log progress periodically for long waits
            remaining_after = max(0, int((reset_time - datetime.now()).total_seconds()))
            if remaining_after > 300 and (last_progress_log - remaining_after) >= 300:
                hours_remaining = remaining_after / 3600
                logger.info(
                    "Rate limit wait: %.1f hours remaining",
                    hours_remaining,
                )
                last_progress_log = remaining_after

        # Final callback at 0
        if progress_callback:
            progress_callback(0, total_seconds)

    async def _wait_countdown(
        self,
        target_info: RateLimitInfo,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Wait using countdown-based approach for non-token limits.

        Used when there's no specific reset_time or for non-token limit types.
        """
        total_seconds = self.get_adjusted_retry_seconds(target_info)

        logger.info(
            "Waiting %d seconds for rate limit reset (type=%s, consecutive=%d)",
            total_seconds,
            target_info.limit_type.value,
            self._consecutive_limits,
        )

        remaining = total_seconds
        last_progress_log = total_seconds

        while remaining > 0:
            if progress_callback:
                progress_callback(remaining, total_seconds)

            # Smart sleep intervals
            if remaining > 3600:
                sleep_interval = 300  # 5 minutes
            elif remaining > 300:
                sleep_interval = 60  # 1 minute
            else:
                sleep_interval = 1  # 1 second

            # Don't sleep longer than remaining time
            sleep_interval = min(sleep_interval, remaining)

            await asyncio.sleep(sleep_interval)
            remaining -= sleep_interval

            # Log progress periodically for long waits
            if remaining > 300 and (last_progress_log - remaining) >= 300:
                hours_remaining = remaining / 3600
                logger.info(
                    "Rate limit wait: %.1f hours remaining",
                    hours_remaining,
                )
                last_progress_log = remaining

        # Final callback at 0
        if progress_callback:
            progress_callback(0, total_seconds)

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
