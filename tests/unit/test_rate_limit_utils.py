"""Unit tests for rate limit utility functions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from perpetualcc.core.rate_limit_utils import (
    ParsedRateLimit,
    extract_reset_time,
    extract_retry_seconds,
    is_rate_limit_message,
    parse_rate_limit_message,
)


class TestIsRateLimitMessage:
    """Tests for is_rate_limit_message function."""

    def test_detects_rate_limit(self):
        """Should detect 'rate limit' phrase."""
        assert is_rate_limit_message("Rate limit exceeded") is True

    def test_detects_429(self):
        """Should detect 429 error."""
        assert is_rate_limit_message("HTTP 429 Too Many Requests") is True

    def test_detects_hit_your_limit(self):
        """Should detect 'hit your limit' phrase."""
        assert is_rate_limit_message("You've hit your limit · resets 2pm") is True

    def test_detects_overloaded(self):
        """Should detect 'overloaded' phrase."""
        assert is_rate_limit_message("Server is currently overloaded") is True

    def test_detects_quota(self):
        """Should detect 'quota' phrase."""
        assert is_rate_limit_message("API quota exhausted") is True

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert is_rate_limit_message("RATE LIMIT") is True
        assert is_rate_limit_message("Hit Your Limit") is True

    def test_returns_false_for_non_rate_limit(self):
        """Should return False for non-rate-limit messages."""
        assert is_rate_limit_message("File not found") is False

    def test_returns_false_for_empty(self):
        """Should return False for empty string."""
        assert is_rate_limit_message("") is False


class TestExtractResetTime:
    """Tests for extract_reset_time function."""

    def test_extracts_12hr_pm(self):
        """Should extract 12-hour PM format."""
        reset_time = extract_reset_time("resets 2pm")
        assert reset_time is not None
        assert reset_time.hour == 14
        assert reset_time.minute == 0

    def test_extracts_12hr_am(self):
        """Should extract 12-hour AM format."""
        reset_time = extract_reset_time("resets 9am")
        assert reset_time is not None
        assert reset_time.hour == 9
        assert reset_time.minute == 0

    def test_extracts_with_minutes(self):
        """Should extract time with minutes."""
        reset_time = extract_reset_time("resets 2:30pm")
        assert reset_time is not None
        assert reset_time.hour == 14
        assert reset_time.minute == 30

    def test_extracts_24hr_format(self):
        """Should extract 24-hour format."""
        reset_time = extract_reset_time("resets 14:00")
        assert reset_time is not None
        assert reset_time.hour == 14
        assert reset_time.minute == 0

    def test_handles_noon(self):
        """Should handle 12pm (noon) correctly."""
        reset_time = extract_reset_time("resets 12pm")
        assert reset_time is not None
        assert reset_time.hour == 12

    def test_handles_midnight(self):
        """Should handle 12am (midnight) correctly."""
        reset_time = extract_reset_time("resets 12am")
        assert reset_time is not None
        assert reset_time.hour == 0

    def test_future_time_today(self):
        """Should return today's date if time hasn't passed."""
        # This test depends on current time, so we just verify the time is in the future
        reset_time = extract_reset_time("resets 11:59pm")
        assert reset_time is not None
        assert reset_time >= datetime.now()

    def test_returns_none_for_no_time(self):
        """Should return None if no time found."""
        assert extract_reset_time("Rate limit exceeded") is None

    def test_returns_none_for_empty(self):
        """Should return None for empty string."""
        assert extract_reset_time("") is None


class TestExtractRetrySeconds:
    """Tests for extract_retry_seconds function."""

    def test_extracts_retry_after(self):
        """Should extract retry-after format."""
        seconds = extract_retry_seconds("Retry-after: 45 seconds")
        assert seconds == 45

    def test_extracts_wait_format(self):
        """Should extract 'wait X seconds' format."""
        seconds = extract_retry_seconds("Please wait: 30 seconds")
        assert seconds == 30

    def test_extracts_try_again(self):
        """Should extract 'try again in X' format."""
        seconds = extract_retry_seconds("Try again in 120 seconds")
        assert seconds == 120

    def test_clamps_to_min(self):
        """Should clamp to minimum value."""
        seconds = extract_retry_seconds("Retry-after: 2 seconds", min_seconds=10)
        assert seconds == 10

    def test_clamps_to_max(self):
        """Should clamp to maximum value."""
        seconds = extract_retry_seconds("Retry-after: 600 seconds", max_seconds=300)
        assert seconds == 300

    def test_returns_default_when_not_found(self):
        """Should return default when no retry seconds found."""
        seconds = extract_retry_seconds("Rate limit exceeded", default=60)
        assert seconds == 60

    def test_extracts_from_time_based(self):
        """Should extract from time-based reset."""
        seconds = extract_retry_seconds("resets 2pm")
        assert seconds > 0  # Should be positive if 2pm is in the future


class TestParseRateLimitMessage:
    """Tests for parse_rate_limit_message function."""

    def test_parses_rate_limit_message(self):
        """Should parse a rate limit message."""
        parsed = parse_rate_limit_message("Rate limit exceeded. Retry-after: 45 seconds")
        assert parsed.is_rate_limit is True
        assert parsed.retry_after_seconds == 45
        assert parsed.reset_time is not None

    def test_parses_time_based_reset(self):
        """Should parse time-based reset."""
        parsed = parse_rate_limit_message("You've hit your limit · resets 2pm")
        assert parsed.is_rate_limit is True
        assert parsed.reset_time is not None
        assert parsed.reset_time.hour == 14

    def test_returns_not_rate_limit_for_other_errors(self):
        """Should return is_rate_limit=False for non-rate-limit messages."""
        parsed = parse_rate_limit_message("File not found")
        assert parsed.is_rate_limit is False

    def test_returns_not_rate_limit_for_empty(self):
        """Should return is_rate_limit=False for empty string."""
        parsed = parse_rate_limit_message("")
        assert parsed.is_rate_limit is False

    def test_preserves_original_message(self):
        """Should preserve the original message."""
        msg = "Rate limit exceeded"
        parsed = parse_rate_limit_message(msg)
        assert parsed.message == msg


class TestParsedRateLimit:
    """Tests for ParsedRateLimit dataclass."""

    def test_is_immutable(self):
        """ParsedRateLimit should be immutable."""
        parsed = ParsedRateLimit(
            is_rate_limit=True,
            retry_after_seconds=60,
            reset_time=datetime.now(),
            message="test",
        )
        with pytest.raises(AttributeError):
            parsed.retry_after_seconds = 120
