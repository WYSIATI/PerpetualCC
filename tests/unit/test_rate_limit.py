"""Unit tests for rate limit detection and monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from perpetualcc.claude.types import RateLimitEvent, ResultEvent, TextEvent
from perpetualcc.core.rate_limit import (
    ProgressCallback,
    RateLimitConfig,
    RateLimitDetector,
    RateLimitInfo,
    RateLimitMonitor,
    RateLimitType,
)


class TestRateLimitType:
    """Tests for RateLimitType enum."""

    def test_all_types_exist(self):
        """Verify all rate limit types are defined."""
        assert RateLimitType.TOKEN_LIMIT.value == "token_limit"
        assert RateLimitType.REQUEST_LIMIT.value == "request_limit"
        assert RateLimitType.USAGE_LIMIT.value == "usage_limit"
        assert RateLimitType.OVERLOADED.value == "overloaded"
        assert RateLimitType.UNKNOWN.value == "unknown"


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_info_is_immutable(self):
        """RateLimitInfo should be immutable."""
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=datetime.now() + timedelta(seconds=60),
            message="Rate limit exceeded",
        )
        with pytest.raises(AttributeError):
            info.retry_after_seconds = 120

    def test_is_expired_when_reset_time_passed(self):
        """is_expired should return True when reset_time has passed."""
        past_reset = datetime.now() - timedelta(seconds=10)
        info = RateLimitInfo(
            detected_at=datetime.now() - timedelta(seconds=70),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=past_reset,
            message="Rate limit",
        )
        assert info.is_expired is True

    def test_is_expired_when_reset_time_not_passed(self):
        """is_expired should return False when reset_time hasn't passed."""
        future_reset = datetime.now() + timedelta(seconds=60)
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=future_reset,
            message="Rate limit",
        )
        assert info.is_expired is False

    def test_is_expired_when_no_reset_time(self):
        """is_expired should return False when reset_time is None."""
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=None,
            message="Rate limit",
        )
        assert info.is_expired is False

    def test_remaining_seconds_with_reset_time(self):
        """remaining_seconds should calculate correctly."""
        future_reset = datetime.now() + timedelta(seconds=30)
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=future_reset,
            message="Rate limit",
        )
        # Should be approximately 30 seconds (may vary by 1-2 due to timing)
        assert 28 <= info.remaining_seconds <= 31

    def test_remaining_seconds_zero_when_expired(self):
        """remaining_seconds should be 0 when expired."""
        past_reset = datetime.now() - timedelta(seconds=10)
        info = RateLimitInfo(
            detected_at=datetime.now() - timedelta(seconds=70),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=past_reset,
            message="Rate limit",
        )
        assert info.remaining_seconds == 0

    def test_remaining_seconds_fallback_without_reset_time(self):
        """remaining_seconds should fall back to retry_after_seconds without reset_time."""
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.REQUEST_LIMIT,
            retry_after_seconds=60,
            reset_time=None,
            message="Rate limit",
        )
        assert info.remaining_seconds == 60


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = RateLimitConfig()
        assert config.default_retry_seconds == 60
        assert config.max_retry_seconds == 300
        assert config.min_retry_seconds == 10
        assert config.backoff_multiplier == 1.5

    def test_config_is_immutable(self):
        """Config should be immutable."""
        config = RateLimitConfig()
        with pytest.raises(AttributeError):
            config.default_retry_seconds = 120

    def test_custom_config(self):
        """Config should accept custom values."""
        config = RateLimitConfig(
            default_retry_seconds=30,
            max_retry_seconds=600,
            min_retry_seconds=5,
            backoff_multiplier=2.0,
        )
        assert config.default_retry_seconds == 30
        assert config.max_retry_seconds == 600
        assert config.min_retry_seconds == 5
        assert config.backoff_multiplier == 2.0


class TestRateLimitDetector:
    """Tests for RateLimitDetector class."""

    @pytest.fixture
    def detector(self) -> RateLimitDetector:
        """Create a detector with default config."""
        return RateLimitDetector()

    @pytest.fixture
    def custom_detector(self) -> RateLimitDetector:
        """Create a detector with custom config."""
        config = RateLimitConfig(default_retry_seconds=30, max_retry_seconds=120)
        return RateLimitDetector(config)

    def test_detect_rate_limit_event(self, detector: RateLimitDetector):
        """Detector should identify RateLimitEvent."""
        event = RateLimitEvent(
            retry_after=60,
            message="Rate limit exceeded, please wait",
        )
        info = detector.detect(event)
        assert info is not None
        assert info.retry_after_seconds == 60
        assert info.message == "Rate limit exceeded, please wait"

    def test_detect_rate_limit_event_with_reset_time(self, detector: RateLimitDetector):
        """Detector should use reset_time from event if provided."""
        reset_time = datetime.now() + timedelta(seconds=90)
        event = RateLimitEvent(
            retry_after=90,
            reset_time=reset_time,
            message="Rate limit exceeded",
        )
        info = detector.detect(event)
        assert info is not None
        assert info.reset_time == reset_time

    def test_detect_result_event_with_rate_limit(self, detector: RateLimitDetector):
        """Detector should identify rate limit from ResultEvent error."""
        event = ResultEvent(
            is_error=True,
            result="Error: Rate limit exceeded (429)",
        )
        info = detector.detect(event)
        assert info is not None
        assert info.limit_type == RateLimitType.REQUEST_LIMIT

    def test_detect_result_event_non_rate_limit(self, detector: RateLimitDetector):
        """Detector should return None for non-rate-limit errors."""
        event = ResultEvent(
            is_error=True,
            result="Error: Connection timeout",
        )
        info = detector.detect(event)
        assert info is None

    def test_detect_non_error_result_event(self, detector: RateLimitDetector):
        """Detector should return None for successful results."""
        event = ResultEvent(
            is_error=False,
            result="Success",
        )
        info = detector.detect(event)
        assert info is None

    def test_detect_non_relevant_event(self, detector: RateLimitDetector):
        """Detector should return None for non-relevant events."""
        event = TextEvent(text="Hello world")
        info = detector.detect(event)
        assert info is None

    def test_detect_from_message_rate_limit(self, detector: RateLimitDetector):
        """detect_from_message should identify rate limit messages."""
        info = detector.detect_from_message("Error: Rate limit exceeded")
        assert info is not None
        assert info.limit_type == RateLimitType.REQUEST_LIMIT

    def test_detect_from_message_429(self, detector: RateLimitDetector):
        """detect_from_message should identify 429 errors."""
        info = detector.detect_from_message("HTTP 429 Too Many Requests")
        assert info is not None
        assert info.limit_type == RateLimitType.REQUEST_LIMIT

    def test_detect_from_message_token_limit(self, detector: RateLimitDetector):
        """detect_from_message should identify token limit errors."""
        info = detector.detect_from_message("Token limit reached for this context")
        assert info is not None
        assert info.limit_type == RateLimitType.TOKEN_LIMIT

    def test_detect_from_message_context_window(self, detector: RateLimitDetector):
        """detect_from_message should identify context window errors."""
        info = detector.detect_from_message("Context window exceeded")
        assert info is not None
        assert info.limit_type == RateLimitType.TOKEN_LIMIT

    def test_detect_from_message_usage_limit(self, detector: RateLimitDetector):
        """detect_from_message should identify usage limit errors."""
        info = detector.detect_from_message("Usage limit exceeded for this billing period")
        assert info is not None
        assert info.limit_type == RateLimitType.USAGE_LIMIT

    def test_detect_from_message_quota(self, detector: RateLimitDetector):
        """detect_from_message should identify quota errors."""
        info = detector.detect_from_message("API quota exhausted")
        assert info is not None
        assert info.limit_type == RateLimitType.USAGE_LIMIT

    def test_detect_from_message_overloaded(self, detector: RateLimitDetector):
        """detect_from_message should identify overloaded errors."""
        info = detector.detect_from_message("Server is currently overloaded")
        assert info is not None
        assert info.limit_type == RateLimitType.OVERLOADED

    def test_detect_from_message_capacity(self, detector: RateLimitDetector):
        """detect_from_message should identify capacity errors."""
        info = detector.detect_from_message("At capacity, please try again later")
        assert info is not None
        assert info.limit_type == RateLimitType.OVERLOADED

    def test_detect_from_message_non_rate_limit(self, detector: RateLimitDetector):
        """detect_from_message should return None for non-rate-limit messages."""
        info = detector.detect_from_message("File not found")
        assert info is None

    def test_detect_from_message_empty(self, detector: RateLimitDetector):
        """detect_from_message should handle empty messages."""
        info = detector.detect_from_message("")
        assert info is None

    def test_extract_retry_after_from_message(self, detector: RateLimitDetector):
        """Detector should extract retry-after seconds from messages."""
        info = detector.detect_from_message("Rate limit exceeded. Retry-after: 45 seconds")
        assert info is not None
        assert info.retry_after_seconds == 45

    def test_extract_retry_seconds_wait_format(self, detector: RateLimitDetector):
        """Detector should handle 'wait X seconds' format."""
        info = detector.detect_from_message("Rate limit. Please wait: 30 seconds")
        assert info is not None
        assert info.retry_after_seconds == 30

    def test_extract_retry_seconds_try_again_format(self, detector: RateLimitDetector):
        """Detector should handle 'try again in X' format."""
        info = detector.detect_from_message("Rate limit reached. Try again in 120 seconds")
        assert info is not None
        assert info.retry_after_seconds == 120

    def test_retry_seconds_clamped_to_max(self, custom_detector: RateLimitDetector):
        """Retry seconds should be clamped to max config value."""
        info = custom_detector.detect_from_message(
            "Rate limit exceeded. Retry-after: 600 seconds"
        )
        assert info is not None
        assert info.retry_after_seconds == 120  # max_retry_seconds

    def test_retry_seconds_clamped_to_min(self, detector: RateLimitDetector):
        """Retry seconds should be clamped to min config value."""
        info = detector.detect_from_message("Rate limit exceeded. Retry-after: 2 seconds")
        assert info is not None
        assert info.retry_after_seconds == 10  # min_retry_seconds

    def test_default_retry_when_not_extractable(self, detector: RateLimitDetector):
        """Detector should use default when retry seconds not extractable."""
        info = detector.detect_from_message("Rate limit exceeded")
        assert info is not None
        assert info.retry_after_seconds == 60  # default_retry_seconds

    def test_rate_limit_event_with_zero_retry(self, detector: RateLimitDetector):
        """Detector should use default for zero retry_after in event."""
        event = RateLimitEvent(
            retry_after=0,
            message="Rate limit exceeded",
        )
        info = detector.detect(event)
        assert info is not None
        assert info.retry_after_seconds == 60  # default_retry_seconds

    def test_detect_hit_your_limit_message(self, detector: RateLimitDetector):
        """Detector should identify 'hit your limit' messages."""
        info = detector.detect_from_message("You've hit your limit · resets 2pm (Asia/Shanghai)")
        assert info is not None
        # Should detect as UNKNOWN type since it doesn't match specific patterns
        assert info.limit_type == RateLimitType.UNKNOWN

    def test_detect_hit_your_limit_result_event(self, detector: RateLimitDetector):
        """Detector should identify 'hit your limit' in ResultEvent."""
        event = ResultEvent(
            is_error=True,
            result="You've hit your limit · resets 2pm (Asia/Shanghai)",
        )
        info = detector.detect(event)
        assert info is not None

    def test_extract_reset_time_12hr_pm(self, detector: RateLimitDetector):
        """Detector should extract time-based reset (12hr PM format)."""
        info = detector.detect_from_message("You've hit your limit · resets 2pm")
        assert info is not None
        assert info.reset_time is not None
        # The reset time should be 2pm today or tomorrow
        assert info.reset_time.hour == 14
        assert info.reset_time.minute == 0

    def test_extract_reset_time_12hr_am(self, detector: RateLimitDetector):
        """Detector should extract time-based reset (12hr AM format)."""
        info = detector.detect_from_message("You've hit your limit · resets 9am")
        assert info is not None
        assert info.reset_time is not None
        assert info.reset_time.hour == 9
        assert info.reset_time.minute == 0

    def test_extract_reset_time_with_minutes(self, detector: RateLimitDetector):
        """Detector should extract time with minutes (e.g., 2:30pm)."""
        info = detector.detect_from_message("You've hit your limit · resets 2:30pm")
        assert info is not None
        assert info.reset_time is not None
        assert info.reset_time.hour == 14
        assert info.reset_time.minute == 30

    def test_extract_reset_time_24hr(self, detector: RateLimitDetector):
        """Detector should extract 24-hour format reset time."""
        info = detector.detect_from_message("You've hit your limit · resets 14:00")
        assert info is not None
        assert info.reset_time is not None
        assert info.reset_time.hour == 14
        assert info.reset_time.minute == 0

    def test_reset_time_calculates_correct_retry_seconds(self, detector: RateLimitDetector):
        """Reset time should calculate appropriate retry seconds."""
        # This test verifies that a time-based reset calculates reasonable retry_seconds
        info = detector.detect_from_message("You've hit your limit · resets 2pm")
        assert info is not None
        # retry_after_seconds should be positive if reset_time is in the future
        assert info.retry_after_seconds > 0

    def test_12hr_noon(self, detector: RateLimitDetector):
        """Detector should handle 12pm (noon) correctly."""
        info = detector.detect_from_message("You've hit your limit · resets 12pm")
        assert info is not None
        assert info.reset_time is not None
        assert info.reset_time.hour == 12

    def test_12hr_midnight(self, detector: RateLimitDetector):
        """Detector should handle 12am (midnight) correctly."""
        info = detector.detect_from_message("You've hit your limit · resets 12am")
        assert info is not None
        assert info.reset_time is not None
        assert info.reset_time.hour == 0


class TestRateLimitMonitor:
    """Tests for RateLimitMonitor class."""

    @pytest.fixture
    def monitor(self) -> RateLimitMonitor:
        """Create a monitor with default config."""
        return RateLimitMonitor()

    @pytest.fixture
    def custom_monitor(self) -> RateLimitMonitor:
        """Create a monitor with custom config."""
        config = RateLimitConfig(default_retry_seconds=30, backoff_multiplier=2.0)
        return RateLimitMonitor(config)

    def test_initial_state(self, monitor: RateLimitMonitor):
        """Monitor should start with no active limit."""
        assert monitor.current_limit is None
        assert monitor.is_rate_limited is False
        assert monitor.consecutive_limits == 0

    def test_detect_tracks_rate_limit(self, monitor: RateLimitMonitor):
        """Monitor should track detected rate limits."""
        event = RateLimitEvent(
            retry_after=60,
            message="Rate limit exceeded",
        )
        info = monitor.detect(event)
        assert info is not None
        assert monitor.current_limit is not None
        assert monitor.is_rate_limited is True
        assert monitor.consecutive_limits == 1

    def test_detect_from_message_tracks_rate_limit(self, monitor: RateLimitMonitor):
        """Monitor should track rate limits from messages."""
        info = monitor.detect_from_message("Rate limit exceeded")
        assert info is not None
        assert monitor.is_rate_limited is True
        assert monitor.consecutive_limits == 1

    def test_consecutive_limits_increment(self, monitor: RateLimitMonitor):
        """Consecutive limits should increment."""
        for i in range(3):
            monitor.detect_from_message("Rate limit exceeded")
        assert monitor.consecutive_limits == 3

    def test_clear_resets_state(self, monitor: RateLimitMonitor):
        """clear() should reset monitor state."""
        monitor.detect_from_message("Rate limit exceeded")
        monitor.detect_from_message("Rate limit exceeded")
        assert monitor.consecutive_limits == 2

        monitor.clear()
        assert monitor.current_limit is None
        assert monitor.is_rate_limited is False
        assert monitor.consecutive_limits == 0

    def test_current_limit_clears_when_expired(self, monitor: RateLimitMonitor):
        """current_limit should return None when limit is expired."""
        event = RateLimitEvent(
            retry_after=1,
            reset_time=datetime.now() - timedelta(seconds=10),
            message="Rate limit",
        )
        monitor.detect(event)
        # Force check by accessing current_limit
        assert monitor.current_limit is None
        assert monitor.is_rate_limited is False

    def test_get_adjusted_retry_seconds_no_backoff(self, monitor: RateLimitMonitor):
        """First limit should not have backoff."""
        event = RateLimitEvent(retry_after=60, message="Rate limit")
        monitor.detect(event)
        assert monitor.get_adjusted_retry_seconds() == 60

    def test_get_adjusted_retry_seconds_with_backoff(self, custom_monitor: RateLimitMonitor):
        """Consecutive limits should have exponential backoff."""
        # First limit
        custom_monitor.detect_from_message("Rate limit")
        assert custom_monitor.get_adjusted_retry_seconds() == 30

        # Second limit - should apply backoff
        custom_monitor.detect_from_message("Rate limit")
        # 30 * 2.0^1 = 60
        assert custom_monitor.get_adjusted_retry_seconds() == 60

        # Third limit
        custom_monitor.detect_from_message("Rate limit")
        # 30 * 2.0^2 = 120
        assert custom_monitor.get_adjusted_retry_seconds() == 120

    def test_backoff_capped_at_max(self, monitor: RateLimitMonitor):
        """Backoff should be capped at max_retry_seconds."""
        # Hit many rate limits
        for _ in range(10):
            monitor.detect_from_message("Rate limit")
        # Should be capped at 300 (max)
        assert monitor.get_adjusted_retry_seconds() <= 300

    def test_get_adjusted_retry_no_limit(self, monitor: RateLimitMonitor):
        """get_adjusted_retry_seconds should return default when no limit."""
        assert monitor.get_adjusted_retry_seconds() == 60

    @pytest.mark.asyncio
    async def test_wait_for_reset(self, monitor: RateLimitMonitor):
        """wait_for_reset should wait for the specified time."""
        event = RateLimitEvent(retry_after=15, message="Rate limit")
        info = monitor.detect(event)

        # Mock sleep to avoid actual waiting
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await monitor.wait_for_reset(info)
            # Should have called sleep for each second (15 seconds)
            assert mock_sleep.call_count == 15

    @pytest.mark.asyncio
    async def test_wait_for_reset_with_callback(self, monitor: RateLimitMonitor):
        """wait_for_reset should call progress callback."""
        event = RateLimitEvent(retry_after=12, message="Rate limit")
        info = monitor.detect(event)

        progress_calls: list[tuple[int, int]] = []

        def progress_callback(remaining: int, total: int) -> None:
            progress_calls.append((remaining, total))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await monitor.wait_for_reset(info, progress_callback=progress_callback)

        # Should have called callback for 12, 11, 10... 1, 0 (13 calls)
        assert len(progress_calls) == 13
        assert progress_calls[0] == (12, 12)
        assert progress_calls[-1] == (0, 12)

    @pytest.mark.asyncio
    async def test_wait_for_reset_no_limit(self, monitor: RateLimitMonitor):
        """wait_for_reset should do nothing when no limit."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await monitor.wait_for_reset()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_reset_uses_current_limit(self, monitor: RateLimitMonitor):
        """wait_for_reset should use current limit if not specified."""
        event = RateLimitEvent(retry_after=20, message="Rate limit")
        monitor.detect(event)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await monitor.wait_for_reset()
            assert mock_sleep.call_count == 20

    def test_get_history(self, monitor: RateLimitMonitor):
        """get_history should return recorded limits."""
        monitor.detect_from_message("Rate limit 1")
        monitor.detect_from_message("Rate limit 2")
        monitor.detect_from_message("Rate limit 3")

        history = monitor.get_history()
        assert len(history) == 3
        assert history[0].message == "Rate limit 1"
        assert history[2].message == "Rate limit 3"

    def test_get_history_with_limit(self, monitor: RateLimitMonitor):
        """get_history should respect limit parameter."""
        for i in range(5):
            monitor.detect_from_message(f"Rate limit {i}")

        history = monitor.get_history(limit=2)
        assert len(history) == 2
        assert history[0].message == "Rate limit 3"
        assert history[1].message == "Rate limit 4"

    def test_get_statistics_empty(self, monitor: RateLimitMonitor):
        """get_statistics should handle empty history."""
        stats = monitor.get_statistics()
        assert stats["total_limits"] == 0
        assert stats["by_type"] == {}
        assert stats["current_limit"] is None
        assert stats["consecutive_limits"] == 0

    def test_get_statistics_with_data(self, monitor: RateLimitMonitor):
        """get_statistics should return correct counts."""
        monitor.detect_from_message("Rate limit exceeded")
        monitor.detect_from_message("Token limit reached")
        monitor.detect_from_message("Rate limit again")

        stats = monitor.get_statistics()
        assert stats["total_limits"] == 3
        assert stats["by_type"]["request_limit"] == 2
        assert stats["by_type"]["token_limit"] == 1
        assert stats["consecutive_limits"] == 3


class TestRateLimitIntegration:
    """Integration tests for rate limit components."""

    def test_detector_and_monitor_work_together(self):
        """Detector and monitor should integrate correctly."""
        config = RateLimitConfig(default_retry_seconds=45)
        detector = RateLimitDetector(config)
        monitor = RateLimitMonitor(config, detector)

        # Simulate rate limit event
        event = RateLimitEvent(
            retry_after=30,
            message="Rate limit hit",
        )

        # Monitor uses its detector
        info = monitor.detect(event)
        assert info is not None
        assert info.retry_after_seconds == 30
        assert monitor.is_rate_limited is True

    def test_end_to_end_rate_limit_handling(self):
        """Test complete rate limit detection and handling flow."""
        monitor = RateLimitMonitor()

        # First rate limit from ResultEvent
        result_event = ResultEvent(
            is_error=True,
            result="Error: 429 Too Many Requests. Retry-after: 30 seconds",
        )
        info = monitor.detect(result_event)
        assert info is not None
        assert info.limit_type == RateLimitType.REQUEST_LIMIT
        assert info.retry_after_seconds == 30

        # Simulate successful operation
        monitor.clear()
        assert monitor.is_rate_limited is False
        assert monitor.consecutive_limits == 0

        # Second rate limit from RateLimitEvent
        limit_event = RateLimitEvent(
            retry_after=60,
            message="Usage limit exceeded",
        )
        info = monitor.detect(limit_event)
        assert info is not None
        assert info.limit_type == RateLimitType.USAGE_LIMIT
        assert info.retry_after_seconds == 60

    def test_non_rate_limit_events_dont_affect_monitor(self):
        """Non-rate-limit events should not change monitor state."""
        monitor = RateLimitMonitor()

        # Various non-rate-limit events
        events = [
            TextEvent(text="Hello"),
            ResultEvent(is_error=False, result="Success"),
            ResultEvent(is_error=True, result="Connection timeout"),
        ]

        for event in events:
            monitor.detect(event)

        assert monitor.is_rate_limited is False
        assert monitor.consecutive_limits == 0
        assert len(monitor.get_history()) == 0
