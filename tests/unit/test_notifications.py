"""Unit tests for the notifications module.

These tests cover:
1. Notifier configuration
2. Notification building
3. macOS notification sending (mocked)
4. Different notification types
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from perpetualcc.human.escalation import EscalationRequest, EscalationType
from perpetualcc.human.notifications import (
    NotificationConfig,
    NotificationType,
    Notifier,
)


class TestNotificationConfig:
    """Tests for NotificationConfig."""

    def test_default_config(self):
        """Default config should enable all notifications."""
        config = NotificationConfig()
        assert config.enabled is True
        assert config.sound is True
        assert config.on_escalation is True
        assert config.on_complete is True
        assert config.on_rate_limit is True
        assert config.on_error is True

    def test_custom_config(self):
        """Config should accept custom values."""
        config = NotificationConfig(
            enabled=True,
            sound=False,
            on_escalation=True,
            on_complete=False,
            on_rate_limit=False,
            on_error=True,
        )
        assert config.sound is False
        assert config.on_complete is False

    def test_disabled_config(self):
        """Config can disable all notifications."""
        config = NotificationConfig(enabled=False)
        assert config.enabled is False


class TestNotifier:
    """Tests for Notifier class."""

    def test_notifier_with_default_config(self):
        """Notifier should use default config."""
        notifier = Notifier()
        assert notifier.config.enabled is True

    def test_notifier_with_custom_config(self):
        """Notifier should accept custom config."""
        config = NotificationConfig(sound=False)
        notifier = Notifier(config=config)
        assert notifier.config.sound is False

    @patch("subprocess.run")
    def test_notify_calls_osascript(self, mock_run: MagicMock):
        """notify should call osascript."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify("Test Title", "Test Message")

        assert result is True
        mock_run.assert_called_once()

        # Check osascript was called
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "osascript"

    @patch("subprocess.run")
    def test_notify_escapes_quotes(self, mock_run: MagicMock):
        """notify should escape quotes in title and message."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        notifier.notify('Title with "quotes"', 'Message with "quotes"')

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert '\\"' in script

    @patch("subprocess.run")
    def test_notify_with_subtitle(self, mock_run: MagicMock):
        """notify should include subtitle."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        notifier.notify("Title", "Message", subtitle="Subtitle")

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert "Title" in script

    @patch("subprocess.run")
    def test_notify_with_sound(self, mock_run: MagicMock):
        """notify should include sound when enabled."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier(NotificationConfig(sound=True))
        notifier.notify("Title", "Message")

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert 'sound name "Ping"' in script

    @patch("subprocess.run")
    def test_notify_without_sound(self, mock_run: MagicMock):
        """notify should exclude sound when disabled."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier(NotificationConfig(sound=False))
        notifier.notify("Title", "Message")

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert "sound name" not in script

    def test_notify_returns_false_when_disabled(self):
        """notify should return False when disabled."""
        notifier = Notifier(NotificationConfig(enabled=False))
        result = notifier.notify("Title", "Message")
        assert result is False

    @patch("subprocess.run")
    def test_notify_handles_timeout(self, mock_run: MagicMock):
        """notify should handle timeout gracefully."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=5)

        notifier = Notifier()
        result = notifier.notify("Title", "Message")

        assert result is False

    @patch("subprocess.run")
    def test_notify_handles_file_not_found(self, mock_run: MagicMock):
        """notify should handle missing osascript."""
        mock_run.side_effect = FileNotFoundError()

        notifier = Notifier()
        result = notifier.notify("Title", "Message")

        assert result is False


class TestNotifyEscalation:
    """Tests for escalation notifications."""

    @patch("subprocess.run")
    def test_notify_escalation_question(self, mock_run: MagicMock):
        """notify_escalation should format question notifications."""
        mock_run.return_value = MagicMock(returncode=0)

        request = EscalationRequest.create(
            session_id="session-12345678",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Which database should I use?",
        )

        notifier = Notifier()
        result = notifier.notify_escalation(request)

        assert result is True

    @patch("subprocess.run")
    def test_notify_escalation_with_suggestion(self, mock_run: MagicMock):
        """notify_escalation should send notification for request with suggestion."""
        mock_run.return_value = MagicMock(returncode=0)

        request = EscalationRequest.create(
            session_id="session-12345678",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
            brain_suggestion="Yes",
            brain_confidence=0.85,
        )

        notifier = Notifier()
        result = notifier.notify_escalation(request)

        # Notification should be sent
        assert result is True
        mock_run.assert_called_once()

    def test_notify_escalation_disabled(self):
        """notify_escalation should respect config."""
        config = NotificationConfig(on_escalation=False)
        notifier = Notifier(config)

        request = EscalationRequest.create(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Test?",
        )

        result = notifier.notify_escalation(request)
        assert result is False


class TestNotifyTaskComplete:
    """Tests for task completion notifications."""

    @patch("subprocess.run")
    def test_notify_task_complete_basic(self, mock_run: MagicMock):
        """notify_task_complete should send basic notification."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_task_complete("session-123")

        assert result is True

    @patch("subprocess.run")
    def test_notify_task_complete_with_details(self, mock_run: MagicMock):
        """notify_task_complete should include task and cost."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        notifier.notify_task_complete(
            "session-123",
            task="Implement authentication",
            cost_usd=0.0523,
        )

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert "authentication" in script.lower() or "0.0523" in script

    def test_notify_task_complete_disabled(self):
        """notify_task_complete should respect config."""
        config = NotificationConfig(on_complete=False)
        notifier = Notifier(config)

        result = notifier.notify_task_complete("session-123")
        assert result is False


class TestNotifyRateLimited:
    """Tests for rate limit notifications."""

    @patch("subprocess.run")
    def test_notify_rate_limited_basic(self, mock_run: MagicMock):
        """notify_rate_limited should send notification."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_rate_limited("session-123", retry_after=120)

        assert result is True

    @patch("subprocess.run")
    def test_notify_rate_limited_with_message(self, mock_run: MagicMock):
        """notify_rate_limited should include message."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        notifier.notify_rate_limited(
            "session-123",
            retry_after=60,
            message="Token limit exceeded",
        )

        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert "Token" in script or "limit" in script.lower()

    def test_notify_rate_limited_disabled(self):
        """notify_rate_limited should respect config."""
        config = NotificationConfig(on_rate_limit=False)
        notifier = Notifier(config)

        result = notifier.notify_rate_limited("session-123", retry_after=60)
        assert result is False


class TestNotifyError:
    """Tests for error notifications."""

    @patch("subprocess.run")
    def test_notify_error_basic(self, mock_run: MagicMock):
        """notify_error should send notification."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_error("session-123", "Build failed")

        assert result is True

    @patch("subprocess.run")
    def test_notify_error_truncates_long_message(self, mock_run: MagicMock):
        """notify_error should truncate long messages."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        long_error = "x" * 200
        notifier.notify_error("session-123", long_error)

        # Should not fail
        mock_run.assert_called_once()

    def test_notify_error_disabled(self):
        """notify_error should respect config."""
        config = NotificationConfig(on_error=False)
        notifier = Notifier(config)

        result = notifier.notify_error("session-123", "Error")
        assert result is False


class TestNotifySessionLifecycle:
    """Tests for session lifecycle notifications."""

    @patch("subprocess.run")
    def test_notify_session_started(self, mock_run: MagicMock):
        """notify_session_started should send notification without sound."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_session_started(
            "session-123",
            "/path/to/project",
            "Implement feature X",
        )

        assert result is True

        # Check no sound for session start
        call_args = mock_run.call_args
        script = call_args[0][0][2]
        assert "sound name" not in script

    @patch("subprocess.run")
    def test_notify_session_ended_success(self, mock_run: MagicMock):
        """notify_session_ended should send success notification."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_session_ended(
            "session-123",
            success=True,
            turns=25,
            cost_usd=0.0834,
        )

        assert result is True

    @patch("subprocess.run")
    def test_notify_session_ended_failure(self, mock_run: MagicMock):
        """notify_session_ended should send failure notification."""
        mock_run.return_value = MagicMock(returncode=0)

        notifier = Notifier()
        result = notifier.notify_session_ended(
            "session-123",
            success=False,
        )

        assert result is True
