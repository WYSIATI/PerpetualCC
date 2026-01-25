"""macOS notification system for human alerts.

This module provides system notifications using macOS osascript.
No external dependencies are required - uses built-in macOS capabilities.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perpetualcc.human.escalation import EscalationRequest

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Type of notification."""

    ESCALATION = "escalation"
    TASK_COMPLETE = "task_complete"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


@dataclass
class NotificationConfig:
    """Configuration for notifications.

    Attributes:
        enabled: Whether notifications are enabled
        sound: Whether to play a sound
        on_escalation: Notify on escalations
        on_complete: Notify on task completion
        on_rate_limit: Notify on rate limits
        on_error: Notify on errors
    """

    enabled: bool = True
    sound: bool = True
    on_escalation: bool = True
    on_complete: bool = True
    on_rate_limit: bool = True
    on_error: bool = True


class Notifier:
    """macOS notification sender.

    Uses osascript to send notifications without any external dependencies.

    Usage:
        notifier = Notifier()
        notifier.notify("Title", "Message")
        notifier.notify_escalation(escalation_request)
    """

    def __init__(self, config: NotificationConfig | None = None):
        """Initialize the notifier."""
        self.config = config or NotificationConfig()

    def notify(
        self,
        title: str,
        message: str,
        subtitle: str | None = None,
        sound: bool | None = None,
    ) -> bool:
        """Send a macOS notification.

        Args:
            title: Notification title
            message: Notification body text
            subtitle: Optional subtitle
            sound: Whether to play sound (uses config default if None)

        Returns:
            True if notification was sent successfully
        """
        if not self.config.enabled:
            return False

        use_sound = sound if sound is not None else self.config.sound

        try:
            script = self._build_script(title, message, subtitle, use_sound)
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
                check=False,
            )
            logger.debug("Notification sent: %s", title)
            return True

        except subprocess.TimeoutExpired:
            logger.warning("Notification timed out: %s", title)
            return False

        except FileNotFoundError:
            logger.warning("osascript not found - notifications unavailable")
            return False

        except Exception as e:
            logger.warning("Failed to send notification: %s", e)
            return False

    def _build_script(
        self,
        title: str,
        message: str,
        subtitle: str | None,
        sound: bool,
    ) -> str:
        """Build the osascript AppleScript command."""
        escaped_title = title.replace('"', '\\"')
        escaped_message = message.replace('"', '\\"')

        parts = [f'display notification "{escaped_message}"']
        parts.append('with title "PerpetualCC"')
        parts.append(f'subtitle "{escaped_title}"')

        if sound:
            parts.append('sound name "Ping"')

        return " ".join(parts)

    def notify_escalation(self, request: EscalationRequest) -> bool:
        """Send notification for an escalation request.

        Args:
            request: The escalation request

        Returns:
            True if notification was sent
        """
        if not self.config.on_escalation:
            return False

        session_short = request.session_id[:8]
        type_label = request.escalation_type.value.title()

        title = f"{type_label} - Session {session_short}"
        message = request.question[:100]
        if len(request.question) > 100:
            message += "..."

        if request.brain_suggestion:
            subtitle = (
                f"Suggestion: {request.brain_suggestion} "
                f"({int(request.brain_confidence * 100)}%)"
            )
        else:
            subtitle = "Response needed"

        return self.notify(title, message, subtitle)

    def notify_task_complete(
        self,
        session_id: str,
        task: str | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """Send notification for task completion.

        Args:
            session_id: The session ID
            task: The completed task description
            cost_usd: Cost of the task

        Returns:
            True if notification was sent
        """
        if not self.config.on_complete:
            return False

        title = f"Task Complete - {session_id[:8]}"
        message = task[:80] if task else "Task completed successfully"
        if task and len(task) > 80:
            message += "..."

        subtitle = f"Cost: ${cost_usd:.4f}" if cost_usd else None

        return self.notify(title, message, subtitle)

    def notify_rate_limited(
        self,
        session_id: str,
        retry_after: int,
        message: str | None = None,
    ) -> bool:
        """Send notification for rate limit.

        Args:
            session_id: The session ID
            retry_after: Seconds until rate limit resets
            message: Optional rate limit message

        Returns:
            True if notification was sent
        """
        if not self.config.on_rate_limit:
            return False

        title = f"Rate Limited - {session_id[:8]}"
        body = message or f"Waiting {retry_after}s before resuming"
        subtitle = f"Resumes in {retry_after}s"

        return self.notify(title, body, subtitle)

    def notify_error(
        self,
        session_id: str,
        error: str,
    ) -> bool:
        """Send notification for an error.

        Args:
            session_id: The session ID
            error: The error message

        Returns:
            True if notification was sent
        """
        if not self.config.on_error:
            return False

        title = f"Error - {session_id[:8]}"
        message = error[:100]
        if len(error) > 100:
            message += "..."

        return self.notify(title, message, "Review needed", sound=True)

    def notify_session_started(
        self,
        session_id: str,
        project_path: str,
        task: str | None = None,
    ) -> bool:
        """Send notification for session start.

        Args:
            session_id: The session ID
            project_path: Path to the project
            task: The initial task

        Returns:
            True if notification was sent
        """
        title = f"Session Started - {session_id[:8]}"
        message = task[:80] if task else "Session started"
        if task and len(task) > 80:
            message += "..."

        # Import Path here to get the project name
        from pathlib import Path

        project_name = Path(project_path).name
        subtitle = f"Project: {project_name}"

        return self.notify(title, message, subtitle, sound=False)

    def notify_session_ended(
        self,
        session_id: str,
        success: bool = True,
        turns: int | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """Send notification for session end.

        Args:
            session_id: The session ID
            success: Whether session ended successfully
            turns: Number of turns
            cost_usd: Total cost

        Returns:
            True if notification was sent
        """
        status = "Completed" if success else "Failed"
        title = f"Session {status} - {session_id[:8]}"

        parts = []
        if turns:
            parts.append(f"{turns} turns")
        if cost_usd:
            parts.append(f"${cost_usd:.4f}")
        message = " | ".join(parts) if parts else status

        return self.notify(title, message, sound=success)
