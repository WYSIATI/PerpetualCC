"""Analytics module for PerpetualCC Web UI."""

import logging
import re
import threading
from datetime import datetime
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

# Session ID validation pattern: alphanumeric, hyphens, underscores only
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
SESSION_ID_MAX_LENGTH = 128

# Valid event types for validation
VALID_EVENT_TYPES = frozenset(
    {
        "action_executed",
        "action_failed",
        "approval_requested",
        "approval_granted",
        "approval_denied",
        "error",
        "warning",
        "info",
        "iteration_complete",
        "plan_created",
        "plan_updated",
        "tool_called",
        "user_input",
        "custom",
    }
)


class AnalyticsError(Exception):
    """Exception raised for analytics-related errors."""

    pass


class SessionAnalytics:
    """Track and analyze session metrics with thread safety."""

    def __init__(self):
        self._metrics: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info("SessionAnalytics initialized")

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID format.

        Args:
            session_id: The session ID to validate.

        Raises:
            AnalyticsError: If session_id is invalid.
        """
        if not session_id:
            raise AnalyticsError("Session ID is required")

        if len(session_id) > SESSION_ID_MAX_LENGTH:
            raise AnalyticsError(f"Session ID must be {SESSION_ID_MAX_LENGTH} characters or less")

        if not SESSION_ID_PATTERN.match(session_id):
            raise AnalyticsError(
                "Session ID must contain only alphanumeric characters, hyphens, and underscores"
            )

    def _validate_event_type(self, event_type: str) -> str:
        """Validate and normalize event type.

        Args:
            event_type: The event type to validate.

        Returns:
            Normalized event type string.
        """
        if not event_type:
            return "custom"

        # Normalize: lowercase, replace spaces with underscores
        normalized = event_type.lower().strip().replace(" ", "_").replace("-", "_")

        # If not in valid types, use 'custom' but log the original
        if normalized not in VALID_EVENT_TYPES:
            logger.debug(f"Unknown event type '{event_type}', using 'custom'")
            return "custom"

        return normalized

    def record_session_start(self, session_id: str, config: dict) -> None:
        """Record session start.

        Args:
            session_id: The session identifier.
            config: Session configuration dictionary.
        """
        try:
            self._validate_session_id(session_id)
        except AnalyticsError as e:
            logger.warning(f"Invalid session_id for record_session_start: {e}")
            return

        with self._lock:
            if session_id in self._metrics:
                logger.warning(f"Session {session_id} already exists in analytics, overwriting")

            self._metrics[session_id] = {
                "start_time": datetime.now().isoformat(),
                "config": config.copy() if config else {},
                "events": [],
                "status": "running",
            }
            logger.info(f"Recorded session start: {session_id}")

    def record_event(self, session_id: str, event_type: str, data: dict | None = None) -> bool:
        """Record session event.

        Args:
            session_id: The session identifier.
            event_type: Type of event (see VALID_EVENT_TYPES).
            data: Additional event data.

        Returns:
            True if event was recorded, False otherwise.
        """
        try:
            self._validate_session_id(session_id)
        except AnalyticsError as e:
            logger.warning(f"Invalid session_id for record_event: {e}")
            return False

        normalized_event_type = self._validate_event_type(event_type)

        with self._lock:
            if session_id not in self._metrics:
                logger.warning(f"Cannot record event for unknown session: {session_id}")
                return False

            event = {
                "timestamp": datetime.now().isoformat(),
                "type": normalized_event_type,
                "data": data.copy() if data else {},
            }

            self._metrics[session_id]["events"].append(event)
            logger.debug(f"Recorded event '{normalized_event_type}' for session: {session_id}")
            return True

    def record_session_end(self, session_id: str, status: str = "completed") -> bool:
        """Record session end.

        Args:
            session_id: The session identifier.
            status: Final session status.

        Returns:
            True if session end was recorded, False otherwise.
        """
        try:
            self._validate_session_id(session_id)
        except AnalyticsError as e:
            logger.warning(f"Invalid session_id for record_session_end: {e}")
            return False

        # Validate status
        valid_statuses = {"completed", "stopped", "failed", "timeout", "error"}
        if status not in valid_statuses:
            logger.warning(f"Invalid status '{status}', using 'completed'")
            status = "completed"

        with self._lock:
            if session_id not in self._metrics:
                logger.warning(f"Cannot record end for unknown session: {session_id}")
                return False

            try:
                start_time_str = self._metrics[session_id]["start_time"]
                start = datetime.fromisoformat(start_time_str)
                duration = (datetime.now() - start).total_seconds()
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to calculate duration for session {session_id}: {e}")
                duration = 0

            self._metrics[session_id].update(
                {
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "status": status,
                }
            )
            logger.info(f"Recorded session end: {session_id} (status={status}, duration={duration:.2f}s)")
            return True

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Dictionary of session statistics, empty if session not found.
        """
        try:
            self._validate_session_id(session_id)
        except AnalyticsError as e:
            logger.warning(f"Invalid session_id for get_session_stats: {e}")
            return {}

        with self._lock:
            if session_id not in self._metrics:
                logger.debug(f"No metrics found for session: {session_id}")
                return {}

            # Create a copy to avoid returning reference to internal data
            metric = self._metrics[session_id].copy()
            events = list(metric.get("events", []))

            return {
                "session_id": session_id,
                "duration_seconds": metric.get("duration_seconds", 0),
                "status": metric.get("status", "unknown"),
                "event_count": len(events),
                "event_breakdown": self._count_events_by_type(events),
                "start_time": metric.get("start_time"),
                "end_time": metric.get("end_time"),
            }

    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall statistics.

        Returns:
            Dictionary of overall analytics statistics.
        """
        with self._lock:
            if not self._metrics:
                return {
                    "total_sessions": 0,
                    "completed_sessions": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "success_rate": 0,
                    "sessions_by_status": {},
                }

            # Count sessions by status
            status_counts: dict[str, int] = {}
            durations: list[float] = []

            for metric in self._metrics.values():
                status = metric.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

                if "duration_seconds" in metric:
                    durations.append(metric["duration_seconds"])

            completed_count = status_counts.get("completed", 0)
            total_sessions = len(self._metrics)

            return {
                "total_sessions": total_sessions,
                "completed_sessions": completed_count,
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "success_rate": (completed_count / total_sessions * 100) if total_sessions else 0,
                "sessions_by_status": status_counts,
            }

    def clear_session(self, session_id: str) -> bool:
        """Clear analytics data for a specific session.

        Args:
            session_id: The session identifier.

        Returns:
            True if session was cleared, False if not found.
        """
        try:
            self._validate_session_id(session_id)
        except AnalyticsError as e:
            logger.warning(f"Invalid session_id for clear_session: {e}")
            return False

        with self._lock:
            if session_id in self._metrics:
                del self._metrics[session_id]
                logger.info(f"Cleared analytics for session: {session_id}")
                return True
            return False

    def clear_all(self) -> int:
        """Clear all analytics data.

        Returns:
            Number of sessions cleared.
        """
        with self._lock:
            count = len(self._metrics)
            self._metrics.clear()
            logger.info(f"Cleared all analytics data ({count} sessions)")
            return count

    def _count_events_by_type(self, events: list) -> dict[str, int]:
        """Count events by type.

        Args:
            events: List of event dictionaries.

        Returns:
            Dictionary mapping event types to counts.
        """
        counts: dict[str, int] = {}
        for event in events:
            if isinstance(event, dict):
                event_type = event.get("type", "unknown")
                counts[event_type] = counts.get(event_type, 0) + 1
        return counts


# Global analytics instance
analytics = SessionAnalytics()
