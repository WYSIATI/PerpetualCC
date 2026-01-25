"""Session checkpoint and state persistence for Claude Code sessions."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from perpetualcc.claude.types import SessionState

logger = logging.getLogger(__name__)

# Default checkpoint storage location
DEFAULT_CHECKPOINT_DIR = Path.home() / ".perpetualcc" / "data" / "checkpoints"


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpoint management.

    Attributes:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum number of checkpoints to keep per session
        auto_checkpoint_interval: Interval in events between auto-checkpoints (0 to disable)
        compress: Whether to compress checkpoint files
    """

    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    max_checkpoints: int = 10
    auto_checkpoint_interval: int = 0  # Disabled by default
    compress: bool = False


@dataclass
class ToolUseRecord:
    """Record of a tool use during the session.

    Attributes:
        tool_use_id: Unique identifier for the tool use
        tool_name: Name of the tool used
        tool_input: Input parameters for the tool
        timestamp: When the tool was used
        result: Result of the tool execution (if available)
        is_error: Whether the tool execution resulted in an error
    """

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    timestamp: datetime
    result: str | None = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolUseRecord:
        """Create from dictionary."""
        return cls(
            tool_use_id=data["tool_use_id"],
            tool_name=data["tool_name"],
            tool_input=data["tool_input"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            result=data.get("result"),
            is_error=data.get("is_error", False),
        )


@dataclass
class SessionCheckpoint:
    """Complete state snapshot of a Claude Code session.

    This captures all necessary information to resume a session after
    interruption (rate limit, crash, manual pause, etc.).

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        session_id: Claude session ID for resumption
        created_at: When this checkpoint was created
        project_path: Path to the project directory
        original_task: The original task prompt
        session_state: Current state of the session
        last_event_type: Type of the last event processed
        last_event_timestamp: Timestamp of the last event
        tool_history: List of tool uses during the session
        pending_questions: Questions waiting for answers
        rate_limit_info: Rate limit information if applicable
        decision_count: Number of permission decisions made
        error_message: Error message if session is in error state
        metadata: Additional metadata for the session
    """

    checkpoint_id: str
    session_id: str
    created_at: datetime
    project_path: str
    original_task: str
    session_state: SessionState = SessionState.IDLE
    last_event_type: str | None = None
    last_event_timestamp: datetime | None = None
    tool_history: list[ToolUseRecord] = field(default_factory=list)
    pending_questions: list[dict[str, Any]] = field(default_factory=list)
    rate_limit_info: dict[str, Any] | None = None
    decision_count: int = 0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        project_path: str | Path,
        original_task: str,
        **kwargs: Any,
    ) -> SessionCheckpoint:
        """Create a new checkpoint with auto-generated ID and timestamp.

        Args:
            session_id: Claude session ID
            project_path: Path to the project directory
            original_task: The original task prompt
            **kwargs: Additional checkpoint attributes

        Returns:
            A new SessionCheckpoint instance
        """
        return cls(
            checkpoint_id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=datetime.now(),
            project_path=str(Path(project_path).resolve()),
            original_task=original_task,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "project_path": self.project_path,
            "original_task": self.original_task,
            "session_state": self.session_state.value,
            "last_event_type": self.last_event_type,
            "last_event_timestamp": (
                self.last_event_timestamp.isoformat() if self.last_event_timestamp else None
            ),
            "tool_history": [t.to_dict() for t in self.tool_history],
            "pending_questions": self.pending_questions,
            "rate_limit_info": self.rate_limit_info,
            "decision_count": self.decision_count,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionCheckpoint:
        """Create a checkpoint from a dictionary.

        Args:
            data: Dictionary containing checkpoint data

        Returns:
            A SessionCheckpoint instance
        """
        tool_history = [ToolUseRecord.from_dict(t) for t in data.get("tool_history", [])]
        last_event_ts = data.get("last_event_timestamp")

        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            project_path=data["project_path"],
            original_task=data["original_task"],
            session_state=SessionState(data.get("session_state", "idle")),
            last_event_type=data.get("last_event_type"),
            last_event_timestamp=datetime.fromisoformat(last_event_ts) if last_event_ts else None,
            tool_history=tool_history,
            pending_questions=data.get("pending_questions", []),
            rate_limit_info=data.get("rate_limit_info"),
            decision_count=data.get("decision_count", 0),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    def with_update(self, **kwargs: Any) -> SessionCheckpoint:
        """Create a new checkpoint with updated fields.

        This follows the immutability pattern by creating a new instance
        rather than mutating the existing one.

        Args:
            **kwargs: Fields to update

        Returns:
            A new SessionCheckpoint with the updates applied
        """
        data = self.to_dict()
        for key, value in kwargs.items():
            if key == "session_state" and isinstance(value, SessionState):
                data[key] = value.value
            elif key == "last_event_timestamp" and isinstance(value, datetime):
                data[key] = value.isoformat()
            elif key == "tool_history":
                data[key] = [t.to_dict() for t in value]
            else:
                data[key] = value
        return SessionCheckpoint.from_dict(data)

    @property
    def can_resume(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.session_state not in (SessionState.COMPLETED, SessionState.ERROR)

    def generate_resume_prompt(self) -> str:
        """Generate a prompt for resuming the session.

        Returns:
            A prompt string to continue the session
        """
        if self.session_state == SessionState.RATE_LIMITED:
            return f"Continue the task after rate limit wait. Original task: {self.original_task}"
        elif self.session_state == SessionState.PAUSED:
            return f"Continue the paused task: {self.original_task}"
        elif self.pending_questions:
            return f"Continue after answering questions. Original task: {self.original_task}"
        else:
            return f"Continue the task: {self.original_task}"


class CheckpointManager:
    """Manages saving and restoring session checkpoints.

    This class handles persistence of session state to disk, allowing
    sessions to be resumed after interruption.
    """

    def __init__(self, config: CheckpointConfig | None = None):
        """Initialize the checkpoint manager.

        Args:
            config: Optional configuration for checkpoint behavior
        """
        self.config = config or CheckpointConfig()
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self) -> None:
        """Ensure the checkpoint directory exists."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a session's checkpoints."""
        return self.config.checkpoint_dir / session_id

    def _get_checkpoint_path(self, session_id: str, checkpoint_id: str) -> Path:
        """Get the path for a specific checkpoint file."""
        return self._get_session_dir(session_id) / f"{checkpoint_id}.json"

    def save(self, checkpoint: SessionCheckpoint) -> Path:
        """Save a checkpoint to disk.

        Args:
            checkpoint: The checkpoint to save

        Returns:
            Path to the saved checkpoint file
        """
        session_dir = self._get_session_dir(checkpoint.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self._get_checkpoint_path(checkpoint.session_id, checkpoint.checkpoint_id)

        # Convert to JSON
        data = checkpoint.to_dict()
        json_content = json.dumps(data, indent=2, ensure_ascii=False)

        # Write atomically using a temp file
        temp_path = checkpoint_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json_content, encoding="utf-8")
            temp_path.replace(checkpoint_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        logger.info(
            "Saved checkpoint: session=%s, checkpoint=%s",
            checkpoint.session_id,
            checkpoint.checkpoint_id,
        )

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(checkpoint.session_id)

        return checkpoint_path

    def load(self, session_id: str, checkpoint_id: str) -> SessionCheckpoint:
        """Load a specific checkpoint from disk.

        Args:
            session_id: The session ID
            checkpoint_id: The checkpoint ID to load

        Returns:
            The loaded SessionCheckpoint

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist
        """
        checkpoint_path = self._get_checkpoint_path(session_id, checkpoint_id)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: session={session_id}, checkpoint={checkpoint_id}"
            )

        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return SessionCheckpoint.from_dict(data)

    def load_latest(self, session_id: str) -> SessionCheckpoint | None:
        """Load the most recent checkpoint for a session.

        Args:
            session_id: The session ID

        Returns:
            The latest SessionCheckpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints(session_id)
        if not checkpoints:
            return None

        # Checkpoints are sorted by creation time, newest last
        latest = checkpoints[-1]
        return self.load(session_id, latest["checkpoint_id"])

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a session.

        Args:
            session_id: The session ID

        Returns:
            List of checkpoint metadata, sorted by creation time
        """
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            return []

        checkpoints = []
        for path in session_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoints.append(
                    {
                        "checkpoint_id": data["checkpoint_id"],
                        "created_at": data["created_at"],
                        "session_state": data.get("session_state", "unknown"),
                        "path": str(path),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to read checkpoint %s: %s", path, e)
                continue

        # Sort by creation time
        checkpoints.sort(key=lambda x: x["created_at"])
        return checkpoints

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with checkpoints.

        Returns:
            List of session metadata with latest checkpoint info
        """
        sessions = []
        if not self.config.checkpoint_dir.exists():
            return sessions

        for session_dir in self.config.checkpoint_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            checkpoints = self.list_checkpoints(session_id)

            if checkpoints:
                latest = checkpoints[-1]
                sessions.append(
                    {
                        "session_id": session_id,
                        "checkpoint_count": len(checkpoints),
                        "latest_checkpoint": latest["checkpoint_id"],
                        "latest_created_at": latest["created_at"],
                        "latest_state": latest["session_state"],
                    }
                )

        # Sort by latest checkpoint time
        sessions.sort(key=lambda x: x["latest_created_at"], reverse=True)
        return sessions

    def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            session_id: The session ID
            checkpoint_id: The checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(session_id, checkpoint_id)

        if not checkpoint_path.exists():
            return False

        checkpoint_path.unlink()
        logger.info("Deleted checkpoint: session=%s, checkpoint=%s", session_id, checkpoint_id)
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete all checkpoints for a session.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if session not found
        """
        session_dir = self._get_session_dir(session_id)

        if not session_dir.exists():
            return False

        shutil.rmtree(session_dir)
        logger.info("Deleted all checkpoints for session: %s", session_id)
        return True

    def _cleanup_old_checkpoints(self, session_id: str) -> None:
        """Remove old checkpoints beyond the max limit.

        Args:
            session_id: The session ID to clean up
        """
        checkpoints = self.list_checkpoints(session_id)

        if len(checkpoints) <= self.config.max_checkpoints:
            return

        # Delete oldest checkpoints beyond the limit
        to_delete = checkpoints[: -self.config.max_checkpoints]
        for cp in to_delete:
            self.delete_checkpoint(session_id, cp["checkpoint_id"])
            logger.debug(
                "Cleaned up old checkpoint: session=%s, checkpoint=%s",
                session_id,
                cp["checkpoint_id"],
            )

    def restore(self, checkpoint: SessionCheckpoint) -> str:
        """Generate the prompt needed to resume from a checkpoint.

        Args:
            checkpoint: The checkpoint to restore from

        Returns:
            Resume prompt string for the Claude session
        """
        return checkpoint.generate_resume_prompt()

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about checkpoint storage.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        total_checkpoints = 0
        session_count = 0

        if self.config.checkpoint_dir.exists():
            for session_dir in self.config.checkpoint_dir.iterdir():
                if session_dir.is_dir():
                    session_count += 1
                    for checkpoint_file in session_dir.glob("*.json"):
                        total_checkpoints += 1
                        total_size += checkpoint_file.stat().st_size

        return {
            "checkpoint_dir": str(self.config.checkpoint_dir),
            "session_count": session_count,
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
