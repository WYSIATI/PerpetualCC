"""Session manager for orchestrating multiple Claude Code sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from perpetualcc.claude.adapter import ClaudeCodeAdapter
from perpetualcc.claude.types import (
    AskQuestionEvent,
    ClaudeEvent,
    InitEvent,
    RateLimitEvent,
    ResultEvent,
    SessionState,
)
from perpetualcc.core.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    SessionCheckpoint,
    ToolUseRecord,
)
from perpetualcc.core.decision_engine import DecisionEngine
from perpetualcc.core.rate_limit import RateLimitInfo, RateLimitMonitor
from perpetualcc.core.task_queue import Task, TaskQueue, TaskQueueConfig, TaskStatus

if TYPE_CHECKING:
    from perpetualcc.brain.base import Brain

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".perpetualcc" / "data"

# Type alias for event callback
EventCallback = Callable[[str, ClaudeEvent], None]


class SessionStatus(Enum):
    """Status of a managed session."""

    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_INPUT = "waiting_input"
    RATE_LIMITED = "rate_limited"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True)
class SessionManagerConfig:
    """Configuration for the session manager.

    Attributes:
        data_dir: Directory for persistent storage
        max_concurrent_sessions: Maximum concurrent sessions (0 for unlimited)
        auto_resume: Whether to auto-resume after rate limits
        persist_state: Whether to persist session state to disk
        checkpoint_interval: Events between auto-checkpoints (0 to disable)
        confidence_threshold: Minimum confidence for auto-answering questions
    """

    data_dir: Path = DEFAULT_DATA_DIR
    max_concurrent_sessions: int = 5
    auto_resume: bool = True
    persist_state: bool = True
    checkpoint_interval: int = 0
    confidence_threshold: float = 0.7

    @property
    def db_path(self) -> Path:
        """Get the full path to the database file."""
        return self.data_dir / "perpetualcc.db"


@dataclass
class ManagedSession:
    """A session managed by the session manager.

    Attributes:
        id: Unique identifier for this managed session
        claude_session_id: Claude Code's internal session ID
        project_path: Path to the project directory
        status: Current status of the session
        current_task: Current task being executed
        created_at: When the session was created
        started_at: When the session started processing
        last_activity: When the session last had activity
        checkpoint: Latest checkpoint for this session
        rate_limit_info: Current rate limit info if limited
        error_message: Error message if in error state
        token_count: Estimated tokens used
        turn_count: Number of turns executed
        total_cost_usd: Total cost in USD
        metadata: Additional session metadata
    """

    id: str
    claude_session_id: str | None
    project_path: str
    status: SessionStatus = SessionStatus.IDLE
    current_task: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    last_activity: datetime | None = None
    checkpoint: SessionCheckpoint | None = None
    rate_limit_info: RateLimitInfo | None = None
    error_message: str | None = None
    token_count: int = 0
    turn_count: int = 0
    total_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_path: str | Path,
        task: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManagedSession:
        """Create a new managed session.

        Args:
            project_path: Path to the project directory
            task: Initial task for the session
            metadata: Optional metadata

        Returns:
            A new ManagedSession instance
        """
        return cls(
            id=str(uuid.uuid4()),
            claude_session_id=None,
            project_path=str(Path(project_path).resolve()),
            current_task=task,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for persistence."""
        return {
            "id": self.id,
            "claude_session_id": self.claude_session_id,
            "project_path": self.project_path,
            "status": self.status.value,
            "current_task": self.current_task,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "checkpoint_id": (self.checkpoint.checkpoint_id if self.checkpoint else None),
            "rate_limit_info": (
                {
                    "detected_at": self.rate_limit_info.detected_at.isoformat(),
                    "limit_type": self.rate_limit_info.limit_type.value,
                    "retry_after_seconds": self.rate_limit_info.retry_after_seconds,
                    "reset_time": (
                        self.rate_limit_info.reset_time.isoformat()
                        if self.rate_limit_info.reset_time
                        else None
                    ),
                    "message": self.rate_limit_info.message,
                }
                if self.rate_limit_info
                else None
            ),
            "error_message": self.error_message,
            "token_count": self.token_count,
            "turn_count": self.turn_count,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManagedSession:
        """Create a session from a dictionary."""
        return cls(
            id=data["id"],
            claude_session_id=data.get("claude_session_id"),
            project_path=data["project_path"],
            status=SessionStatus(data.get("status", "idle")),
            current_task=data.get("current_task"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            last_activity=(
                datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else None
            ),
            checkpoint=None,  # Loaded separately
            rate_limit_info=None,  # Loaded separately
            error_message=data.get("error_message"),
            token_count=data.get("token_count", 0),
            turn_count=data.get("turn_count", 0),
            total_cost_usd=data.get("total_cost_usd"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ManagedSession:
        """Create a session from a database row."""
        return cls(
            id=row["id"],
            claude_session_id=row["claude_session_id"],
            project_path=row["project_path"],
            status=SessionStatus(row["status"]),
            current_task=row["current_task"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            last_activity=(
                datetime.fromisoformat(row["last_activity"]) if row["last_activity"] else None
            ),
            checkpoint=None,
            rate_limit_info=None,
            error_message=row["error_message"],
            token_count=row["token_count"],
            turn_count=row["turn_count"],
            total_cost_usd=row["total_cost_usd"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def with_update(self, **kwargs: Any) -> ManagedSession:
        """Create a new session with updated fields (immutability pattern).

        Args:
            **kwargs: Fields to update

        Returns:
            A new ManagedSession with the updates applied
        """
        # Create a copy of current values
        new_values = {
            "id": self.id,
            "claude_session_id": self.claude_session_id,
            "project_path": self.project_path,
            "status": self.status,
            "current_task": self.current_task,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "checkpoint": self.checkpoint,
            "rate_limit_info": self.rate_limit_info,
            "error_message": self.error_message,
            "token_count": self.token_count,
            "turn_count": self.turn_count,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata.copy(),
        }
        new_values.update(kwargs)
        return ManagedSession(**new_values)

    @property
    def is_active(self) -> bool:
        """Check if session is currently active (running or waiting)."""
        return self.status in (
            SessionStatus.PROCESSING,
            SessionStatus.WAITING_INPUT,
            SessionStatus.RATE_LIMITED,
        )

    @property
    def can_resume(self) -> bool:
        """Check if session can be resumed."""
        return self.status in (
            SessionStatus.PAUSED,
            SessionStatus.RATE_LIMITED,
            SessionStatus.IDLE,
        )

    @property
    def is_completed(self) -> bool:
        """Check if session is in a terminal state."""
        return self.status in (SessionStatus.COMPLETED, SessionStatus.ERROR)


class SessionManager:
    """Manages multiple concurrent Claude Code sessions.

    The session manager handles:
    - Creating and tracking sessions
    - Routing events to appropriate handlers
    - Managing session lifecycle (pause, resume, complete)
    - Persisting session state
    - Coordinating with decision engine and brain
    """

    # SQL for creating the sessions table
    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            claude_session_id TEXT,
            project_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'idle',
            current_task TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            last_activity TEXT,
            error_message TEXT,
            token_count INTEGER DEFAULT 0,
            turn_count INTEGER DEFAULT 0,
            total_cost_usd REAL,
            metadata TEXT
        )
    """

    _CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        CREATE INDEX IF NOT EXISTS idx_sessions_project_path ON sessions(project_path);
    """

    def __init__(
        self,
        config: SessionManagerConfig | None = None,
        brain: Brain | None = None,
        event_callback: EventCallback | None = None,
    ):
        """Initialize the session manager.

        Args:
            config: Optional configuration
            brain: Optional brain for answering questions
            event_callback: Optional callback for session events
        """
        self.config = config or SessionManagerConfig()
        self.brain = brain
        self.event_callback = event_callback

        # Runtime state
        self._sessions: dict[str, ManagedSession] = {}
        self._adapters: dict[str, ClaudeCodeAdapter] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._decision_engines: dict[str, DecisionEngine] = {}
        self._rate_monitors: dict[str, RateLimitMonitor] = {}

        # Initialize subsystems
        self._ensure_data_dir()
        self._init_db()

        # Task queue (shared across sessions)
        task_config = TaskQueueConfig(data_dir=self.config.data_dir)
        self.task_queue = TaskQueue(task_config)

        # Checkpoint manager (shared across sessions)
        checkpoint_config = CheckpointConfig(checkpoint_dir=self.config.data_dir / "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_config)

        # Load persisted sessions
        if self.config.persist_state:
            self._load_sessions()

    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.config.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute(self._CREATE_TABLE_SQL)
            conn.executescript(self._CREATE_INDEX_SQL)
            conn.commit()
        logger.debug("Session manager database initialized: %s", self.config.db_path)

    def _load_sessions(self) -> None:
        """Load persisted sessions from database."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE status NOT IN ('completed', 'error')"
            ).fetchall()

            for row in rows:
                session = ManagedSession.from_row(row)
                # Load checkpoint if available
                checkpoint = self.checkpoint_manager.load_latest(session.id)
                if checkpoint:
                    session = session.with_update(checkpoint=checkpoint)
                self._sessions[session.id] = session

        logger.info("Loaded %d persisted sessions", len(self._sessions))

    def _save_session(self, session: ManagedSession) -> None:
        """Persist a session to the database."""
        if not self.config.persist_state:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    id, claude_session_id, project_path, status, current_task,
                    created_at, started_at, last_activity, error_message,
                    token_count, turn_count, total_cost_usd, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.claude_session_id,
                    session.project_path,
                    session.status.value,
                    session.current_task,
                    session.created_at.isoformat(),
                    session.started_at.isoformat() if session.started_at else None,
                    session.last_activity.isoformat() if session.last_activity else None,
                    session.error_message,
                    session.token_count,
                    session.turn_count,
                    session.total_cost_usd,
                    json.dumps(session.metadata),
                ),
            )
            conn.commit()

    def _delete_session_from_db(self, session_id: str) -> None:
        """Delete a session from the database."""
        if not self.config.persist_state:
            return

        with self._get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    async def create_session(
        self,
        project_path: str | Path,
        task: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManagedSession:
        """Create a new managed session.

        Args:
            project_path: Path to the project directory
            task: Initial task for the session
            metadata: Optional session metadata

        Returns:
            The created ManagedSession
        """
        # Check max concurrent sessions
        active_count = sum(1 for s in self._sessions.values() if s.is_active)
        if (
            self.config.max_concurrent_sessions > 0
            and active_count >= self.config.max_concurrent_sessions
        ):
            raise RuntimeError(
                f"Maximum concurrent sessions ({self.config.max_concurrent_sessions}) reached"
            )

        # Create session
        session = ManagedSession.create(project_path, task, metadata)
        self._sessions[session.id] = session

        # Create decision engine for this session
        self._decision_engines[session.id] = DecisionEngine(
            project_path=project_path,
            brain=self.brain,
        )

        # Create rate limit monitor
        self._rate_monitors[session.id] = RateLimitMonitor()

        # Add initial task to queue if provided
        if task:
            self.task_queue.add(
                session_id=session.id,
                description=task,
            )

        # Persist
        self._save_session(session)

        logger.info(
            "Created session: id=%s, project=%s, task=%s",
            session.id,
            project_path,
            task[:50] if task else None,
        )

        return session

    def get_session(self, session_id: str) -> ManagedSession | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The ManagedSession, or None if not found
        """
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        status: SessionStatus | None = None,
        project_path: str | None = None,
    ) -> list[ManagedSession]:
        """List all managed sessions with optional filters.

        Args:
            status: Optional filter by status
            project_path: Optional filter by project path

        Returns:
            List of matching ManagedSession objects
        """
        sessions = list(self._sessions.values())

        if status:
            sessions = [s for s in sessions if s.status == status]
        if project_path:
            resolved = str(Path(project_path).resolve())
            sessions = [s for s in sessions if s.project_path == resolved]

        # Sort by last activity (most recent first)
        sessions.sort(
            key=lambda s: s.last_activity or s.created_at,
            reverse=True,
        )

        return sessions

    async def start_session(self, session_id: str) -> ManagedSession:
        """Start or resume a session.

        Args:
            session_id: The session ID to start

        Returns:
            The updated ManagedSession

        Raises:
            ValueError: If session not found
            RuntimeError: If session cannot be started
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if not session.can_resume and not session.status == SessionStatus.IDLE:
            raise RuntimeError(f"Session cannot be started: status={session.status.value}")

        # Get next task from queue
        next_task = self.task_queue.next(session_id)
        if not next_task and not session.current_task:
            raise RuntimeError("No tasks in queue for session")

        task_description = session.current_task or (next_task.description if next_task else None)

        # Update session state
        session = session.with_update(
            status=SessionStatus.PROCESSING,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            current_task=task_description,
        )
        self._sessions[session_id] = session
        self._save_session(session)

        # Mark task as started
        if next_task:
            self.task_queue.start(next_task.task_id)

        # Create adapter with resume session ID if available
        adapter = ClaudeCodeAdapter(
            project_path=session.project_path,
            resume_session_id=session.claude_session_id,
        )
        self._adapters[session_id] = adapter

        # Start processing task
        async_task = asyncio.create_task(self._run_session(session_id, task_description or ""))
        self._tasks[session_id] = async_task

        logger.info("Started session: %s", session_id)
        return session

    async def _run_session(self, session_id: str, task: str) -> None:
        """Run the main event loop for a session.

        Args:
            session_id: The session ID
            task: The task to execute
        """
        session = self._sessions.get(session_id)
        adapter = self._adapters.get(session_id)
        if not session or not adapter:
            return

        rate_monitor = self._rate_monitors.get(session_id, RateLimitMonitor())
        tool_history: list[ToolUseRecord] = []

        try:
            # Connect and start session using ClaudeSDKClient
            await adapter.connect(task)

            async for event in adapter.receive_events():
                # Update last activity
                session = self._sessions[session_id].with_update(last_activity=datetime.now())
                self._sessions[session_id] = session

                # Emit event to callback
                if self.event_callback:
                    self.event_callback(session_id, event)

                # Handle event types
                if isinstance(event, InitEvent):
                    session = session.with_update(claude_session_id=event.session_id)
                    self._sessions[session_id] = session
                    self._save_session(session)

                elif isinstance(event, RateLimitEvent):
                    info = rate_monitor.detect(event)
                    if info:
                        await self._handle_rate_limit(session_id, info)
                        return  # Exit event loop

                elif isinstance(event, ResultEvent):
                    await self._handle_result(session_id, event)
                    return  # Exit event loop

                elif isinstance(event, AskQuestionEvent):
                    session = session.with_update(status=SessionStatus.WAITING_INPUT)
                    self._sessions[session_id] = session
                    self._save_session(session)
                    # Questions will be answered by the brain or escalated

        except asyncio.CancelledError:
            logger.info("Session cancelled: %s", session_id)
            session = self._sessions[session_id].with_update(status=SessionStatus.PAUSED)
            self._sessions[session_id] = session
            self._save_session(session)

        except Exception as e:
            logger.exception("Session error: %s", session_id)
            session = self._sessions[session_id].with_update(
                status=SessionStatus.ERROR,
                error_message=str(e),
            )
            self._sessions[session_id] = session
            self._save_session(session)

        finally:
            # Keep adapter connected for potential follow-ups
            pass

    async def stream_events(self, session: ManagedSession) -> AsyncIterator[ClaudeEvent]:
        """Stream events from a session for use by MasterAgent.

        This method provides direct access to the event stream for external
        orchestration by MasterAgent. Uses ClaudeSDKClient for session
        continuity and follow-up support.

        Args:
            session: The managed session to stream events from

        Yields:
            ClaudeEvent: Events from the Claude Code session
        """
        session_id = session.id

        # Ensure we have an adapter
        if session_id not in self._adapters:
            adapter = ClaudeCodeAdapter(
                project_path=session.project_path,
                resume_session_id=session.claude_session_id,
            )
            self._adapters[session_id] = adapter
        else:
            adapter = self._adapters[session_id]
            # Update session_id if we have a newer one from ManagedSession
            if session.claude_session_id and not adapter.session_id:
                adapter.session_id = session.claude_session_id

        # Determine task
        task = session.current_task
        if not task:
            next_task = self.task_queue.next(session_id)
            if next_task:
                task = next_task.description
                self.task_queue.start(next_task.task_id)

        if not task:
            return

        # Connect and stream events using ClaudeSDKClient
        try:
            await adapter.connect(task)

            async for event in adapter.receive_events():
                # Update last activity
                updated_session = self._sessions.get(session_id)
                if updated_session:
                    updated_session = updated_session.with_update(last_activity=datetime.now())
                    self._sessions[session_id] = updated_session

                # Emit event to callback
                if self.event_callback:
                    self.event_callback(session_id, event)

                # Handle InitEvent to capture Claude session ID
                if isinstance(event, InitEvent):
                    updated_session = self._sessions.get(session_id)
                    if updated_session:
                        updated_session = updated_session.with_update(
                            claude_session_id=event.session_id
                        )
                        self._sessions[session_id] = updated_session
                        self._save_session(updated_session)

                yield event

        except asyncio.CancelledError:
            logger.info("Session stream cancelled: %s", session_id)
            raise

    async def send_response(
        self,
        session_id: str,
        response: str,
    ) -> AsyncIterator[ClaudeEvent]:
        """Send a response to a question and stream resulting events.

        This is used by MasterAgent to answer questions within a
        continuous conversation session.

        Args:
            session_id: The session to respond to
            response: The response text

        Yields:
            ClaudeEvent: Events from the continued session

        Raises:
            ValueError: If no adapter for session
        """
        adapter = self._adapters.get(session_id)
        if not adapter:
            raise ValueError(f"No adapter for session: {session_id}")

        if not adapter.connected:
            raise ValueError(f"Session not connected: {session_id}")

        async for event in adapter.send_response(response):
            # Update last activity
            updated_session = self._sessions.get(session_id)
            if updated_session:
                updated_session = updated_session.with_update(last_activity=datetime.now())
                self._sessions[session_id] = updated_session

            # Emit event to callback
            if self.event_callback:
                self.event_callback(session_id, event)

            yield event

    async def interrupt_session(self, session_id: str) -> None:
        """Interrupt a running session.

        Args:
            session_id: The session to interrupt
        """
        adapter = self._adapters.get(session_id)
        if adapter and adapter.connected:
            await adapter.interrupt()
            logger.info("Interrupted session: %s", session_id)

    async def close_session(self, session_id: str) -> None:
        """Close and disconnect a session.

        This disconnects from Claude but preserves session state for
        potential later resumption.

        Args:
            session_id: The session to close
        """
        adapter = self._adapters.pop(session_id, None)
        if adapter:
            await adapter.disconnect()
            logger.info("Closed session connection: %s", session_id)

    async def _handle_rate_limit(self, session_id: str, info: RateLimitInfo) -> None:
        """Handle a rate limit event.

        Args:
            session_id: The session ID
            info: Rate limit information
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        # Update session state
        session = session.with_update(
            status=SessionStatus.RATE_LIMITED,
            rate_limit_info=info,
        )
        self._sessions[session_id] = session

        # Create checkpoint
        checkpoint = SessionCheckpoint.create(
            session_id=session.claude_session_id or session_id,
            project_path=session.project_path,
            original_task=session.current_task or "",
            session_state=SessionState.RATE_LIMITED,
            rate_limit_info={
                "retry_after": info.retry_after_seconds,
                "message": info.message,
                "reset_time": info.reset_time.isoformat() if info.reset_time else None,
            },
        )
        self.checkpoint_manager.save(checkpoint)
        session = session.with_update(checkpoint=checkpoint)
        self._sessions[session_id] = session
        self._save_session(session)

        logger.warning(
            "Session rate limited: id=%s, retry_after=%ds",
            session_id,
            info.retry_after_seconds,
        )

        # Auto-resume if configured
        if self.config.auto_resume:
            rate_monitor = self._rate_monitors.get(session_id, RateLimitMonitor())
            await rate_monitor.wait_for_reset(info)
            await self.resume_session(session_id)

    async def _handle_result(self, session_id: str, event: ResultEvent) -> None:
        """Handle a session result event.

        Args:
            session_id: The session ID
            event: The result event
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        # Update session state
        new_status = SessionStatus.ERROR if event.is_error else SessionStatus.COMPLETED
        session = session.with_update(
            status=new_status,
            turn_count=event.num_turns,
            total_cost_usd=event.total_cost_usd,
            error_message=event.result if event.is_error else None,
        )
        self._sessions[session_id] = session

        # Mark current task as completed
        tasks = self.task_queue.list_tasks(session_id, status=TaskStatus.IN_PROGRESS)
        for task in tasks:
            if event.is_error:
                self.task_queue.complete(task.task_id, error_message=event.result)
            else:
                self.task_queue.complete(task.task_id)

        # Check for more tasks
        next_task = self.task_queue.next(session_id)
        if next_task and not event.is_error:
            # Start next task
            session = session.with_update(
                status=SessionStatus.IDLE,
                current_task=next_task.description,
            )
            self._sessions[session_id] = session
            self._save_session(session)
            await self.start_session(session_id)
        else:
            self._save_session(session)
            logger.info(
                "Session %s: status=%s, turns=%d, cost=$%.4f",
                session_id,
                new_status.value,
                event.num_turns,
                event.total_cost_usd or 0,
            )

    async def pause_session(self, session_id: str) -> ManagedSession:
        """Pause a running session.

        Args:
            session_id: The session ID to pause

        Returns:
            The updated ManagedSession

        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Cancel the running task
        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Create checkpoint
        if session.claude_session_id:
            checkpoint = SessionCheckpoint.create(
                session_id=session.claude_session_id,
                project_path=session.project_path,
                original_task=session.current_task or "",
                session_state=SessionState.PAUSED,
            )
            self.checkpoint_manager.save(checkpoint)
            session = session.with_update(checkpoint=checkpoint)

        # Update state
        session = session.with_update(status=SessionStatus.PAUSED)
        self._sessions[session_id] = session
        self._save_session(session)

        logger.info("Paused session: %s", session_id)
        return session

    async def resume_session(self, session_id: str) -> ManagedSession:
        """Resume a paused or rate-limited session.

        Args:
            session_id: The session ID to resume

        Returns:
            The updated ManagedSession

        Raises:
            ValueError: If session not found
            RuntimeError: If session cannot be resumed
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if not session.can_resume:
            raise RuntimeError(f"Session cannot be resumed: status={session.status.value}")

        # Clear rate limit info
        session = session.with_update(rate_limit_info=None)
        self._sessions[session_id] = session

        return await self.start_session(session_id)

    async def add_task(
        self,
        session_id: str,
        task: str,
        priority: int = 5,
    ) -> Task:
        """Add a task to a session's queue.

        Args:
            session_id: The session ID
            task: Task description
            priority: Task priority (higher = sooner)

        Returns:
            The created Task

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        return self.task_queue.add(
            session_id=session_id,
            description=task,
            priority=priority,
        )

    async def stop_session(self, session_id: str) -> ManagedSession:
        """Stop and clean up a session.

        Args:
            session_id: The session ID to stop

        Returns:
            The updated ManagedSession

        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Cancel running task
        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Update state
        session = session.with_update(status=SessionStatus.COMPLETED)
        self._sessions[session_id] = session
        self._save_session(session)

        # Cleanup resources
        self._adapters.pop(session_id, None)
        self._tasks.pop(session_id, None)

        logger.info("Stopped session: %s", session_id)
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Stop if running
        if session.is_active:
            await self.stop_session(session_id)

        # Remove from memory
        self._sessions.pop(session_id, None)
        self._adapters.pop(session_id, None)
        self._tasks.pop(session_id, None)
        self._decision_engines.pop(session_id, None)
        self._rate_monitors.pop(session_id, None)

        # Delete from database
        self._delete_session_from_db(session_id)

        # Delete checkpoints
        self.checkpoint_manager.delete_session(session_id)

        # Delete tasks
        self.task_queue.clear_session(session_id)

        logger.info("Deleted session: %s", session_id)
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get overall session manager statistics.

        Returns:
            Dictionary with statistics
        """
        sessions = list(self._sessions.values())

        by_status: dict[str, int] = {}
        for status in SessionStatus:
            by_status[status.value] = sum(1 for s in sessions if s.status == status)

        total_cost = sum(s.total_cost_usd or 0 for s in sessions)
        total_turns = sum(s.turn_count for s in sessions)

        return {
            "total_sessions": len(sessions),
            "active_sessions": sum(1 for s in sessions if s.is_active),
            "by_status": by_status,
            "total_cost_usd": total_cost,
            "total_turns": total_turns,
            "task_queue": self.task_queue.get_statistics(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the session manager.

        Pauses all active sessions and persists state.
        """
        logger.info("Shutting down session manager...")

        # Pause all active sessions
        active_sessions = [s for s in self._sessions.values() if s.is_active]
        for session in active_sessions:
            try:
                await self.pause_session(session.id)
            except Exception as e:
                logger.error("Error pausing session %s: %s", session.id, e)

        logger.info("Session manager shutdown complete")
