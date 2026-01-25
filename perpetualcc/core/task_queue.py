"""Priority task queue with SQLite persistence for Claude Code sessions."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DATA_DIR = Path.home() / ".perpetualcc" / "data"


class TaskStatus(Enum):
    """Status of a task in the queue."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass(frozen=True)
class TaskQueueConfig:
    """Configuration for task queue behavior.

    Attributes:
        data_dir: Directory for SQLite database
        db_name: Name of the database file
        max_tasks_per_session: Maximum tasks per session (0 for unlimited)
        auto_cleanup_completed: Whether to auto-remove completed tasks
        completed_retention_days: Days to keep completed tasks before cleanup
    """

    data_dir: Path = DEFAULT_DATA_DIR
    db_name: str = "perpetualcc.db"
    max_tasks_per_session: int = 0  # Unlimited
    auto_cleanup_completed: bool = False
    completed_retention_days: int = 7

    @property
    def db_path(self) -> Path:
        """Get the full path to the database file."""
        return self.data_dir / self.db_name


@dataclass
class Task:
    """A task in the queue.

    Attributes:
        task_id: Unique identifier for the task
        session_id: Session this task belongs to
        description: Task description/prompt
        status: Current status of the task
        priority: Priority level (higher = sooner)
        created_at: When the task was created
        started_at: When the task started executing
        completed_at: When the task was completed
        error_message: Error message if the task failed
        metadata: Additional task metadata
        depends_on: List of task IDs this task depends on
    """

    task_id: str
    session_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = TaskPriority.NORMAL.value
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        session_id: str,
        description: str,
        priority: int | TaskPriority = TaskPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> Task:
        """Create a new task with auto-generated ID.

        Args:
            session_id: Session this task belongs to
            description: Task description
            priority: Priority level (int or TaskPriority enum)
            metadata: Optional metadata
            depends_on: Optional list of task IDs this depends on

        Returns:
            A new Task instance
        """
        pri_value = priority.value if isinstance(priority, TaskPriority) else priority
        return cls(
            task_id=str(uuid.uuid4()),
            session_id=session_id,
            description=description,
            priority=pri_value,
            metadata=metadata or {},
            depends_on=depends_on or [],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create a task from a dictionary."""
        return cls(
            task_id=data["task_id"],
            session_id=data["session_id"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            priority=data.get("priority", TaskPriority.NORMAL.value),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            depends_on=data.get("depends_on", []),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Task:
        """Create a task from a database row."""
        return cls(
            task_id=row["task_id"],
            session_id=row["session_id"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            priority=row["priority"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            depends_on=json.loads(row["depends_on"]) if row["depends_on"] else [],
        )

    def with_update(self, **kwargs: Any) -> Task:
        """Create a new task with updated fields (immutability pattern).

        Args:
            **kwargs: Fields to update

        Returns:
            A new Task with the updates applied
        """
        data = self.to_dict()
        for key, value in kwargs.items():
            if key == "status" and isinstance(value, TaskStatus):
                data[key] = value.value
            elif key in ("started_at", "completed_at", "created_at") and isinstance(
                value, datetime
            ):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return Task.from_dict(data)

    @property
    def is_pending(self) -> bool:
        """Check if task is waiting to be executed."""
        return self.status == TaskStatus.PENDING

    @property
    def is_completed(self) -> bool:
        """Check if task is completed (success or failure)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def can_execute(self) -> bool:
        """Check if task is ready to execute (pending, no blockers)."""
        return self.status == TaskStatus.PENDING


class TaskQueue:
    """Priority task queue with SQLite persistence.

    This class manages a queue of tasks for Claude Code sessions, with
    support for priority ordering, dependencies, and persistence.
    """

    # SQL for creating the tasks table
    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 5,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error_message TEXT,
            metadata TEXT,
            depends_on TEXT
        )
    """

    _CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC);
    """

    def __init__(self, config: TaskQueueConfig | None = None):
        """Initialize the task queue.

        Args:
            config: Optional configuration for queue behavior
        """
        self.config = config or TaskQueueConfig()
        self._ensure_data_dir()
        self._init_db()

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
        logger.debug("Task queue database initialized: %s", self.config.db_path)

    def add(
        self,
        session_id: str,
        description: str,
        priority: int | TaskPriority = TaskPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> Task:
        """Add a new task to the queue.

        Args:
            session_id: Session this task belongs to
            description: Task description
            priority: Priority level (higher = sooner)
            metadata: Optional metadata
            depends_on: Optional list of task IDs this depends on

        Returns:
            The created Task
        """
        task = Task.create(
            session_id=session_id,
            description=description,
            priority=priority,
            metadata=metadata,
            depends_on=depends_on,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, session_id, description, status, priority,
                    created_at, started_at, completed_at, error_message,
                    metadata, depends_on
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.session_id,
                    task.description,
                    task.status.value,
                    task.priority,
                    task.created_at.isoformat(),
                    None,
                    None,
                    None,
                    json.dumps(task.metadata),
                    json.dumps(task.depends_on),
                ),
            )
            conn.commit()

        logger.info(
            "Added task to queue: session=%s, task=%s, priority=%d",
            session_id,
            task.task_id,
            task.priority,
        )
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task ID to retrieve

        Returns:
            The Task, or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row:
                return Task.from_row(row)
        return None

    def next(self, session_id: str) -> Task | None:
        """Get the next task to execute for a session.

        Returns the highest-priority pending task whose dependencies are satisfied.

        Args:
            session_id: Session ID to get next task for

        Returns:
            The next Task to execute, or None if queue is empty
        """
        with self._get_connection() as conn:
            # Get all pending tasks for this session, ordered by priority
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE session_id = ? AND status = 'pending'
                ORDER BY priority DESC, created_at ASC
                """,
                (session_id,),
            ).fetchall()

            if not rows:
                return None

            # Get set of completed task IDs for dependency checking
            completed_ids = set(
                row["task_id"]
                for row in conn.execute(
                    """
                    SELECT task_id FROM tasks
                    WHERE session_id = ? AND status IN ('completed')
                    """,
                    (session_id,),
                ).fetchall()
            )

            # Find first task with satisfied dependencies
            for row in rows:
                task = Task.from_row(row)
                if not task.depends_on or all(dep in completed_ids for dep in task.depends_on):
                    return task

        return None

    def start(self, task_id: str) -> Task | None:
        """Mark a task as in progress.

        Args:
            task_id: The task ID to start

        Returns:
            The updated Task, or None if not found
        """
        task = self.get(task_id)
        if not task or not task.is_pending:
            return None

        now = datetime.now()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, started_at = ?
                WHERE task_id = ?
                """,
                (TaskStatus.IN_PROGRESS.value, now.isoformat(), task_id),
            )
            conn.commit()

        logger.info("Started task: %s", task_id)
        return task.with_update(status=TaskStatus.IN_PROGRESS, started_at=now)

    def complete(self, task_id: str, error_message: str | None = None) -> Task | None:
        """Mark a task as completed or failed.

        Args:
            task_id: The task ID to complete
            error_message: If provided, marks task as failed

        Returns:
            The updated Task, or None if not found
        """
        task = self.get(task_id)
        if not task:
            return None

        now = datetime.now()
        status = TaskStatus.FAILED if error_message else TaskStatus.COMPLETED

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, completed_at = ?, error_message = ?
                WHERE task_id = ?
                """,
                (status.value, now.isoformat(), error_message, task_id),
            )
            conn.commit()

        logger.info("Completed task: %s (status=%s)", task_id, status.value)
        return task.with_update(status=status, completed_at=now, error_message=error_message)

    def cancel(self, task_id: str) -> Task | None:
        """Cancel a pending task.

        Args:
            task_id: The task ID to cancel

        Returns:
            The updated Task, or None if not found or not cancellable
        """
        task = self.get(task_id)
        if not task or task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            return None

        now = datetime.now()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (TaskStatus.CANCELLED.value, now.isoformat(), task_id),
            )
            conn.commit()

        logger.info("Cancelled task: %s", task_id)
        return task.with_update(status=TaskStatus.CANCELLED, completed_at=now)

    def update_priority(self, task_id: str, priority: int | TaskPriority) -> Task | None:
        """Update a task's priority.

        Args:
            task_id: The task ID to update
            priority: New priority value

        Returns:
            The updated Task, or None if not found
        """
        task = self.get(task_id)
        if not task:
            return None

        pri_value = priority.value if isinstance(priority, TaskPriority) else priority

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET priority = ? WHERE task_id = ?",
                (pri_value, task_id),
            )
            conn.commit()

        logger.info("Updated task priority: %s -> %d", task_id, pri_value)
        return task.with_update(priority=pri_value)

    def list_tasks(
        self,
        session_id: str | None = None,
        status: TaskStatus | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """List tasks with optional filters.

        Args:
            session_id: Optional filter by session
            status: Optional filter by status
            limit: Optional limit on results

        Returns:
            List of Tasks matching the filters
        """
        conditions = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM tasks
            WHERE {where_clause}
            ORDER BY priority DESC, created_at ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Task.from_row(row) for row in rows]

    def count(self, session_id: str | None = None, status: TaskStatus | None = None) -> int:
        """Count tasks with optional filters.

        Args:
            session_id: Optional filter by session
            status: Optional filter by status

        Returns:
            Number of matching tasks
        """
        conditions = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM tasks WHERE {where_clause}",
                params,
            ).fetchone()
            return row["cnt"] if row else 0

    def clear_session(self, session_id: str, keep_completed: bool = False) -> int:
        """Remove all tasks for a session.

        Args:
            session_id: The session ID to clear
            keep_completed: Whether to keep completed tasks

        Returns:
            Number of tasks removed
        """
        with self._get_connection() as conn:
            if keep_completed:
                result = conn.execute(
                    """
                    DELETE FROM tasks
                    WHERE session_id = ? AND status NOT IN ('completed', 'failed')
                    """,
                    (session_id,),
                )
            else:
                result = conn.execute(
                    "DELETE FROM tasks WHERE session_id = ?",
                    (session_id,),
                )
            conn.commit()
            count = result.rowcount

        logger.info("Cleared %d tasks for session: %s", count, session_id)
        return count

    def delete(self, task_id: str) -> bool:
        """Delete a specific task.

        Args:
            task_id: The task ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            result = conn.execute(
                "DELETE FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
            deleted = result.rowcount > 0

        if deleted:
            logger.info("Deleted task: %s", task_id)
        return deleted

    def cleanup_completed(self, days_old: int | None = None) -> int:
        """Remove old completed tasks.

        Args:
            days_old: Remove tasks completed more than this many days ago
                     (defaults to config value)

        Returns:
            Number of tasks removed
        """
        retention = days_old or self.config.completed_retention_days
        cutoff = datetime.now()

        with self._get_connection() as conn:
            result = conn.execute(
                """
                DELETE FROM tasks
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < datetime(?, '-' || ? || ' days')
                """,
                (cutoff.isoformat(), retention),
            )
            conn.commit()
            count = result.rowcount

        logger.info("Cleaned up %d old completed tasks", count)
        return count

    def get_statistics(self, session_id: str | None = None) -> dict[str, Any]:
        """Get queue statistics.

        Args:
            session_id: Optional filter by session

        Returns:
            Dictionary with queue statistics
        """
        with self._get_connection() as conn:
            conditions = ["1=1"]
            params: list[Any] = []
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            where_clause = " AND ".join(conditions)

            # Get counts by status
            rows = conn.execute(
                f"""
                SELECT status, COUNT(*) as cnt
                FROM tasks
                WHERE {where_clause}
                GROUP BY status
                """,
                params,
            ).fetchall()

            by_status = {row["status"]: row["cnt"] for row in rows}

            # Get total
            total = sum(by_status.values())

            # Get unique sessions
            session_rows = conn.execute(
                f"SELECT DISTINCT session_id FROM tasks WHERE {where_clause}",
                params,
            ).fetchall()
            session_count = len(session_rows)

        return {
            "total": total,
            "by_status": by_status,
            "pending": by_status.get("pending", 0),
            "in_progress": by_status.get("in_progress", 0),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "cancelled": by_status.get("cancelled", 0),
            "session_count": session_count,
        }

    def reorder_session(
        self, session_id: str, task_ids: list[str], base_priority: int = 100
    ) -> list[Task]:
        """Reorder tasks for a session by setting priorities.

        Tasks are assigned priorities in descending order based on their
        position in the provided list (first task gets highest priority).

        Args:
            session_id: The session ID
            task_ids: List of task IDs in desired order (first = highest priority)
            base_priority: Starting priority for the first task

        Returns:
            List of updated Tasks
        """
        updated_tasks = []
        with self._get_connection() as conn:
            for i, task_id in enumerate(task_ids):
                priority = base_priority - i
                conn.execute(
                    """
                    UPDATE tasks
                    SET priority = ?
                    WHERE task_id = ? AND session_id = ?
                    """,
                    (priority, task_id, session_id),
                )

            conn.commit()

        # Fetch updated tasks
        for task_id in task_ids:
            task = self.get(task_id)
            if task:
                updated_tasks.append(task)

        logger.info("Reordered %d tasks for session: %s", len(task_ids), session_id)
        return updated_tasks
