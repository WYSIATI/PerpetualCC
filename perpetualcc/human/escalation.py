"""Escalation queue for human intervention.

This module provides the infrastructure for:
- Creating and tracking escalation requests
- Persisting escalations to SQLite database
- Waiting for and processing human responses

Note: Pending escalations never expire - they wait indefinitely until
a human responds or the escalation is explicitly cancelled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from perpetualcc.core.session_manager import ManagedSession

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".perpetualcc" / "data"


class EscalationType(Enum):
    """Type of escalation request."""

    PERMISSION = "permission"
    QUESTION = "question"
    ERROR = "error"
    REVIEW = "review"
    CONFIRMATION = "confirmation"


class EscalationStatus(Enum):
    """Status of an escalation request."""

    PENDING = "pending"
    RESPONDED = "responded"
    CANCELLED = "cancelled"


@dataclass
class EscalationRequest:
    """A request for human intervention.

    Attributes:
        id: Unique identifier for this request
        session_id: Session that generated this request
        escalation_type: Type of escalation
        context: What is happening that requires intervention
        question: The specific question or request
        options: Available choices (if any)
        brain_suggestion: What the brain would answer (if available)
        brain_confidence: Confidence in brain's suggestion (0.0-1.0)
        metadata: Additional context data
        created_at: When this request was created
        status: Current status
        response: Human's response (when responded)
        responded_at: When the response was received
    """

    id: str
    session_id: str
    escalation_type: EscalationType
    context: str
    question: str
    options: list[str] = field(default_factory=list)
    brain_suggestion: str | None = None
    brain_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: EscalationStatus = EscalationStatus.PENDING
    response: str | None = None
    responded_at: datetime | None = None

    @classmethod
    def create(
        cls,
        session_id: str,
        escalation_type: EscalationType,
        context: str,
        question: str,
        options: list[str] | None = None,
        brain_suggestion: str | None = None,
        brain_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> EscalationRequest:
        """Create a new escalation request."""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            escalation_type=escalation_type,
            context=context,
            question=question,
            options=options or [],
            brain_suggestion=brain_suggestion,
            brain_confidence=brain_confidence,
            metadata=metadata or {},
            created_at=datetime.now(),
            status=EscalationStatus.PENDING,
        )

    @property
    def is_pending(self) -> bool:
        """Check if this request is still pending."""
        return self.status == EscalationStatus.PENDING

    @property
    def age_seconds(self) -> int:
        """Get the age of this request in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())

    def with_response(self, response: str) -> EscalationRequest:
        """Create a copy with the response set."""
        return EscalationRequest(
            id=self.id,
            session_id=self.session_id,
            escalation_type=self.escalation_type,
            context=self.context,
            question=self.question,
            options=self.options,
            brain_suggestion=self.brain_suggestion,
            brain_confidence=self.brain_confidence,
            metadata=self.metadata,
            created_at=self.created_at,
            status=EscalationStatus.RESPONDED,
            response=response,
            responded_at=datetime.now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "escalation_type": self.escalation_type.value,
            "context": self.context,
            "question": self.question,
            "options": self.options,
            "brain_suggestion": self.brain_suggestion,
            "brain_confidence": self.brain_confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "response": self.response,
            "responded_at": (
                self.responded_at.isoformat() if self.responded_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationRequest:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            escalation_type=EscalationType(data["escalation_type"]),
            context=data["context"],
            question=data["question"],
            options=data.get("options", []),
            brain_suggestion=data.get("brain_suggestion"),
            brain_confidence=data.get("brain_confidence", 0.0),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=EscalationStatus(data["status"]),
            response=data.get("response"),
            responded_at=(
                datetime.fromisoformat(data["responded_at"])
                if data.get("responded_at")
                else None
            ),
        )


ESCALATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    escalation_type TEXT NOT NULL,
    context TEXT NOT NULL,
    question TEXT NOT NULL,
    options TEXT DEFAULT '[]',
    brain_suggestion TEXT,
    brain_confidence REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    response TEXT,
    responded_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_escalations_session_id ON escalations(session_id);
CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
CREATE INDEX IF NOT EXISTS idx_escalations_created_at ON escalations(created_at);
"""


class EscalationQueue:
    """Queue for managing escalation requests.

    This class provides:
    - SQLite-backed persistence for escalations
    - Methods to create, query, and respond to escalations
    - Async waiting for responses (waits indefinitely until response)
    - Notification callbacks when escalations are created
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        notification_callback: Callable[[EscalationRequest], None] | None = None,
    ):
        """Initialize the escalation queue."""
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.db_path = self.data_dir / "perpetualcc.db"
        self.notification_callback = notification_callback

        self._connection = None
        self._initialized = False
        self._response_events: dict[str, asyncio.Event] = {}
        self._responses: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the database connection and schema."""
        if self._initialized:
            return

        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite is required. Install with: pip install aiosqlite"
            ) from e

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._connection.executescript(ESCALATION_SCHEMA_SQL)
        await self._connection.commit()

        self._initialized = True
        logger.info("Escalation queue initialized: %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False

    async def escalate(
        self,
        session_id: str,
        escalation_type: EscalationType,
        context: str,
        question: str,
        options: list[str] | None = None,
        brain_suggestion: str | None = None,
        brain_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> EscalationRequest:
        """Create a new escalation request."""
        if not self._initialized:
            await self.initialize()

        request = EscalationRequest.create(
            session_id=session_id,
            escalation_type=escalation_type,
            context=context,
            question=question,
            options=options,
            brain_suggestion=brain_suggestion,
            brain_confidence=brain_confidence,
            metadata=metadata,
        )

        await self._insert_request(request)

        self._response_events[request.id] = asyncio.Event()

        if self.notification_callback:
            try:
                self.notification_callback(request)
            except Exception as e:
                logger.warning("Notification callback failed: %s", e)

        logger.info(
            "Escalation created: %s (%s) for session %s",
            request.id[:8],
            escalation_type.value,
            session_id[:8],
        )

        return request

    async def _insert_request(self, request: EscalationRequest) -> None:
        """Insert a request into the database."""
        data = request.to_dict()
        await self._connection.execute(
            """
            INSERT INTO escalations (
                id, session_id, escalation_type, context, question, options,
                brain_suggestion, brain_confidence, metadata, created_at,
                status, response, responded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["session_id"],
                data["escalation_type"],
                data["context"],
                data["question"],
                json.dumps(data["options"]),
                data["brain_suggestion"],
                data["brain_confidence"],
                json.dumps(data["metadata"]),
                data["created_at"],
                data["status"],
                data["response"],
                data["responded_at"],
            ),
        )
        await self._connection.commit()

    async def respond(
        self, request_id: str, response: str
    ) -> EscalationRequest | None:
        """Respond to an escalation request."""
        if not self._initialized:
            await self.initialize()

        request = await self.get_request(request_id)
        if not request:
            logger.warning("Escalation not found: %s", request_id)
            return None

        if not request.is_pending:
            logger.warning(
                "Escalation %s is not pending (status=%s)",
                request_id[:8],
                request.status.value,
            )
            return None

        now = datetime.now().isoformat()
        await self._connection.execute(
            """
            UPDATE escalations
            SET status = ?, response = ?, responded_at = ?
            WHERE id = ?
            """,
            (EscalationStatus.RESPONDED.value, response, now, request_id),
        )
        await self._connection.commit()

        self._responses[request_id] = response
        if request_id in self._response_events:
            self._response_events[request_id].set()

        logger.info("Escalation responded: %s -> %s", request_id[:8], response[:50])

        return request.with_response(response)

    async def wait_for_response(
        self,
        request_id: str,
        poll_interval: float = 0.5,
    ) -> str | None:
        """Wait indefinitely for a response to an escalation.

        This method blocks until a response is received. There is no timeout -
        pending escalations wait indefinitely until responded to or cancelled.
        """
        if not self._initialized:
            await self.initialize()

        request = await self.get_request(request_id)
        if not request:
            return None

        if not request.is_pending:
            return request.response

        if request_id not in self._response_events:
            self._response_events[request_id] = asyncio.Event()

        event = self._response_events[request_id]

        try:
            await event.wait()

            if request_id in self._responses:
                return self._responses.pop(request_id)

            updated = await self.get_request(request_id)
            return updated.response if updated else None

        finally:
            if request_id in self._response_events:
                del self._response_events[request_id]

    async def get_request(self, request_id: str) -> EscalationRequest | None:
        """Get an escalation request by ID."""
        if not self._initialized:
            await self.initialize()

        async with self._connection.execute(
            "SELECT * FROM escalations WHERE id = ?",
            (request_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_request(row)

    async def get_pending(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[EscalationRequest]:
        """Get pending escalation requests."""
        if not self._initialized:
            await self.initialize()

        if session_id:
            async with self._connection.execute(
                """
                SELECT * FROM escalations
                WHERE status = ? AND session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (EscalationStatus.PENDING.value, session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._connection.execute(
                """
                SELECT * FROM escalations
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (EscalationStatus.PENDING.value, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_request(row) for row in rows]

    async def get_history(
        self,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EscalationRequest]:
        """Get escalation history (including responded/cancelled)."""
        if not self._initialized:
            await self.initialize()

        if session_id:
            async with self._connection.execute(
                """
                SELECT * FROM escalations
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._connection.execute(
                """
                SELECT * FROM escalations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_request(row) for row in rows]

    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending escalation."""
        if not self._initialized:
            await self.initialize()

        async with self._connection.execute(
            """
            UPDATE escalations
            SET status = ?
            WHERE id = ? AND status = ?
            """,
            (
                EscalationStatus.CANCELLED.value,
                request_id,
                EscalationStatus.PENDING.value,
            ),
        ) as cursor:
            await self._connection.commit()
            changed = cursor.rowcount > 0

        if changed:
            if request_id in self._response_events:
                self._response_events[request_id].set()
            logger.info("Escalation cancelled: %s", request_id[:8])

        return changed

    async def clear_session(self, session_id: str) -> int:
        """Cancel all pending escalations for a session."""
        if not self._initialized:
            await self.initialize()

        async with self._connection.execute(
            """
            UPDATE escalations
            SET status = ?
            WHERE session_id = ? AND status = ?
            """,
            (
                EscalationStatus.CANCELLED.value,
                session_id,
                EscalationStatus.PENDING.value,
            ),
        ) as cursor:
            await self._connection.commit()
            return cursor.rowcount

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about escalations."""
        if not self._initialized:
            await self.initialize()

        stats: dict[str, Any] = {}

        async with self._connection.execute(
            "SELECT COUNT(*) FROM escalations"
        ) as cursor:
            stats["total_escalations"] = (await cursor.fetchone())[0]

        async with self._connection.execute(
            "SELECT status, COUNT(*) FROM escalations GROUP BY status"
        ) as cursor:
            stats["by_status"] = {
                row[0]: row[1] for row in await cursor.fetchall()
            }

        async with self._connection.execute(
            "SELECT escalation_type, COUNT(*) FROM escalations "
            "GROUP BY escalation_type"
        ) as cursor:
            stats["by_type"] = {
                row[0]: row[1] for row in await cursor.fetchall()
            }

        stats["pending_count"] = stats.get("by_status", {}).get("pending", 0)

        async with self._connection.execute(
            """
            SELECT AVG(
                CAST(
                    (julianday(responded_at) - julianday(created_at))
                    * 24 * 60 * 60 AS INTEGER
                )
            )
            FROM escalations
            WHERE status = 'responded' AND responded_at IS NOT NULL
            """
        ) as cursor:
            avg_time = (await cursor.fetchone())[0]
            stats["avg_response_time_seconds"] = (
                round(avg_time, 1) if avg_time else None
            )

        return stats

    def _row_to_request(self, row) -> EscalationRequest:
        """Convert a database row to an EscalationRequest."""
        return EscalationRequest(
            id=row["id"],
            session_id=row["session_id"],
            escalation_type=EscalationType(row["escalation_type"]),
            context=row["context"],
            question=row["question"],
            options=json.loads(row["options"]) if row["options"] else [],
            brain_suggestion=row["brain_suggestion"],
            brain_confidence=row["brain_confidence"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            status=EscalationStatus(row["status"]),
            response=row["response"],
            responded_at=(
                datetime.fromisoformat(row["responded_at"])
                if row["responded_at"]
                else None
            ),
        )


class HumanBridge:
    """High-level interface for human intervention.

    This class coordinates between:
    - EscalationQueue for persistence and response handling
    - Notification system for alerting humans
    - CLI prompts for interactive response

    Note: By default, all permissions are granted and all tools are allowed.
    Escalation only happens for truly exceptional cases that need human review.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        enable_notifications: bool = True,
    ):
        """Initialize the human bridge."""
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.enable_notifications = enable_notifications

        self._queue: EscalationQueue | None = None
        self._notifier = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the human bridge components."""
        if self._initialized:
            return

        self._queue = EscalationQueue(
            data_dir=self.data_dir,
            notification_callback=(
                self._on_escalation if self.enable_notifications else None
            ),
        )
        await self._queue.initialize()

        if self.enable_notifications:
            from perpetualcc.human.notifications import Notifier

            self._notifier = Notifier()

        self._initialized = True
        logger.info("Human bridge initialized")

    async def close(self) -> None:
        """Close human bridge resources."""
        if self._queue:
            await self._queue.close()
            self._queue = None
        self._initialized = False

    def _on_escalation(self, request: EscalationRequest) -> None:
        """Handle new escalation (send notification)."""
        if self._notifier:
            self._notifier.notify_escalation(request)

    async def escalate_question(
        self,
        session: ManagedSession,
        question: str,
        options: list[str] | None = None,
        suggestion: str | None = None,
        confidence: float = 0.0,
        context: str | None = None,
        wait_for_response: bool = True,
    ) -> str | None:
        """Escalate a question to the human.

        Waits indefinitely until a response is received.
        """
        if not self._initialized:
            await self.initialize()

        request = await self._queue.escalate(
            session_id=session.id,
            escalation_type=EscalationType.QUESTION,
            context=context or f"Working on: {session.current_task or 'task'}",
            question=question,
            options=options,
            brain_suggestion=suggestion,
            brain_confidence=confidence,
            metadata={
                "project_path": session.project_path,
                "current_task": session.current_task,
            },
        )

        if wait_for_response:
            return await self._queue.wait_for_response(request.id)

        return None

    async def escalate_permission(
        self,
        session: ManagedSession,
        tool_name: str,
        tool_input: dict[str, Any],
        risk_level: str,
        suggestion: str | None = None,
        confidence: float = 0.0,
        wait_for_response: bool = True,
    ) -> str | None:
        """Escalate a permission request to the human.

        Note: By default, permissions are granted. This is only called
        for exceptional cases that truly need human review.

        Waits indefinitely until a response is received.
        """
        if not self._initialized:
            await self.initialize()

        if tool_name == "Bash":
            detail = tool_input.get("command", "?")
        elif tool_name in ("Read", "Write", "Edit"):
            detail = tool_input.get("file_path", "?")
        else:
            detail = json.dumps(tool_input)[:100]

        question = f"Allow {tool_name}: {detail}?"
        context = f"Risk level: {risk_level}"

        request = await self._queue.escalate(
            session_id=session.id,
            escalation_type=EscalationType.PERMISSION,
            context=context,
            question=question,
            options=["Approve", "Deny"],
            brain_suggestion=suggestion,
            brain_confidence=confidence,
            metadata={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "risk_level": risk_level,
                "project_path": session.project_path,
            },
        )

        if wait_for_response:
            response = await self._queue.wait_for_response(request.id)
            if response:
                return (
                    "approve"
                    if response.lower() in ("approve", "yes", "allow", "1")
                    else "deny"
                )
            return None

        return None

    async def escalate_error(
        self,
        session: ManagedSession,
        error_message: str,
        options: list[str] | None = None,
        wait_for_response: bool = True,
    ) -> str | None:
        """Escalate an error for human decision.

        Waits indefinitely until a response is received.
        """
        if not self._initialized:
            await self.initialize()

        request = await self._queue.escalate(
            session_id=session.id,
            escalation_type=EscalationType.ERROR,
            context=f"Error in session {session.id[:8]}",
            question=(
                f"Error occurred: {error_message}\n\nHow should I proceed?"
            ),
            options=options or ["Retry", "Skip", "Abort"],
            metadata={
                "error_message": error_message,
                "project_path": session.project_path,
                "current_task": session.current_task,
            },
        )

        if wait_for_response:
            return await self._queue.wait_for_response(request.id)

        return None

    async def get_pending(
        self, session_id: str | None = None
    ) -> list[EscalationRequest]:
        """Get pending escalations."""
        if not self._initialized:
            await self.initialize()

        return await self._queue.get_pending(session_id=session_id)

    async def respond(
        self, request_id: str, response: str
    ) -> EscalationRequest | None:
        """Respond to an escalation."""
        if not self._initialized:
            await self.initialize()

        return await self._queue.respond(request_id, response)

    async def get_statistics(self) -> dict[str, Any]:
        """Get escalation statistics."""
        if not self._initialized:
            await self.initialize()

        return await self._queue.get_statistics()
