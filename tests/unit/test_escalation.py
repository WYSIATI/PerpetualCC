"""Unit tests for the escalation module.

These tests cover:
1. EscalationRequest creation and state management
2. EscalationQueue persistence and retrieval
3. Response handling and async waiting
4. HumanBridge high-level interface
5. Real-world Claude Code escalation scenarios
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from perpetualcc.human.escalation import (
    EscalationQueue,
    EscalationRequest,
    EscalationStatus,
    EscalationType,
    HumanBridge,
)


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def queue(temp_dir: Path) -> EscalationQueue:
    """Create an escalation queue."""
    return EscalationQueue(data_dir=temp_dir, notification_callback=None)


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_create_basic_request(self):
        """Create a basic escalation request."""
        request = EscalationRequest.create(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="Working on authentication",
            question="Which database should I use?",
        )
        assert request.session_id == "session-123"
        assert request.escalation_type == EscalationType.QUESTION
        assert request.question == "Which database should I use?"
        assert request.status == EscalationStatus.PENDING
        assert request.is_pending

    def test_create_request_with_options(self):
        """Create request with multiple choice options."""
        request = EscalationRequest.create(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="Selecting framework",
            question="Which web framework?",
            options=["FastAPI", "Django", "Flask"],
        )
        assert request.options == ["FastAPI", "Django", "Flask"]

    def test_create_request_with_suggestion(self):
        """Create request with brain suggestion."""
        request = EscalationRequest.create(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="Confirming action",
            question="Should I proceed?",
            options=["Yes", "No"],
            brain_suggestion="Yes",
            brain_confidence=0.85,
        )
        assert request.brain_suggestion == "Yes"
        assert request.brain_confidence == 0.85

    def test_create_permission_request(self):
        """Create a permission escalation request."""
        request = EscalationRequest.create(
            session_id="session-456",
            escalation_type=EscalationType.PERMISSION,
            context="Risk level: HIGH",
            question="Allow Bash: rm -rf /tmp/test?",
            options=["Approve", "Deny"],
            metadata={"tool_name": "Bash", "command": "rm -rf /tmp/test"},
        )
        assert request.escalation_type == EscalationType.PERMISSION
        assert "Bash" in request.metadata["tool_name"]

    def test_create_error_request(self):
        """Create an error escalation request."""
        request = EscalationRequest.create(
            session_id="session-789",
            escalation_type=EscalationType.ERROR,
            context="Build failed",
            question="TypeScript error: Cannot find module 'lodash'",
            options=["Retry", "Skip", "Abort"],
        )
        assert request.escalation_type == EscalationType.ERROR
        assert len(request.options) == 3

    def test_request_is_pending(self):
        """Request should report pending status correctly."""
        request = EscalationRequest.create(
            session_id="test",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Test?",
        )
        assert request.is_pending
        assert request.status == EscalationStatus.PENDING

    def test_request_age_seconds(self):
        """Request age should increase over time."""
        request = EscalationRequest.create(
            session_id="test",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Test?",
        )
        assert request.age_seconds >= 0

    def test_with_response_creates_new_instance(self):
        """with_response should create a new immutable instance."""
        original = EscalationRequest.create(
            session_id="test",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
        )
        responded = original.with_response("Yes")

        assert original.is_pending
        assert original.response is None
        assert not responded.is_pending
        assert responded.response == "Yes"
        assert responded.status == EscalationStatus.RESPONDED
        assert responded.responded_at is not None

    def test_to_dict_and_from_dict(self):
        """Request should serialize and deserialize correctly."""
        original = EscalationRequest.create(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="Test context",
            question="Test question?",
            options=["A", "B"],
            brain_suggestion="A",
            brain_confidence=0.9,
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = EscalationRequest.from_dict(data)

        assert restored.id == original.id
        assert restored.session_id == original.session_id
        assert restored.escalation_type == original.escalation_type
        assert restored.context == original.context
        assert restored.question == original.question
        assert restored.options == original.options
        assert restored.brain_suggestion == original.brain_suggestion
        assert restored.brain_confidence == original.brain_confidence


class TestEscalationQueue:
    """Tests for EscalationQueue database operations."""

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, temp_dir: Path):
        """Queue should create database on initialize."""
        queue = EscalationQueue(data_dir=temp_dir)
        await queue.initialize()

        assert queue.db_path.exists()
        await queue.close()

    @pytest.mark.asyncio
    async def test_escalate_creates_request(self, queue: EscalationQueue):
        """escalate should create and persist a request."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="Test context",
            question="Test question?",
        )

        assert request.id is not None
        assert request.is_pending

        # Verify persistence
        retrieved = await queue.get_request(request.id)
        assert retrieved is not None
        assert retrieved.id == request.id
        assert retrieved.question == "Test question?"

        await queue.close()

    @pytest.mark.asyncio
    async def test_escalate_with_full_options(self, queue: EscalationQueue):
        """escalate should handle all parameters."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-456",
            escalation_type=EscalationType.PERMISSION,
            context="Risk level: MEDIUM",
            question="Allow git push?",
            options=["Approve", "Deny"],
            brain_suggestion="Approve",
            brain_confidence=0.75,
            metadata={"command": "git push origin main"},
        )

        retrieved = await queue.get_request(request.id)
        assert retrieved.options == ["Approve", "Deny"]
        assert retrieved.brain_suggestion == "Approve"
        assert retrieved.brain_confidence == 0.75
        assert retrieved.metadata["command"] == "git push origin main"

        await queue.close()

    @pytest.mark.asyncio
    async def test_respond_updates_request(self, queue: EscalationQueue):
        """respond should update request status and response."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
        )

        updated = await queue.respond(request.id, "Yes")

        assert updated is not None
        assert updated.status == EscalationStatus.RESPONDED
        assert updated.response == "Yes"
        assert updated.responded_at is not None

        await queue.close()

    @pytest.mark.asyncio
    async def test_respond_to_nonexistent_returns_none(self, queue: EscalationQueue):
        """respond to non-existent request should return None."""
        await queue.initialize()

        result = await queue.respond("nonexistent-id", "Yes")
        assert result is None

        await queue.close()

    @pytest.mark.asyncio
    async def test_respond_to_already_responded_returns_none(
        self, queue: EscalationQueue
    ):
        """respond to already responded request should return None."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
        )
        await queue.respond(request.id, "Yes")

        # Try to respond again
        result = await queue.respond(request.id, "No")
        assert result is None

        await queue.close()

    @pytest.mark.asyncio
    async def test_get_pending_returns_only_pending(self, queue: EscalationQueue):
        """get_pending should return only pending requests."""
        await queue.initialize()

        # Create mix of pending and responded
        r1 = await queue.escalate(
            session_id="s1",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q1?",
        )
        r2 = await queue.escalate(
            session_id="s2",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q2?",
        )
        await queue.respond(r1.id, "Answered")

        pending = await queue.get_pending()
        assert len(pending) == 1
        assert pending[0].id == r2.id

        await queue.close()

    @pytest.mark.asyncio
    async def test_get_pending_filters_by_session(self, queue: EscalationQueue):
        """get_pending should filter by session_id."""
        await queue.initialize()

        await queue.escalate(
            session_id="session-A",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q1?",
        )
        await queue.escalate(
            session_id="session-B",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q2?",
        )

        pending_a = await queue.get_pending(session_id="session-A")
        pending_b = await queue.get_pending(session_id="session-B")

        assert len(pending_a) == 1
        assert pending_a[0].session_id == "session-A"
        assert len(pending_b) == 1
        assert pending_b[0].session_id == "session-B"

        await queue.close()

    @pytest.mark.asyncio
    async def test_get_history_includes_all_statuses(self, queue: EscalationQueue):
        """get_history should include responded and cancelled."""
        await queue.initialize()

        r1 = await queue.escalate(
            session_id="session-1",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q1?",
        )
        r2 = await queue.escalate(
            session_id="session-1",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q2?",
        )
        await queue.respond(r1.id, "Yes")
        await queue.cancel(r2.id)

        history = await queue.get_history(session_id="session-1")
        assert len(history) == 2

        await queue.close()

    @pytest.mark.asyncio
    async def test_cancel_updates_status(self, queue: EscalationQueue):
        """cancel should mark request as cancelled."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
        )

        cancelled = await queue.cancel(request.id)
        assert cancelled is True

        retrieved = await queue.get_request(request.id)
        assert retrieved.status == EscalationStatus.CANCELLED

        await queue.close()

    @pytest.mark.asyncio
    async def test_clear_session_cancels_all_pending(self, queue: EscalationQueue):
        """clear_session should cancel all pending for a session."""
        await queue.initialize()

        await queue.escalate(
            session_id="session-to-clear",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q1?",
        )
        await queue.escalate(
            session_id="session-to-clear",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q2?",
        )
        await queue.escalate(
            session_id="other-session",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q3?",
        )

        count = await queue.clear_session("session-to-clear")
        assert count == 2

        # Verify session is cleared
        pending_cleared = await queue.get_pending(session_id="session-to-clear")
        pending_other = await queue.get_pending(session_id="other-session")
        assert len(pending_cleared) == 0
        assert len(pending_other) == 1

        await queue.close()

    @pytest.mark.asyncio
    async def test_get_statistics(self, queue: EscalationQueue):
        """get_statistics should return accurate counts."""
        await queue.initialize()

        r1 = await queue.escalate(
            session_id="s1",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Q1?",
        )
        r2 = await queue.escalate(
            session_id="s1",
            escalation_type=EscalationType.PERMISSION,
            context="",
            question="Allow?",
        )
        await queue.respond(r1.id, "Yes")

        stats = await queue.get_statistics()

        assert stats["total_escalations"] == 2
        assert stats["pending_count"] == 1
        assert "question" in stats["by_type"]
        assert "permission" in stats["by_type"]

        await queue.close()

    @pytest.mark.asyncio
    async def test_wait_for_response_returns_on_respond(self, queue: EscalationQueue):
        """wait_for_response should return when response is received."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="session-123",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
        )

        async def delayed_respond():
            await asyncio.sleep(0.1)
            await queue.respond(request.id, "Yes")

        # Start background responder
        respond_task = asyncio.create_task(delayed_respond())

        # Wait for response
        response = await queue.wait_for_response(request.id)

        assert response == "Yes"

        await respond_task
        await queue.close()


class TestRealWorldScenarios:
    """Tests simulating real Claude Code escalation scenarios."""

    @pytest.mark.asyncio
    async def test_claude_asks_which_database(self, queue: EscalationQueue):
        """Scenario: Claude asks which database to use."""
        await queue.initialize()

        # Claude asks a question during implementation
        request = await queue.escalate(
            session_id="impl-session-001",
            escalation_type=EscalationType.QUESTION,
            context="Implementing user authentication system",
            question="Which database should I use for storing user credentials?",
            options=["PostgreSQL", "SQLite", "MongoDB"],
            brain_suggestion="PostgreSQL",
            brain_confidence=0.6,  # Not confident enough to auto-answer
            metadata={"feature": "authentication", "phase": "implementation"},
        )

        # Human responds
        await queue.respond(request.id, "PostgreSQL")

        # Verify response was stored
        updated = await queue.get_request(request.id)
        assert updated.response == "PostgreSQL"
        assert updated.status == EscalationStatus.RESPONDED

        await queue.close()

    @pytest.mark.asyncio
    async def test_dangerous_command_permission(self, queue: EscalationQueue):
        """Scenario: Claude wants to run a dangerous command."""
        await queue.initialize()

        # Master Agent escalates HIGH risk permission
        request = await queue.escalate(
            session_id="deploy-session-002",
            escalation_type=EscalationType.PERMISSION,
            context="Risk level: HIGH - Contains rm -rf",
            question="Allow Bash: rm -rf node_modules && npm install?",
            options=["Approve", "Deny"],
            brain_suggestion="Deny",
            brain_confidence=0.3,  # Brain not confident
            metadata={
                "tool_name": "Bash",
                "command": "rm -rf node_modules && npm install",
                "risk_level": "HIGH",
            },
        )

        # Human approves after review
        await queue.respond(request.id, "Approve")

        updated = await queue.get_request(request.id)
        assert updated.response == "Approve"

        await queue.close()

    @pytest.mark.asyncio
    async def test_build_error_escalation(self, queue: EscalationQueue):
        """Scenario: Build fails and needs human decision."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="build-session-003",
            escalation_type=EscalationType.ERROR,
            context="TypeScript build failed after 3 retries",
            question=(
                "Build error: Cannot find module '@types/lodash'\n\n"
                "Attempts: 3/3\n"
                "How should I proceed?"
            ),
            options=["Retry", "Skip this step", "Install missing types", "Abort"],
            metadata={
                "error_type": "typescript_build",
                "error_code": "TS2307",
                "retry_count": 3,
            },
        )

        await queue.respond(request.id, "Install missing types")

        updated = await queue.get_request(request.id)
        assert updated.response == "Install missing types"

        await queue.close()

    @pytest.mark.asyncio
    async def test_git_force_push_confirmation(self, queue: EscalationQueue):
        """Scenario: Claude wants to force push to main."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="git-session-004",
            escalation_type=EscalationType.CONFIRMATION,
            context="Git operation on protected branch",
            question=(
                "You are about to force push to main branch.\n"
                "This will overwrite remote history.\n"
                "Are you sure?"
            ),
            options=["Yes, force push", "No, cancel"],
            brain_suggestion="No, cancel",
            brain_confidence=0.95,  # Brain strongly suggests cancel
            metadata={
                "command": "git push --force origin main",
                "branch": "main",
                "is_protected": True,
            },
        )

        # Human confirms brain's suggestion
        await queue.respond(request.id, "No, cancel")

        await queue.close()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_escalations(self, queue: EscalationQueue):
        """Scenario: Multiple sessions have pending escalations."""
        await queue.initialize()

        # Create escalations for different sessions
        sessions = ["auth-session", "api-session", "ui-session"]
        for session_id in sessions:
            await queue.escalate(
                session_id=session_id,
                escalation_type=EscalationType.QUESTION,
                context=f"Working on {session_id.split('-')[0]}",
                question="Should I proceed with the implementation?",
                options=["Yes", "No"],
            )

        # Get all pending
        all_pending = await queue.get_pending()
        assert len(all_pending) == 3

        # Respond to one
        await queue.respond(all_pending[0].id, "Yes")

        # Check remaining
        remaining = await queue.get_pending()
        assert len(remaining) == 2

        await queue.close()

    @pytest.mark.asyncio
    async def test_env_file_modification_review(self, queue: EscalationQueue):
        """Scenario: Claude wants to modify .env file."""
        await queue.initialize()

        request = await queue.escalate(
            session_id="config-session-005",
            escalation_type=EscalationType.REVIEW,
            context="Configuration file modification",
            question=(
                "I need to add the following to .env:\n"
                "```\n"
                "DATABASE_URL=postgresql://localhost/myapp\n"
                "SECRET_KEY=<generated>\n"
                "```\n"
                "Should I proceed?"
            ),
            options=["Approve", "Modify", "Reject"],
            brain_suggestion=None,  # No suggestion for sensitive files
            brain_confidence=0.0,
            metadata={
                "file_path": ".env",
                "operation": "write",
                "risk_level": "HIGH",
            },
        )

        await queue.respond(request.id, "Approve")

        await queue.close()


class TestNotificationCallback:
    """Tests for notification callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_called_on_escalate(self, temp_dir: Path):
        """Notification callback should be called on new escalation."""
        notifications = []

        def callback(request: EscalationRequest):
            notifications.append(request)

        queue = EscalationQueue(data_dir=temp_dir, notification_callback=callback)
        await queue.initialize()

        await queue.escalate(
            session_id="test",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Test?",
        )

        assert len(notifications) == 1
        assert notifications[0].question == "Test?"

        await queue.close()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_escalate(self, temp_dir: Path):
        """Callback exception should not prevent escalation."""

        def failing_callback(request: EscalationRequest):
            raise RuntimeError("Callback failed")

        queue = EscalationQueue(data_dir=temp_dir, notification_callback=failing_callback)
        await queue.initialize()

        # Should not raise
        request = await queue.escalate(
            session_id="test",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Test?",
        )

        assert request is not None
        assert request.is_pending

        await queue.close()
