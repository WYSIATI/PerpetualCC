"""Unit tests for session manager."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perpetualcc.claude.types import (
    InitEvent,
    RateLimitEvent,
    ResultEvent,
    SessionState,
    TextEvent,
)
from perpetualcc.core.checkpoint import SessionCheckpoint
from perpetualcc.core.rate_limit import RateLimitInfo, RateLimitType
from perpetualcc.core.session_manager import (
    EventCallback,
    ManagedSession,
    SessionManager,
    SessionManagerConfig,
    SessionStatus,
)
from perpetualcc.core.task_queue import TaskStatus


class TestManagedSession:
    """Tests for ManagedSession dataclass."""

    def test_create_basic_session(self):
        """ManagedSession.create should generate ID."""
        session = ManagedSession.create(
            project_path="/path/to/project",
            task="Implement feature",
        )
        assert session.id  # UUID generated
        assert session.project_path == str(Path("/path/to/project").resolve())
        assert session.current_task == "Implement feature"
        assert session.status == SessionStatus.IDLE
        assert session.created_at <= datetime.now()

    def test_create_with_metadata(self):
        """ManagedSession.create should accept metadata."""
        session = ManagedSession.create(
            project_path="/path/to/project",
            metadata={"priority": "high", "assignee": "developer"},
        )
        assert session.metadata["priority"] == "high"
        assert session.metadata["assignee"] == "developer"

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        session = ManagedSession.create(
            project_path="/path/to/project",
            task="Test task",
            metadata={"key": "value"},
        )
        data = session.to_dict()

        assert data["id"] == session.id
        assert data["project_path"] == session.project_path
        assert data["status"] == "idle"
        assert data["current_task"] == "Test task"
        assert data["metadata"] == {"key": "value"}
        assert data["claude_session_id"] is None
        assert data["started_at"] is None

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        data = {
            "id": "session-123",
            "claude_session_id": "claude-456",
            "project_path": "/path/to/project",
            "status": "processing",
            "current_task": "Running task",
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:05:00",
            "last_activity": "2024-01-15T10:10:00",
            "error_message": None,
            "token_count": 5000,
            "turn_count": 10,
            "total_cost_usd": 0.05,
            "metadata": {"env": "production"},
        }
        session = ManagedSession.from_dict(data)

        assert session.id == "session-123"
        assert session.claude_session_id == "claude-456"
        assert session.status == SessionStatus.PROCESSING
        assert session.current_task == "Running task"
        assert session.started_at == datetime(2024, 1, 15, 10, 5, 0)
        assert session.turn_count == 10
        assert session.total_cost_usd == 0.05

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {
            "id": "session-123",
            "project_path": "/path",
            "created_at": "2024-01-15T10:00:00",
        }
        session = ManagedSession.from_dict(data)

        assert session.status == SessionStatus.IDLE
        assert session.current_task is None
        assert session.token_count == 0
        assert session.metadata == {}

    def test_roundtrip(self):
        """to_dict and from_dict should be reversible."""
        original = ManagedSession.create(
            project_path="/path/to/project",
            task="Roundtrip test",
            metadata={"test": True},
        )
        data = original.to_dict()
        restored = ManagedSession.from_dict(data)

        assert restored.id == original.id
        assert restored.project_path == original.project_path
        assert restored.current_task == original.current_task
        assert restored.metadata == original.metadata

    def test_with_update(self):
        """with_update should create new session with updates."""
        session = ManagedSession.create(
            project_path="/path",
            task="Original",
        )
        updated = session.with_update(
            status=SessionStatus.PROCESSING,
            started_at=datetime.now(),
            token_count=1000,
        )

        # Original unchanged
        assert session.status == SessionStatus.IDLE
        assert session.started_at is None
        assert session.token_count == 0

        # New session updated
        assert updated.status == SessionStatus.PROCESSING
        assert updated.started_at is not None
        assert updated.token_count == 1000
        assert updated.id == session.id  # Same ID

    def test_is_active(self):
        """is_active should return True for active states."""
        session = ManagedSession.create(project_path="/path")

        # IDLE is not active
        assert session.is_active is False

        # PROCESSING is active
        processing = session.with_update(status=SessionStatus.PROCESSING)
        assert processing.is_active is True

        # WAITING_INPUT is active
        waiting = session.with_update(status=SessionStatus.WAITING_INPUT)
        assert waiting.is_active is True

        # RATE_LIMITED is active
        limited = session.with_update(status=SessionStatus.RATE_LIMITED)
        assert limited.is_active is True

        # PAUSED is not active
        paused = session.with_update(status=SessionStatus.PAUSED)
        assert paused.is_active is False

    def test_can_resume(self):
        """can_resume should return True for resumable states."""
        session = ManagedSession.create(project_path="/path")

        # IDLE can resume
        assert session.can_resume is True

        # PAUSED can resume
        paused = session.with_update(status=SessionStatus.PAUSED)
        assert paused.can_resume is True

        # RATE_LIMITED can resume
        limited = session.with_update(status=SessionStatus.RATE_LIMITED)
        assert limited.can_resume is True

        # COMPLETED cannot resume
        completed = session.with_update(status=SessionStatus.COMPLETED)
        assert completed.can_resume is False

        # ERROR cannot resume
        error = session.with_update(status=SessionStatus.ERROR)
        assert error.can_resume is False

        # PROCESSING cannot resume (already running)
        processing = session.with_update(status=SessionStatus.PROCESSING)
        assert processing.can_resume is False

    def test_is_completed(self):
        """is_completed should return True for terminal states."""
        session = ManagedSession.create(project_path="/path")
        assert session.is_completed is False

        completed = session.with_update(status=SessionStatus.COMPLETED)
        assert completed.is_completed is True

        error = session.with_update(status=SessionStatus.ERROR)
        assert error.is_completed is True


class TestSessionManagerConfig:
    """Tests for SessionManagerConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = SessionManagerConfig()
        assert config.data_dir == Path.home() / ".perpetualcc" / "data"
        assert config.max_concurrent_sessions == 5
        assert config.auto_resume is True
        assert config.persist_state is True
        assert config.confidence_threshold == 0.7

    def test_config_is_immutable(self):
        """Config should be immutable."""
        config = SessionManagerConfig()
        with pytest.raises(AttributeError):
            config.max_concurrent_sessions = 10

    def test_custom_config(self):
        """Config should accept custom values."""
        custom_dir = Path("/tmp/custom")
        config = SessionManagerConfig(
            data_dir=custom_dir,
            max_concurrent_sessions=10,
            auto_resume=False,
            confidence_threshold=0.8,
        )
        assert config.data_dir == custom_dir
        assert config.max_concurrent_sessions == 10
        assert config.auto_resume is False

    def test_db_path_property(self):
        """db_path should return full database path."""
        config = SessionManagerConfig(data_dir=Path("/tmp/test"))
        assert config.db_path == Path("/tmp/test/perpetualcc.db")


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir: Path) -> SessionManager:
        """Create a session manager with temp directory."""
        config = SessionManagerConfig(data_dir=temp_dir)
        return SessionManager(config)

    @pytest.fixture
    def temp_project(self, temp_dir: Path) -> Path:
        """Create a temporary project directory."""
        project_dir = temp_dir / "test-project"
        project_dir.mkdir()
        return project_dir

    def test_init_creates_directory(self, temp_dir: Path):
        """Manager should create data directory on init."""
        nested_dir = temp_dir / "nested" / "data"
        config = SessionManagerConfig(data_dir=nested_dir)
        manager = SessionManager(config)
        assert nested_dir.exists()

    @pytest.mark.asyncio
    async def test_create_session(self, manager: SessionManager, temp_project: Path):
        """create_session should create and persist a session."""
        session = await manager.create_session(
            project_path=temp_project,
            task="Implement user authentication",
        )

        assert session.id
        assert session.project_path == str(temp_project.resolve())
        assert session.current_task == "Implement user authentication"
        assert session.status == SessionStatus.IDLE

        # Verify it's tracked
        retrieved = manager.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(
        self, manager: SessionManager, temp_project: Path
    ):
        """create_session should accept metadata."""
        session = await manager.create_session(
            project_path=temp_project,
            task="Build API",
            metadata={"framework": "FastAPI", "version": "0.100.0"},
        )
        assert session.metadata["framework"] == "FastAPI"

    @pytest.mark.asyncio
    async def test_create_session_adds_task_to_queue(
        self, manager: SessionManager, temp_project: Path
    ):
        """create_session should add task to queue if provided."""
        session = await manager.create_session(
            project_path=temp_project,
            task="Initial task",
        )

        # Check task queue
        tasks = manager.task_queue.list_tasks(session.id)
        assert len(tasks) == 1
        assert tasks[0].description == "Initial task"

    @pytest.mark.asyncio
    async def test_create_session_max_concurrent(
        self, temp_dir: Path, temp_project: Path
    ):
        """create_session should enforce max concurrent sessions."""
        config = SessionManagerConfig(data_dir=temp_dir, max_concurrent_sessions=2)
        manager = SessionManager(config)

        # Create two sessions and start them
        s1 = await manager.create_session(temp_project, "Task 1")
        s2 = await manager.create_session(temp_project, "Task 2")

        # Mark them as active
        manager._sessions[s1.id] = s1.with_update(status=SessionStatus.PROCESSING)
        manager._sessions[s2.id] = s2.with_update(status=SessionStatus.PROCESSING)

        # Third should fail
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            await manager.create_session(temp_project, "Task 3")

    def test_get_session_existing(self, manager: SessionManager, temp_project: Path):
        """get_session should retrieve existing session."""
        session = asyncio.run(manager.create_session(temp_project, "Test"))
        retrieved = manager.get_session(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id

    def test_get_session_nonexistent(self, manager: SessionManager):
        """get_session should return None for nonexistent session."""
        result = manager.get_session("nonexistent-id")
        assert result is None

    def test_list_sessions_all(self, manager: SessionManager, temp_project: Path):
        """list_sessions should return all sessions."""
        asyncio.run(manager.create_session(temp_project, "Task 1"))
        asyncio.run(manager.create_session(temp_project, "Task 2"))

        sessions = manager.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_by_status(
        self, manager: SessionManager, temp_project: Path
    ):
        """list_sessions should filter by status."""
        s1 = asyncio.run(manager.create_session(temp_project, "Idle"))
        s2 = asyncio.run(manager.create_session(temp_project, "Processing"))

        # Update status
        manager._sessions[s2.id] = s2.with_update(status=SessionStatus.PROCESSING)

        idle = manager.list_sessions(status=SessionStatus.IDLE)
        assert len(idle) == 1
        assert idle[0].id == s1.id

        processing = manager.list_sessions(status=SessionStatus.PROCESSING)
        assert len(processing) == 1
        assert processing[0].id == s2.id

    def test_list_sessions_by_project(
        self, manager: SessionManager, temp_dir: Path
    ):
        """list_sessions should filter by project path."""
        project_a = temp_dir / "project-a"
        project_b = temp_dir / "project-b"
        project_a.mkdir()
        project_b.mkdir()

        asyncio.run(manager.create_session(project_a, "Task A"))
        asyncio.run(manager.create_session(project_b, "Task B"))

        sessions_a = manager.list_sessions(project_path=str(project_a))
        assert len(sessions_a) == 1
        assert sessions_a[0].project_path == str(project_a.resolve())

    def test_list_sessions_sorted_by_activity(
        self, manager: SessionManager, temp_project: Path
    ):
        """list_sessions should sort by last activity."""
        s1 = asyncio.run(manager.create_session(temp_project, "First"))
        s2 = asyncio.run(manager.create_session(temp_project, "Second"))

        # Update last activity
        manager._sessions[s1.id] = s1.with_update(
            last_activity=datetime.now() - timedelta(hours=1)
        )
        manager._sessions[s2.id] = s2.with_update(
            last_activity=datetime.now()
        )

        sessions = manager.list_sessions()
        # Most recent first
        assert sessions[0].id == s2.id
        assert sessions[1].id == s1.id

    @pytest.mark.asyncio
    async def test_add_task(self, manager: SessionManager, temp_project: Path):
        """add_task should add task to session queue."""
        session = await manager.create_session(temp_project, "Initial")
        task = await manager.add_task(session.id, "Additional task", priority=10)

        assert task.session_id == session.id
        assert task.description == "Additional task"
        assert task.priority == 10

        # Verify in queue
        tasks = manager.task_queue.list_tasks(session.id)
        assert len(tasks) == 2  # Initial + additional

    @pytest.mark.asyncio
    async def test_add_task_nonexistent_session(self, manager: SessionManager):
        """add_task should raise for nonexistent session."""
        with pytest.raises(ValueError, match="Session not found"):
            await manager.add_task("nonexistent", "Task")

    @pytest.mark.asyncio
    async def test_pause_session(self, manager: SessionManager, temp_project: Path):
        """pause_session should pause a running session."""
        session = await manager.create_session(temp_project, "Task")

        # Mark as active
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING
        )

        # Create a mock task
        manager._tasks[session.id] = asyncio.create_task(asyncio.sleep(10))

        paused = await manager.pause_session(session.id)
        assert paused.status == SessionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_pause_session_nonexistent(self, manager: SessionManager):
        """pause_session should raise for nonexistent session."""
        with pytest.raises(ValueError, match="Session not found"):
            await manager.pause_session("nonexistent")

    @pytest.mark.asyncio
    async def test_stop_session(self, manager: SessionManager, temp_project: Path):
        """stop_session should stop and cleanup session."""
        session = await manager.create_session(temp_project, "Task")

        # Mark as active
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING
        )

        stopped = await manager.stop_session(session.id)
        assert stopped.status == SessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_delete_session(self, manager: SessionManager, temp_project: Path):
        """delete_session should remove session and data."""
        session = await manager.create_session(temp_project, "Task")

        # Add more tasks
        await manager.add_task(session.id, "Task 2")

        result = await manager.delete_session(session.id)
        assert result is True

        # Session should be gone
        assert manager.get_session(session.id) is None

        # Tasks should be cleared
        tasks = manager.task_queue.list_tasks(session.id)
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, manager: SessionManager):
        """delete_session should return False for nonexistent session."""
        result = await manager.delete_session("nonexistent")
        assert result is False

    def test_get_statistics(self, manager: SessionManager, temp_project: Path):
        """get_statistics should return manager stats."""
        s1 = asyncio.run(manager.create_session(temp_project, "Task 1"))
        s2 = asyncio.run(manager.create_session(temp_project, "Task 2"))

        # Update statuses
        manager._sessions[s1.id] = s1.with_update(
            status=SessionStatus.PROCESSING,
            total_cost_usd=0.05,
            turn_count=10,
        )
        manager._sessions[s2.id] = s2.with_update(
            status=SessionStatus.COMPLETED,
            total_cost_usd=0.10,
            turn_count=20,
        )

        stats = manager.get_statistics()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 1
        assert stats["total_cost_usd"] == pytest.approx(0.15)
        assert stats["total_turns"] == 30
        assert stats["by_status"]["processing"] == 1
        assert stats["by_status"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_shutdown(self, manager: SessionManager, temp_project: Path):
        """shutdown should pause all active sessions."""
        s1 = await manager.create_session(temp_project, "Task 1")
        s2 = await manager.create_session(temp_project, "Task 2")

        # Make one active
        manager._sessions[s1.id] = s1.with_update(status=SessionStatus.PROCESSING)
        manager._tasks[s1.id] = asyncio.create_task(asyncio.sleep(10))

        await manager.shutdown()

        # Active session should be paused
        assert manager.get_session(s1.id).status == SessionStatus.PAUSED


class TestSessionManagerPersistence:
    """Tests for session manager persistence."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_session_persists_across_instances(self, temp_dir: Path):
        """Sessions should persist across manager instances."""
        config = SessionManagerConfig(data_dir=temp_dir)
        project = temp_dir / "project"
        project.mkdir()

        # Create session in first instance
        manager1 = SessionManager(config)
        session = asyncio.run(manager1.create_session(project, "Persistent task"))
        session_id = session.id

        # Create new instance
        manager2 = SessionManager(config)
        retrieved = manager2.get_session(session_id)

        assert retrieved is not None
        assert retrieved.current_task == "Persistent task"

    def test_completed_sessions_not_loaded(self, temp_dir: Path):
        """Completed sessions should not be loaded on startup."""
        config = SessionManagerConfig(data_dir=temp_dir)
        project = temp_dir / "project"
        project.mkdir()

        # Create and complete session
        manager1 = SessionManager(config)
        session = asyncio.run(manager1.create_session(project, "Task"))
        manager1._sessions[session.id] = session.with_update(
            status=SessionStatus.COMPLETED
        )
        manager1._save_session(manager1._sessions[session.id])

        # New instance should not load completed session
        manager2 = SessionManager(config)
        assert manager2.get_session(session.id) is None


class TestSessionManagerEventHandling:
    """Tests for session manager event handling."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir: Path) -> SessionManager:
        """Create a session manager with temp directory."""
        config = SessionManagerConfig(data_dir=temp_dir)
        return SessionManager(config)

    @pytest.fixture
    def temp_project(self, temp_dir: Path) -> Path:
        """Create a temporary project directory."""
        project_dir = temp_dir / "test-project"
        project_dir.mkdir()
        return project_dir

    @pytest.mark.asyncio
    async def test_event_callback_called(
        self, manager: SessionManager, temp_project: Path
    ):
        """Event callback should be called for events."""
        events_received = []

        def callback(session_id: str, event):
            events_received.append((session_id, event))

        manager.event_callback = callback

        session = await manager.create_session(temp_project, "Task")

        # Simulate event
        event = TextEvent(text="Hello from Claude")
        manager.event_callback(session.id, event)

        assert len(events_received) == 1
        assert events_received[0][0] == session.id
        assert isinstance(events_received[0][1], TextEvent)

    @pytest.mark.asyncio
    async def test_handle_rate_limit(
        self, manager: SessionManager, temp_project: Path
    ):
        """_handle_rate_limit should update session state and create checkpoint."""
        session = await manager.create_session(temp_project, "Task")
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING,
            claude_session_id="claude-123",
        )

        # Disable auto-resume for this test
        manager.config = SessionManagerConfig(
            data_dir=manager.config.data_dir,
            auto_resume=False,
        )

        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.TOKEN_LIMIT,
            retry_after_seconds=60,
            reset_time=datetime.now() + timedelta(seconds=60),
            message="Rate limited",
        )

        await manager._handle_rate_limit(session.id, info)

        updated = manager.get_session(session.id)
        assert updated.status == SessionStatus.RATE_LIMITED
        assert updated.rate_limit_info is not None

    @pytest.mark.asyncio
    async def test_handle_result_success(
        self, manager: SessionManager, temp_project: Path
    ):
        """_handle_result should mark session as completed on success."""
        session = await manager.create_session(temp_project, "Task")
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING
        )

        # Start task
        tasks = manager.task_queue.list_tasks(session.id)
        manager.task_queue.start(tasks[0].task_id)

        event = ResultEvent(
            is_error=False,
            result="Task completed successfully",
            num_turns=5,
            total_cost_usd=0.05,
        )

        await manager._handle_result(session.id, event)

        updated = manager.get_session(session.id)
        assert updated.status == SessionStatus.COMPLETED
        assert updated.turn_count == 5
        assert updated.total_cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_handle_result_error(
        self, manager: SessionManager, temp_project: Path
    ):
        """_handle_result should mark session as error on failure."""
        session = await manager.create_session(temp_project, "Task")
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING
        )

        event = ResultEvent(
            is_error=True,
            result="Build failed",
            num_turns=3,
            total_cost_usd=0.02,
        )

        await manager._handle_result(session.id, event)

        updated = manager.get_session(session.id)
        assert updated.status == SessionStatus.ERROR
        assert updated.error_message == "Build failed"

    @pytest.mark.asyncio
    async def test_handle_result_starts_next_task(
        self, manager: SessionManager, temp_project: Path
    ):
        """_handle_result should start next task if available."""
        session = await manager.create_session(temp_project, "Task 1")

        # Add another task
        await manager.add_task(session.id, "Task 2")

        # Start first task
        tasks = manager.task_queue.list_tasks(session.id)
        manager.task_queue.start(tasks[0].task_id)

        # Mark session as processing
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING
        )

        # Mock start_session to avoid actual Claude interaction
        with patch.object(manager, "start_session", new_callable=AsyncMock) as mock_start:
            event = ResultEvent(is_error=False, num_turns=5)
            await manager._handle_result(session.id, event)

            # Should have tried to start next task
            # The session should be in IDLE waiting for start
            updated = manager.get_session(session.id)
            # First task completed, second task ready
            assert mock_start.called or updated.current_task == "Task 2"


class TestSessionManagerRealWorldScenarios:
    """Tests with real-world Claude Code session scenarios."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir: Path) -> SessionManager:
        """Create a session manager with temp directory."""
        config = SessionManagerConfig(data_dir=temp_dir)
        return SessionManager(config)

    @pytest.fixture
    def temp_project(self, temp_dir: Path) -> Path:
        """Create a temporary project directory."""
        project_dir = temp_dir / "test-project"
        project_dir.mkdir()
        return project_dir

    @pytest.mark.asyncio
    async def test_multi_project_development(
        self, manager: SessionManager, temp_dir: Path
    ):
        """Simulate working on multiple projects simultaneously."""
        # Create project directories
        frontend = temp_dir / "frontend-app"
        backend = temp_dir / "backend-api"
        frontend.mkdir()
        backend.mkdir()

        # Create sessions for both projects
        fe_session = await manager.create_session(
            frontend,
            "Build React login form",
            metadata={"tech": "React", "priority": "high"},
        )

        be_session = await manager.create_session(
            backend,
            "Create authentication endpoints",
            metadata={"tech": "FastAPI", "priority": "high"},
        )

        # Add more tasks to each
        await manager.add_task(fe_session.id, "Add form validation")
        await manager.add_task(fe_session.id, "Write component tests")

        await manager.add_task(be_session.id, "Add JWT token handling")
        await manager.add_task(be_session.id, "Write API tests")

        # List by project
        fe_sessions = manager.list_sessions(project_path=str(frontend))
        be_sessions = manager.list_sessions(project_path=str(backend))

        assert len(fe_sessions) == 1
        assert len(be_sessions) == 1
        assert fe_sessions[0].metadata["tech"] == "React"
        assert be_sessions[0].metadata["tech"] == "FastAPI"

        # Check task queues
        fe_tasks = manager.task_queue.list_tasks(fe_session.id)
        be_tasks = manager.task_queue.list_tasks(be_session.id)

        assert len(fe_tasks) == 3
        assert len(be_tasks) == 3

    @pytest.mark.asyncio
    async def test_session_recovery_workflow(
        self, manager: SessionManager, temp_project: Path
    ):
        """Simulate session crash and recovery."""
        # Create and start a session
        session = await manager.create_session(
            temp_project,
            "Long running refactoring task",
        )

        # Simulate the session was running
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING,
            claude_session_id="claude-sess-123",
            started_at=datetime.now() - timedelta(minutes=30),
            last_activity=datetime.now() - timedelta(minutes=5),
            turn_count=15,
        )
        manager._save_session(manager._sessions[session.id])

        # Simulate crash - create new manager instance
        config = manager.config
        manager2 = SessionManager(config)

        # Session should be loaded
        recovered = manager2.get_session(session.id)
        assert recovered is not None
        assert recovered.claude_session_id == "claude-sess-123"
        assert recovered.turn_count == 15

        # Can resume from where it left off
        assert recovered.can_resume or recovered.status == SessionStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_rate_limit_handling_workflow(
        self, temp_dir: Path, temp_project: Path
    ):
        """Simulate rate limit and recovery workflow."""
        config = SessionManagerConfig(data_dir=temp_dir, auto_resume=False)
        manager = SessionManager(config)

        session = await manager.create_session(temp_project, "Build feature")

        # Simulate session running
        manager._sessions[session.id] = session.with_update(
            status=SessionStatus.PROCESSING,
            claude_session_id="claude-123",
        )

        # Simulate rate limit
        info = RateLimitInfo(
            detected_at=datetime.now(),
            limit_type=RateLimitType.TOKEN_LIMIT,
            retry_after_seconds=300,
            reset_time=datetime.now() + timedelta(seconds=300),
            message="Token limit reached. Retry after 300 seconds.",
        )

        await manager._handle_rate_limit(session.id, info)

        # Check session state
        limited = manager.get_session(session.id)
        assert limited.status == SessionStatus.RATE_LIMITED
        assert limited.checkpoint is not None

        # Simulate time passing and manual resume
        # In real scenario, auto_resume would handle this

    @pytest.mark.asyncio
    async def test_task_chain_execution(
        self, manager: SessionManager, temp_project: Path
    ):
        """Simulate chained task execution."""
        session = await manager.create_session(temp_project, "Setup project")

        # Add chained tasks
        setup_task = manager.task_queue.list_tasks(session.id)[0]
        await manager.add_task(
            session.id,
            "Install dependencies",
            priority=9,
        )
        await manager.add_task(
            session.id,
            "Create database schema",
            priority=8,
        )
        await manager.add_task(
            session.id,
            "Build API endpoints",
            priority=7,
        )
        await manager.add_task(
            session.id,
            "Write tests",
            priority=6,
        )

        # Verify task order (sorted by priority descending, then created_at)
        tasks = manager.task_queue.list_tasks(session.id)
        assert len(tasks) == 5
        # Tasks are ordered by priority (higher number = higher priority)
        assert tasks[0].description == "Install dependencies"  # priority 9
        assert tasks[1].description == "Create database schema"  # priority 8
        assert tasks[2].description == "Build API endpoints"  # priority 7
        assert tasks[3].description == "Write tests"  # priority 6
        assert tasks[4].description == "Setup project"  # default priority 5

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(
        self, manager: SessionManager, temp_project: Path
    ):
        """Test cleanup of completed sessions."""
        # Create multiple sessions with different states
        active = await manager.create_session(temp_project, "Active task")
        completed = await manager.create_session(temp_project, "Completed task")
        failed = await manager.create_session(temp_project, "Failed task")

        # Update states
        manager._sessions[completed.id] = completed.with_update(
            status=SessionStatus.COMPLETED
        )
        manager._sessions[failed.id] = failed.with_update(
            status=SessionStatus.ERROR,
            error_message="Build failed",
        )
        manager._save_session(manager._sessions[completed.id])
        manager._save_session(manager._sessions[failed.id])

        # Delete completed sessions
        await manager.delete_session(completed.id)
        await manager.delete_session(failed.id)

        # Only active remains
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].id == active.id

    @pytest.mark.asyncio
    async def test_concurrent_session_limit_enforcement(
        self, temp_dir: Path, temp_project: Path
    ):
        """Test enforcement of concurrent session limits."""
        config = SessionManagerConfig(data_dir=temp_dir, max_concurrent_sessions=3)
        manager = SessionManager(config)

        sessions = []
        for i in range(3):
            s = await manager.create_session(temp_project, f"Task {i}")
            manager._sessions[s.id] = s.with_update(status=SessionStatus.PROCESSING)
            sessions.append(s)

        # Should not allow 4th
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            await manager.create_session(temp_project, "Task 4")

        # Complete one
        manager._sessions[sessions[0].id] = sessions[0].with_update(
            status=SessionStatus.COMPLETED
        )

        # Now we can create another
        s4 = await manager.create_session(temp_project, "Task 4")
        assert s4.id is not None
