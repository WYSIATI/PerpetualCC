"""Unit tests for checkpoint management."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from perpetualcc.claude.types import SessionState
from perpetualcc.core.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    SessionCheckpoint,
    ToolUseRecord,
)


class TestToolUseRecord:
    """Tests for ToolUseRecord dataclass."""

    @pytest.fixture
    def tool_record(self) -> ToolUseRecord:
        """Create a sample tool use record."""
        return ToolUseRecord(
            tool_use_id="tool-123",
            tool_name="Read",
            tool_input={"file_path": "/path/to/file.py"},
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            result="file contents",
            is_error=False,
        )

    def test_basic_creation(self, tool_record: ToolUseRecord):
        """ToolUseRecord should store all fields."""
        assert tool_record.tool_use_id == "tool-123"
        assert tool_record.tool_name == "Read"
        assert tool_record.tool_input == {"file_path": "/path/to/file.py"}
        assert tool_record.result == "file contents"
        assert tool_record.is_error is False

    def test_to_dict(self, tool_record: ToolUseRecord):
        """to_dict should serialize all fields."""
        data = tool_record.to_dict()
        assert data["tool_use_id"] == "tool-123"
        assert data["tool_name"] == "Read"
        assert data["tool_input"] == {"file_path": "/path/to/file.py"}
        assert data["timestamp"] == "2024-01-15T10:30:00"
        assert data["result"] == "file contents"
        assert data["is_error"] is False

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        data = {
            "tool_use_id": "tool-456",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
            "timestamp": "2024-01-15T11:00:00",
            "result": "file list",
            "is_error": False,
        }
        record = ToolUseRecord.from_dict(data)
        assert record.tool_use_id == "tool-456"
        assert record.tool_name == "Bash"
        assert record.timestamp == datetime(2024, 1, 15, 11, 0, 0)

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {
            "tool_use_id": "tool-789",
            "tool_name": "Glob",
            "tool_input": {"pattern": "*.py"},
            "timestamp": "2024-01-15T12:00:00",
        }
        record = ToolUseRecord.from_dict(data)
        assert record.result is None
        assert record.is_error is False

    def test_roundtrip(self, tool_record: ToolUseRecord):
        """to_dict and from_dict should be reversible."""
        data = tool_record.to_dict()
        restored = ToolUseRecord.from_dict(data)
        assert restored.tool_use_id == tool_record.tool_use_id
        assert restored.tool_name == tool_record.tool_name
        assert restored.tool_input == tool_record.tool_input
        assert restored.timestamp == tool_record.timestamp
        assert restored.result == tool_record.result
        assert restored.is_error == tool_record.is_error


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CheckpointConfig()
        assert config.checkpoint_dir == Path.home() / ".perpetualcc" / "data" / "checkpoints"
        assert config.max_checkpoints == 10
        assert config.auto_checkpoint_interval == 0
        assert config.compress is False

    def test_config_is_immutable(self):
        """Config should be immutable."""
        config = CheckpointConfig()
        with pytest.raises(AttributeError):
            config.max_checkpoints = 20

    def test_custom_config(self):
        """Config should accept custom values."""
        custom_dir = Path("/tmp/checkpoints")
        config = CheckpointConfig(
            checkpoint_dir=custom_dir,
            max_checkpoints=5,
            auto_checkpoint_interval=100,
            compress=True,
        )
        assert config.checkpoint_dir == custom_dir
        assert config.max_checkpoints == 5
        assert config.auto_checkpoint_interval == 100
        assert config.compress is True


class TestSessionCheckpoint:
    """Tests for SessionCheckpoint dataclass."""

    @pytest.fixture
    def basic_checkpoint(self) -> SessionCheckpoint:
        """Create a basic checkpoint."""
        return SessionCheckpoint(
            checkpoint_id="cp-123",
            session_id="sess-456",
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            project_path="/path/to/project",
            original_task="Implement feature X",
        )

    @pytest.fixture
    def full_checkpoint(self) -> SessionCheckpoint:
        """Create a checkpoint with all fields populated."""
        return SessionCheckpoint(
            checkpoint_id="cp-full",
            session_id="sess-full",
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            project_path="/path/to/project",
            original_task="Implement feature X",
            session_state=SessionState.RATE_LIMITED,
            last_event_type="rate_limit",
            last_event_timestamp=datetime(2024, 1, 15, 10, 30, 0),
            tool_history=[
                ToolUseRecord(
                    tool_use_id="tool-1",
                    tool_name="Read",
                    tool_input={"file_path": "test.py"},
                    timestamp=datetime(2024, 1, 15, 10, 15, 0),
                )
            ],
            pending_questions=[{"question": "Continue?", "options": ["Yes", "No"]}],
            rate_limit_info={"retry_after": 60, "message": "Rate limited"},
            decision_count=5,
            error_message=None,
            metadata={"attempt": 2},
        )

    def test_basic_creation(self, basic_checkpoint: SessionCheckpoint):
        """Basic checkpoint should have required fields."""
        assert basic_checkpoint.checkpoint_id == "cp-123"
        assert basic_checkpoint.session_id == "sess-456"
        assert basic_checkpoint.project_path == "/path/to/project"
        assert basic_checkpoint.original_task == "Implement feature X"
        assert basic_checkpoint.session_state == SessionState.IDLE

    def test_create_factory_method(self):
        """create() should generate ID and timestamp."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess-new",
            project_path="/path/to/project",
            original_task="New task",
        )
        assert checkpoint.checkpoint_id  # UUID generated
        assert checkpoint.session_id == "sess-new"
        assert checkpoint.created_at <= datetime.now()

    def test_create_with_kwargs(self):
        """create() should accept additional kwargs."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess-new",
            project_path="/path/to/project",
            original_task="New task",
            session_state=SessionState.PROCESSING,
            decision_count=3,
        )
        assert checkpoint.session_state == SessionState.PROCESSING
        assert checkpoint.decision_count == 3

    def test_to_dict(self, full_checkpoint: SessionCheckpoint):
        """to_dict should serialize all fields correctly."""
        data = full_checkpoint.to_dict()
        assert data["checkpoint_id"] == "cp-full"
        assert data["session_id"] == "sess-full"
        assert data["created_at"] == "2024-01-15T10:00:00"
        assert data["project_path"] == "/path/to/project"
        assert data["original_task"] == "Implement feature X"
        assert data["session_state"] == "rate_limited"
        assert data["last_event_type"] == "rate_limit"
        assert data["last_event_timestamp"] == "2024-01-15T10:30:00"
        assert len(data["tool_history"]) == 1
        assert data["tool_history"][0]["tool_name"] == "Read"
        assert len(data["pending_questions"]) == 1
        assert data["rate_limit_info"]["retry_after"] == 60
        assert data["decision_count"] == 5
        assert data["error_message"] is None
        assert data["metadata"]["attempt"] == 2

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        data = {
            "checkpoint_id": "cp-test",
            "session_id": "sess-test",
            "created_at": "2024-01-15T10:00:00",
            "project_path": "/path/to/project",
            "original_task": "Test task",
            "session_state": "processing",
            "last_event_type": "tool_use",
            "last_event_timestamp": "2024-01-15T10:15:00",
            "tool_history": [],
            "pending_questions": [],
            "rate_limit_info": None,
            "decision_count": 2,
            "error_message": None,
            "metadata": {},
        }
        checkpoint = SessionCheckpoint.from_dict(data)
        assert checkpoint.checkpoint_id == "cp-test"
        assert checkpoint.session_state == SessionState.PROCESSING
        assert checkpoint.last_event_timestamp == datetime(2024, 1, 15, 10, 15, 0)

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {
            "checkpoint_id": "cp-minimal",
            "session_id": "sess-minimal",
            "created_at": "2024-01-15T10:00:00",
            "project_path": "/path",
            "original_task": "Minimal task",
        }
        checkpoint = SessionCheckpoint.from_dict(data)
        assert checkpoint.session_state == SessionState.IDLE
        assert checkpoint.last_event_type is None
        assert checkpoint.last_event_timestamp is None
        assert checkpoint.tool_history == []
        assert checkpoint.decision_count == 0

    def test_roundtrip(self, full_checkpoint: SessionCheckpoint):
        """to_dict and from_dict should be reversible."""
        data = full_checkpoint.to_dict()
        restored = SessionCheckpoint.from_dict(data)
        assert restored.checkpoint_id == full_checkpoint.checkpoint_id
        assert restored.session_id == full_checkpoint.session_id
        assert restored.session_state == full_checkpoint.session_state
        assert len(restored.tool_history) == len(full_checkpoint.tool_history)
        assert restored.decision_count == full_checkpoint.decision_count

    def test_with_update(self, basic_checkpoint: SessionCheckpoint):
        """with_update should create new checkpoint with updates."""
        updated = basic_checkpoint.with_update(
            session_state=SessionState.PROCESSING,
            decision_count=10,
        )
        # Original unchanged
        assert basic_checkpoint.session_state == SessionState.IDLE
        assert basic_checkpoint.decision_count == 0
        # New checkpoint updated
        assert updated.session_state == SessionState.PROCESSING
        assert updated.decision_count == 10
        # Other fields preserved
        assert updated.checkpoint_id == basic_checkpoint.checkpoint_id
        assert updated.original_task == basic_checkpoint.original_task

    def test_with_update_datetime(self, basic_checkpoint: SessionCheckpoint):
        """with_update should handle datetime fields."""
        new_timestamp = datetime(2024, 1, 15, 12, 0, 0)
        updated = basic_checkpoint.with_update(
            last_event_timestamp=new_timestamp,
        )
        assert updated.last_event_timestamp == new_timestamp

    def test_can_resume_idle(self, basic_checkpoint: SessionCheckpoint):
        """IDLE state should be resumable."""
        assert basic_checkpoint.can_resume is True

    def test_can_resume_rate_limited(self, full_checkpoint: SessionCheckpoint):
        """RATE_LIMITED state should be resumable."""
        assert full_checkpoint.can_resume is True

    def test_can_resume_completed(self, basic_checkpoint: SessionCheckpoint):
        """COMPLETED state should not be resumable."""
        completed = basic_checkpoint.with_update(session_state=SessionState.COMPLETED)
        assert completed.can_resume is False

    def test_can_resume_error(self, basic_checkpoint: SessionCheckpoint):
        """ERROR state should not be resumable."""
        errored = basic_checkpoint.with_update(session_state=SessionState.ERROR)
        assert errored.can_resume is False

    def test_generate_resume_prompt_rate_limited(self):
        """Resume prompt for rate limited session."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess",
            project_path="/path",
            original_task="Build feature",
            session_state=SessionState.RATE_LIMITED,
        )
        prompt = checkpoint.generate_resume_prompt()
        assert "rate limit" in prompt.lower()
        assert "Build feature" in prompt

    def test_generate_resume_prompt_paused(self):
        """Resume prompt for paused session."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess",
            project_path="/path",
            original_task="Build feature",
            session_state=SessionState.PAUSED,
        )
        prompt = checkpoint.generate_resume_prompt()
        assert "paused" in prompt.lower()
        assert "Build feature" in prompt

    def test_generate_resume_prompt_with_questions(self):
        """Resume prompt when questions are pending."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess",
            project_path="/path",
            original_task="Build feature",
            pending_questions=[{"question": "Which framework?"}],
        )
        prompt = checkpoint.generate_resume_prompt()
        assert "question" in prompt.lower()
        assert "Build feature" in prompt

    def test_generate_resume_prompt_default(self):
        """Resume prompt for normal continuation."""
        checkpoint = SessionCheckpoint.create(
            session_id="sess",
            project_path="/path",
            original_task="Build feature",
        )
        prompt = checkpoint.generate_resume_prompt()
        assert "Build feature" in prompt


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir: Path) -> CheckpointManager:
        """Create a checkpoint manager with temp directory."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        return CheckpointManager(config)

    @pytest.fixture
    def sample_checkpoint(self) -> SessionCheckpoint:
        """Create a sample checkpoint for testing."""
        return SessionCheckpoint.create(
            session_id="test-session-123",
            project_path="/path/to/project",
            original_task="Implement testing",
            session_state=SessionState.PROCESSING,
            decision_count=5,
        )

    def test_init_creates_directory(self, temp_dir: Path):
        """Manager should create checkpoint directory on init."""
        config = CheckpointConfig(checkpoint_dir=temp_dir / "nested" / "checkpoints")
        manager = CheckpointManager(config)
        assert manager.config.checkpoint_dir.exists()

    def test_save_creates_file(
        self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint, temp_dir: Path
    ):
        """save() should create a checkpoint file."""
        path = manager.save(sample_checkpoint)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_file_content(
        self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint
    ):
        """Saved file should contain valid JSON."""
        path = manager.save(sample_checkpoint)
        data = json.loads(path.read_text())
        assert data["session_id"] == sample_checkpoint.session_id
        assert data["original_task"] == sample_checkpoint.original_task

    def test_save_atomic(
        self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint, temp_dir: Path
    ):
        """save() should not leave temp files on success."""
        manager.save(sample_checkpoint)
        # Check for .tmp files
        tmp_files = list(temp_dir.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_load(self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint):
        """load() should retrieve saved checkpoint."""
        manager.save(sample_checkpoint)
        loaded = manager.load(sample_checkpoint.session_id, sample_checkpoint.checkpoint_id)
        assert loaded.session_id == sample_checkpoint.session_id
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.original_task == sample_checkpoint.original_task

    def test_load_not_found(self, manager: CheckpointManager):
        """load() should raise FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent-session", "nonexistent-checkpoint")

    def test_load_latest(self, manager: CheckpointManager):
        """load_latest() should return most recent checkpoint."""
        session_id = "test-session"

        # Create multiple checkpoints
        cp1 = SessionCheckpoint.create(
            session_id=session_id,
            project_path="/path",
            original_task="Task 1",
        )
        manager.save(cp1)

        cp2 = SessionCheckpoint.create(
            session_id=session_id,
            project_path="/path",
            original_task="Task 2",
        )
        manager.save(cp2)

        latest = manager.load_latest(session_id)
        assert latest is not None
        assert latest.checkpoint_id == cp2.checkpoint_id

    def test_load_latest_no_checkpoints(self, manager: CheckpointManager):
        """load_latest() should return None when no checkpoints exist."""
        latest = manager.load_latest("nonexistent-session")
        assert latest is None

    def test_list_checkpoints(self, manager: CheckpointManager):
        """list_checkpoints() should return all checkpoints for a session."""
        session_id = "test-session"

        # Create multiple checkpoints
        for i in range(3):
            cp = SessionCheckpoint.create(
                session_id=session_id,
                project_path="/path",
                original_task=f"Task {i}",
            )
            manager.save(cp)

        checkpoints = manager.list_checkpoints(session_id)
        assert len(checkpoints) == 3
        # Should be sorted by creation time
        assert checkpoints[0]["created_at"] <= checkpoints[1]["created_at"]

    def test_list_checkpoints_empty(self, manager: CheckpointManager):
        """list_checkpoints() should return empty list for nonexistent session."""
        checkpoints = manager.list_checkpoints("nonexistent")
        assert checkpoints == []

    def test_list_sessions(self, manager: CheckpointManager):
        """list_sessions() should return all sessions with checkpoints."""
        # Create checkpoints for multiple sessions
        for i in range(3):
            cp = SessionCheckpoint.create(
                session_id=f"session-{i}",
                project_path="/path",
                original_task=f"Task {i}",
            )
            manager.save(cp)

        sessions = manager.list_sessions()
        assert len(sessions) == 3
        # Should have expected fields
        assert all("session_id" in s for s in sessions)
        assert all("checkpoint_count" in s for s in sessions)
        assert all("latest_checkpoint" in s for s in sessions)

    def test_list_sessions_sorted_by_latest(self, manager: CheckpointManager):
        """list_sessions() should sort by latest checkpoint time."""
        # Create sessions with different times
        sessions = manager.list_sessions()
        if len(sessions) > 1:
            for i in range(len(sessions) - 1):
                assert sessions[i]["latest_created_at"] >= sessions[i + 1]["latest_created_at"]

    def test_delete_checkpoint(self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint):
        """delete_checkpoint() should remove specific checkpoint."""
        manager.save(sample_checkpoint)
        assert manager.load_latest(sample_checkpoint.session_id) is not None

        result = manager.delete_checkpoint(
            sample_checkpoint.session_id, sample_checkpoint.checkpoint_id
        )
        assert result is True
        assert manager.load_latest(sample_checkpoint.session_id) is None

    def test_delete_checkpoint_not_found(self, manager: CheckpointManager):
        """delete_checkpoint() should return False when not found."""
        result = manager.delete_checkpoint("nonexistent", "nonexistent")
        assert result is False

    def test_delete_session(self, manager: CheckpointManager):
        """delete_session() should remove all checkpoints for session."""
        session_id = "to-delete"

        # Create multiple checkpoints
        for i in range(3):
            cp = SessionCheckpoint.create(
                session_id=session_id,
                project_path="/path",
                original_task=f"Task {i}",
            )
            manager.save(cp)

        assert len(manager.list_checkpoints(session_id)) == 3

        result = manager.delete_session(session_id)
        assert result is True
        assert len(manager.list_checkpoints(session_id)) == 0

    def test_delete_session_not_found(self, manager: CheckpointManager):
        """delete_session() should return False when not found."""
        result = manager.delete_session("nonexistent")
        assert result is False

    def test_cleanup_old_checkpoints(self, temp_dir: Path):
        """Manager should clean up old checkpoints beyond limit."""
        config = CheckpointConfig(checkpoint_dir=temp_dir, max_checkpoints=3)
        manager = CheckpointManager(config)

        session_id = "cleanup-test"

        # Create more checkpoints than the limit
        for i in range(5):
            cp = SessionCheckpoint.create(
                session_id=session_id,
                project_path="/path",
                original_task=f"Task {i}",
            )
            manager.save(cp)

        # Should only have max_checkpoints remaining
        checkpoints = manager.list_checkpoints(session_id)
        assert len(checkpoints) == 3

    def test_restore_returns_resume_prompt(
        self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint
    ):
        """restore() should return resume prompt."""
        manager.save(sample_checkpoint)
        loaded = manager.load(sample_checkpoint.session_id, sample_checkpoint.checkpoint_id)
        prompt = manager.restore(loaded)
        assert isinstance(prompt, str)
        assert sample_checkpoint.original_task in prompt

    def test_get_storage_info_empty(self, manager: CheckpointManager):
        """get_storage_info() should handle empty storage."""
        info = manager.get_storage_info()
        assert info["session_count"] == 0
        assert info["total_checkpoints"] == 0
        assert info["total_size_bytes"] == 0

    def test_get_storage_info_with_data(
        self, manager: CheckpointManager, sample_checkpoint: SessionCheckpoint
    ):
        """get_storage_info() should report correct counts."""
        manager.save(sample_checkpoint)
        info = manager.get_storage_info()
        assert info["session_count"] == 1
        assert info["total_checkpoints"] == 1
        assert info["total_size_bytes"] > 0


class TestCheckpointIntegration:
    """Integration tests for checkpoint components."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_checkpoint_lifecycle(self, temp_dir: Path):
        """Test complete checkpoint save/load/update/delete cycle."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        manager = CheckpointManager(config)

        # Create initial checkpoint
        checkpoint = SessionCheckpoint.create(
            session_id="lifecycle-test",
            project_path="/path/to/project",
            original_task="Complete lifecycle test",
            session_state=SessionState.IDLE,
        )
        manager.save(checkpoint)

        # Update checkpoint state
        updated = checkpoint.with_update(
            session_state=SessionState.PROCESSING,
            decision_count=5,
            tool_history=[
                ToolUseRecord(
                    tool_use_id="tool-1",
                    tool_name="Read",
                    tool_input={"file_path": "test.py"},
                    timestamp=datetime.now(),
                )
            ],
        )
        manager.save(updated)

        # Load latest
        loaded = manager.load_latest("lifecycle-test")
        assert loaded is not None
        assert loaded.session_state == SessionState.PROCESSING
        assert loaded.decision_count == 5
        assert len(loaded.tool_history) == 1

        # Simulate rate limit
        rate_limited = loaded.with_update(
            session_state=SessionState.RATE_LIMITED,
            rate_limit_info={"retry_after": 60},
        )
        manager.save(rate_limited)

        # Verify can_resume
        latest = manager.load_latest("lifecycle-test")
        assert latest.can_resume is True
        assert latest.session_state == SessionState.RATE_LIMITED

        # Generate resume prompt
        prompt = latest.generate_resume_prompt()
        assert "rate limit" in prompt.lower()

        # Complete and clean up
        completed = latest.with_update(session_state=SessionState.COMPLETED)
        manager.save(completed)

        final = manager.load_latest("lifecycle-test")
        assert final.can_resume is False

        # Delete session
        manager.delete_session("lifecycle-test")
        assert manager.load_latest("lifecycle-test") is None

    def test_multiple_sessions_isolation(self, temp_dir: Path):
        """Checkpoints from different sessions should be isolated."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        manager = CheckpointManager(config)

        # Create checkpoints for two sessions
        cp1 = SessionCheckpoint.create(
            session_id="session-a",
            project_path="/path/a",
            original_task="Task A",
        )
        cp2 = SessionCheckpoint.create(
            session_id="session-b",
            project_path="/path/b",
            original_task="Task B",
        )

        manager.save(cp1)
        manager.save(cp2)

        # Each session should have its own checkpoint
        sessions = manager.list_sessions()
        assert len(sessions) == 2

        a_checkpoints = manager.list_checkpoints("session-a")
        b_checkpoints = manager.list_checkpoints("session-b")
        assert len(a_checkpoints) == 1
        assert len(b_checkpoints) == 1

        # Deleting one session shouldn't affect the other
        manager.delete_session("session-a")
        assert manager.load_latest("session-a") is None
        assert manager.load_latest("session-b") is not None

    def test_checkpoint_with_complex_tool_history(self, temp_dir: Path):
        """Checkpoint should handle complex tool history."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        manager = CheckpointManager(config)

        # Create checkpoint with many tool uses
        tool_history = [
            ToolUseRecord(
                tool_use_id=f"tool-{i}",
                tool_name=["Read", "Write", "Bash", "Glob"][i % 4],
                tool_input={"param": f"value-{i}"},
                timestamp=datetime.now() + timedelta(minutes=i),
                result=f"result-{i}" if i % 2 == 0 else None,
                is_error=i % 5 == 0,
            )
            for i in range(20)
        ]

        checkpoint = SessionCheckpoint.create(
            session_id="complex-history",
            project_path="/path",
            original_task="Complex task",
            tool_history=tool_history,
        )
        manager.save(checkpoint)

        loaded = manager.load_latest("complex-history")
        assert len(loaded.tool_history) == 20
        assert loaded.tool_history[0].tool_use_id == "tool-0"
        assert loaded.tool_history[19].tool_use_id == "tool-19"
        assert loaded.tool_history[0].is_error is True  # 0 % 5 == 0
        assert loaded.tool_history[1].is_error is False

    def test_corrupted_checkpoint_handling(self, temp_dir: Path):
        """Manager should handle corrupted checkpoint files gracefully."""
        config = CheckpointConfig(checkpoint_dir=temp_dir)
        manager = CheckpointManager(config)

        # Create a valid checkpoint
        cp = SessionCheckpoint.create(
            session_id="corruption-test",
            project_path="/path",
            original_task="Test",
        )
        manager.save(cp)

        # Corrupt the file
        session_dir = temp_dir / "corruption-test"
        checkpoint_files = list(session_dir.glob("*.json"))
        assert len(checkpoint_files) == 1
        checkpoint_files[0].write_text("{ invalid json }")

        # list_checkpoints should skip corrupted files
        checkpoints = manager.list_checkpoints("corruption-test")
        assert len(checkpoints) == 0  # Corrupted file is skipped
