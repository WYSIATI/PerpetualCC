"""Unit tests for task queue with SQLite persistence."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from perpetualcc.core.task_queue import (
    Task,
    TaskPriority,
    TaskQueue,
    TaskQueueConfig,
    TaskStatus,
)


class TestTaskDataclass:
    """Tests for Task dataclass."""

    def test_create_basic_task(self):
        """Task.create should generate ID and timestamp."""
        task = Task.create(
            session_id="session-123",
            description="Implement user authentication",
        )
        assert task.task_id  # UUID generated
        assert task.session_id == "session-123"
        assert task.description == "Implement user authentication"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL.value
        assert task.created_at <= datetime.now()

    def test_create_with_priority_enum(self):
        """Task.create should accept TaskPriority enum."""
        task = Task.create(
            session_id="session-123",
            description="Fix critical bug",
            priority=TaskPriority.URGENT,
        )
        assert task.priority == TaskPriority.URGENT.value

    def test_create_with_priority_int(self):
        """Task.create should accept priority as int."""
        task = Task.create(
            session_id="session-123",
            description="Refactor code",
            priority=15,
        )
        assert task.priority == 15

    def test_create_with_metadata(self):
        """Task.create should accept metadata."""
        task = Task.create(
            session_id="session-123",
            description="Build API endpoint",
            metadata={"endpoint": "/api/users", "method": "POST"},
        )
        assert task.metadata["endpoint"] == "/api/users"
        assert task.metadata["method"] == "POST"

    def test_create_with_dependencies(self):
        """Task.create should accept task dependencies."""
        task = Task.create(
            session_id="session-123",
            description="Write tests",
            depends_on=["task-1", "task-2"],
        )
        assert task.depends_on == ["task-1", "task-2"]

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        task = Task.create(
            session_id="session-123",
            description="Test task",
            priority=TaskPriority.HIGH,
            metadata={"key": "value"},
            depends_on=["dep-1"],
        )
        data = task.to_dict()

        assert data["task_id"] == task.task_id
        assert data["session_id"] == "session-123"
        assert data["description"] == "Test task"
        assert data["status"] == "pending"
        assert data["priority"] == TaskPriority.HIGH.value
        assert data["metadata"] == {"key": "value"}
        assert data["depends_on"] == ["dep-1"]
        assert data["started_at"] is None
        assert data["completed_at"] is None

    def test_from_dict(self):
        """from_dict should deserialize correctly."""
        data = {
            "task_id": "task-456",
            "session_id": "session-789",
            "description": "Implement feature",
            "status": "in_progress",
            "priority": 10,
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:05:00",
            "completed_at": None,
            "error_message": None,
            "metadata": {"feature": "auth"},
            "depends_on": [],
        }
        task = Task.from_dict(data)

        assert task.task_id == "task-456"
        assert task.session_id == "session-789"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == 10
        assert task.started_at == datetime(2024, 1, 15, 10, 5, 0)
        assert task.metadata["feature"] == "auth"

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing optional fields."""
        data = {
            "task_id": "task-123",
            "session_id": "session-456",
            "description": "Basic task",
            "created_at": "2024-01-15T10:00:00",
        }
        task = Task.from_dict(data)

        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL.value
        assert task.metadata == {}
        assert task.depends_on == []

    def test_roundtrip(self):
        """to_dict and from_dict should be reversible."""
        original = Task.create(
            session_id="session-123",
            description="Roundtrip test",
            priority=TaskPriority.HIGH,
            metadata={"test": True},
            depends_on=["task-a", "task-b"],
        )
        data = original.to_dict()
        restored = Task.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.session_id == original.session_id
        assert restored.description == original.description
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.metadata == original.metadata
        assert restored.depends_on == original.depends_on

    def test_with_update(self):
        """with_update should create new task with updates."""
        task = Task.create(
            session_id="session-123",
            description="Original",
        )
        updated = task.with_update(
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now(),
        )

        # Original unchanged
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None

        # New task updated
        assert updated.status == TaskStatus.IN_PROGRESS
        assert updated.started_at is not None
        assert updated.task_id == task.task_id  # Same ID

    def test_is_pending(self):
        """is_pending should return True for pending tasks."""
        task = Task.create(session_id="s1", description="t1")
        assert task.is_pending is True

        started = task.with_update(status=TaskStatus.IN_PROGRESS)
        assert started.is_pending is False

    def test_is_completed(self):
        """is_completed should return True for terminal states."""
        task = Task.create(session_id="s1", description="t1")
        assert task.is_completed is False

        completed = task.with_update(status=TaskStatus.COMPLETED)
        assert completed.is_completed is True

        failed = task.with_update(status=TaskStatus.FAILED)
        assert failed.is_completed is True

        cancelled = task.with_update(status=TaskStatus.CANCELLED)
        assert cancelled.is_completed is True

    def test_can_execute(self):
        """can_execute should return True only for pending tasks."""
        task = Task.create(session_id="s1", description="t1")
        assert task.can_execute is True

        started = task.with_update(status=TaskStatus.IN_PROGRESS)
        assert started.can_execute is False


class TestTaskQueueConfig:
    """Tests for TaskQueueConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = TaskQueueConfig()
        assert config.data_dir == Path.home() / ".perpetualcc" / "data"
        assert config.db_name == "perpetualcc.db"
        assert config.max_tasks_per_session == 0  # Unlimited
        assert config.auto_cleanup_completed is False
        assert config.completed_retention_days == 7

    def test_config_is_immutable(self):
        """Config should be immutable."""
        config = TaskQueueConfig()
        with pytest.raises(AttributeError):
            config.db_name = "other.db"

    def test_custom_config(self):
        """Config should accept custom values."""
        custom_dir = Path("/tmp/custom")
        config = TaskQueueConfig(
            data_dir=custom_dir,
            db_name="custom.db",
            max_tasks_per_session=100,
            auto_cleanup_completed=True,
            completed_retention_days=30,
        )
        assert config.data_dir == custom_dir
        assert config.db_name == "custom.db"
        assert config.max_tasks_per_session == 100

    def test_db_path_property(self):
        """db_path should return full database path."""
        config = TaskQueueConfig(data_dir=Path("/tmp/test"), db_name="test.db")
        assert config.db_path == Path("/tmp/test/test.db")


class TestTaskQueue:
    """Tests for TaskQueue class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def queue(self, temp_dir: Path) -> TaskQueue:
        """Create a task queue with temp directory."""
        config = TaskQueueConfig(data_dir=temp_dir)
        return TaskQueue(config)

    def test_init_creates_directory(self, temp_dir: Path):
        """Queue should create data directory on init."""
        nested_dir = temp_dir / "nested" / "data"
        config = TaskQueueConfig(data_dir=nested_dir)
        queue = TaskQueue(config)
        assert nested_dir.exists()
        assert (nested_dir / "perpetualcc.db").exists()

    def test_add_basic_task(self, queue: TaskQueue):
        """add() should create and persist a task."""
        task = queue.add(
            session_id="session-123",
            description="Implement user login",
        )

        assert task.task_id
        assert task.session_id == "session-123"
        assert task.description == "Implement user login"
        assert task.status == TaskStatus.PENDING

        # Verify persistence
        retrieved = queue.get(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    def test_add_with_priority(self, queue: TaskQueue):
        """add() should accept priority parameter."""
        task = queue.add(
            session_id="session-123",
            description="Critical fix",
            priority=TaskPriority.URGENT,
        )
        assert task.priority == TaskPriority.URGENT.value

    def test_add_with_metadata_and_deps(self, queue: TaskQueue):
        """add() should accept metadata and dependencies."""
        task = queue.add(
            session_id="session-123",
            description="Build endpoint",
            metadata={"path": "/api/users"},
            depends_on=["task-1", "task-2"],
        )
        assert task.metadata["path"] == "/api/users"
        assert task.depends_on == ["task-1", "task-2"]

    def test_get_existing_task(self, queue: TaskQueue):
        """get() should retrieve existing task."""
        created = queue.add(session_id="s1", description="Test")
        retrieved = queue.get(created.task_id)

        assert retrieved is not None
        assert retrieved.task_id == created.task_id
        assert retrieved.description == created.description

    def test_get_nonexistent_task(self, queue: TaskQueue):
        """get() should return None for nonexistent task."""
        result = queue.get("nonexistent-task-id")
        assert result is None

    def test_next_returns_highest_priority(self, queue: TaskQueue):
        """next() should return highest priority pending task."""
        session_id = "session-123"

        # Add tasks with different priorities
        low = queue.add(session_id, "Low priority", priority=TaskPriority.LOW)
        high = queue.add(session_id, "High priority", priority=TaskPriority.HIGH)
        normal = queue.add(session_id, "Normal priority", priority=TaskPriority.NORMAL)

        next_task = queue.next(session_id)
        assert next_task is not None
        assert next_task.task_id == high.task_id

    def test_next_respects_creation_order_for_same_priority(self, queue: TaskQueue):
        """next() should use creation order for same priority."""
        session_id = "session-123"

        # Add tasks with same priority
        first = queue.add(session_id, "First task")
        time.sleep(0.01)  # Ensure different timestamps
        second = queue.add(session_id, "Second task")

        next_task = queue.next(session_id)
        assert next_task is not None
        assert next_task.task_id == first.task_id

    def test_next_respects_dependencies(self, queue: TaskQueue):
        """next() should skip tasks with unsatisfied dependencies."""
        session_id = "session-123"

        # Create independent task
        independent = queue.add(session_id, "Independent", priority=TaskPriority.LOW)

        # Create dependent task with higher priority
        dependent = queue.add(
            session_id,
            "Dependent",
            priority=TaskPriority.HIGH,
            depends_on=[independent.task_id],
        )

        # Should return independent (lower priority but no deps)
        next_task = queue.next(session_id)
        assert next_task is not None
        assert next_task.task_id == independent.task_id

    def test_next_allows_satisfied_dependencies(self, queue: TaskQueue):
        """next() should return task when dependencies are completed."""
        session_id = "session-123"

        # Create and complete prerequisite
        prereq = queue.add(session_id, "Prerequisite")
        queue.start(prereq.task_id)
        queue.complete(prereq.task_id)

        # Create dependent task
        dependent = queue.add(
            session_id,
            "Dependent",
            depends_on=[prereq.task_id],
        )

        next_task = queue.next(session_id)
        assert next_task is not None
        assert next_task.task_id == dependent.task_id

    def test_next_empty_queue(self, queue: TaskQueue):
        """next() should return None for empty queue."""
        result = queue.next("nonexistent-session")
        assert result is None

    def test_next_all_completed(self, queue: TaskQueue):
        """next() should return None when all tasks completed."""
        session_id = "session-123"

        task = queue.add(session_id, "Only task")
        queue.start(task.task_id)
        queue.complete(task.task_id)

        result = queue.next(session_id)
        assert result is None

    def test_start_task(self, queue: TaskQueue):
        """start() should mark task as in progress."""
        task = queue.add("session-123", "Task to start")
        started = queue.start(task.task_id)

        assert started is not None
        assert started.status == TaskStatus.IN_PROGRESS
        assert started.started_at is not None

        # Verify persistence
        retrieved = queue.get(task.task_id)
        assert retrieved.status == TaskStatus.IN_PROGRESS

    def test_start_nonexistent_task(self, queue: TaskQueue):
        """start() should return None for nonexistent task."""
        result = queue.start("nonexistent")
        assert result is None

    def test_start_already_started_task(self, queue: TaskQueue):
        """start() should return None for non-pending task."""
        task = queue.add("session-123", "Task")
        queue.start(task.task_id)

        # Try to start again
        result = queue.start(task.task_id)
        assert result is None

    def test_complete_task_success(self, queue: TaskQueue):
        """complete() should mark task as completed."""
        task = queue.add("session-123", "Task to complete")
        queue.start(task.task_id)
        completed = queue.complete(task.task_id)

        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.error_message is None

    def test_complete_task_failure(self, queue: TaskQueue):
        """complete() should mark task as failed with error."""
        task = queue.add("session-123", "Task to fail")
        queue.start(task.task_id)
        failed = queue.complete(task.task_id, error_message="Build failed")

        assert failed is not None
        assert failed.status == TaskStatus.FAILED
        assert failed.completed_at is not None
        assert failed.error_message == "Build failed"

    def test_complete_nonexistent_task(self, queue: TaskQueue):
        """complete() should return None for nonexistent task."""
        result = queue.complete("nonexistent")
        assert result is None

    def test_cancel_pending_task(self, queue: TaskQueue):
        """cancel() should cancel a pending task."""
        task = queue.add("session-123", "Task to cancel")
        cancelled = queue.cancel(task.task_id)

        assert cancelled is not None
        assert cancelled.status == TaskStatus.CANCELLED
        assert cancelled.completed_at is not None

    def test_cancel_in_progress_task(self, queue: TaskQueue):
        """cancel() should cancel an in-progress task."""
        task = queue.add("session-123", "Running task")
        queue.start(task.task_id)
        cancelled = queue.cancel(task.task_id)

        assert cancelled is not None
        assert cancelled.status == TaskStatus.CANCELLED

    def test_cancel_completed_task(self, queue: TaskQueue):
        """cancel() should return None for completed task."""
        task = queue.add("session-123", "Completed task")
        queue.start(task.task_id)
        queue.complete(task.task_id)

        result = queue.cancel(task.task_id)
        assert result is None

    def test_update_priority(self, queue: TaskQueue):
        """update_priority() should change task priority."""
        task = queue.add("session-123", "Task", priority=TaskPriority.NORMAL)
        updated = queue.update_priority(task.task_id, TaskPriority.URGENT)

        assert updated is not None
        assert updated.priority == TaskPriority.URGENT.value

        # Verify persistence
        retrieved = queue.get(task.task_id)
        assert retrieved.priority == TaskPriority.URGENT.value

    def test_update_priority_nonexistent(self, queue: TaskQueue):
        """update_priority() should return None for nonexistent task."""
        result = queue.update_priority("nonexistent", TaskPriority.HIGH)
        assert result is None

    def test_list_tasks_all(self, queue: TaskQueue):
        """list_tasks() should return all tasks."""
        queue.add("session-1", "Task 1")
        queue.add("session-1", "Task 2")
        queue.add("session-2", "Task 3")

        tasks = queue.list_tasks()
        assert len(tasks) == 3

    def test_list_tasks_by_session(self, queue: TaskQueue):
        """list_tasks() should filter by session."""
        queue.add("session-1", "Task 1")
        queue.add("session-1", "Task 2")
        queue.add("session-2", "Task 3")

        tasks = queue.list_tasks(session_id="session-1")
        assert len(tasks) == 2
        assert all(t.session_id == "session-1" for t in tasks)

    def test_list_tasks_by_status(self, queue: TaskQueue):
        """list_tasks() should filter by status."""
        t1 = queue.add("session-1", "Pending")
        t2 = queue.add("session-1", "Running")
        queue.start(t2.task_id)

        pending = queue.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == t1.task_id

        in_progress = queue.list_tasks(status=TaskStatus.IN_PROGRESS)
        assert len(in_progress) == 1
        assert in_progress[0].task_id == t2.task_id

    def test_list_tasks_with_limit(self, queue: TaskQueue):
        """list_tasks() should respect limit parameter."""
        for i in range(10):
            queue.add("session-1", f"Task {i}")

        tasks = queue.list_tasks(limit=5)
        assert len(tasks) == 5

    def test_list_tasks_ordered_by_priority(self, queue: TaskQueue):
        """list_tasks() should return tasks ordered by priority."""
        queue.add("session-1", "Low", priority=TaskPriority.LOW)
        queue.add("session-1", "High", priority=TaskPriority.HIGH)
        queue.add("session-1", "Normal", priority=TaskPriority.NORMAL)

        tasks = queue.list_tasks(session_id="session-1")
        assert tasks[0].priority == TaskPriority.HIGH.value
        assert tasks[1].priority == TaskPriority.NORMAL.value
        assert tasks[2].priority == TaskPriority.LOW.value

    def test_count_all(self, queue: TaskQueue):
        """count() should return total task count."""
        queue.add("session-1", "Task 1")
        queue.add("session-2", "Task 2")

        assert queue.count() == 2

    def test_count_by_session(self, queue: TaskQueue):
        """count() should filter by session."""
        queue.add("session-1", "Task 1")
        queue.add("session-1", "Task 2")
        queue.add("session-2", "Task 3")

        assert queue.count(session_id="session-1") == 2
        assert queue.count(session_id="session-2") == 1

    def test_count_by_status(self, queue: TaskQueue):
        """count() should filter by status."""
        t1 = queue.add("session-1", "Pending 1")
        t2 = queue.add("session-1", "Pending 2")
        t3 = queue.add("session-1", "Running")
        queue.start(t3.task_id)

        assert queue.count(status=TaskStatus.PENDING) == 2
        assert queue.count(status=TaskStatus.IN_PROGRESS) == 1

    def test_clear_session(self, queue: TaskQueue):
        """clear_session() should remove all tasks for session."""
        queue.add("session-1", "Task 1")
        queue.add("session-1", "Task 2")
        queue.add("session-2", "Task 3")

        removed = queue.clear_session("session-1")
        assert removed == 2
        assert queue.count(session_id="session-1") == 0
        assert queue.count(session_id="session-2") == 1

    def test_clear_session_keep_completed(self, queue: TaskQueue):
        """clear_session() should optionally keep completed tasks."""
        t1 = queue.add("session-1", "Pending")
        t2 = queue.add("session-1", "Completed")
        queue.start(t2.task_id)
        queue.complete(t2.task_id)

        removed = queue.clear_session("session-1", keep_completed=True)
        assert removed == 1  # Only pending removed
        assert queue.count(session_id="session-1") == 1  # Completed remains

    def test_delete_task(self, queue: TaskQueue):
        """delete() should remove a specific task."""
        task = queue.add("session-1", "Task to delete")
        assert queue.delete(task.task_id) is True
        assert queue.get(task.task_id) is None

    def test_delete_nonexistent(self, queue: TaskQueue):
        """delete() should return False for nonexistent task."""
        result = queue.delete("nonexistent")
        assert result is False

    def test_get_statistics(self, queue: TaskQueue):
        """get_statistics() should return queue stats."""
        t1 = queue.add("session-1", "Pending")
        t2 = queue.add("session-1", "Running")
        t3 = queue.add("session-2", "Completed")
        queue.start(t2.task_id)
        queue.start(t3.task_id)
        queue.complete(t3.task_id)

        stats = queue.get_statistics()
        assert stats["total"] == 3
        assert stats["pending"] == 1
        assert stats["in_progress"] == 1
        assert stats["completed"] == 1
        assert stats["session_count"] == 2

    def test_get_statistics_by_session(self, queue: TaskQueue):
        """get_statistics() should filter by session."""
        queue.add("session-1", "Task 1")
        queue.add("session-1", "Task 2")
        queue.add("session-2", "Task 3")

        stats = queue.get_statistics(session_id="session-1")
        assert stats["total"] == 2
        assert stats["session_count"] == 1

    def test_reorder_session(self, queue: TaskQueue):
        """reorder_session() should update task priorities."""
        session_id = "session-123"
        t1 = queue.add(session_id, "Task 1")
        t2 = queue.add(session_id, "Task 2")
        t3 = queue.add(session_id, "Task 3")

        # Reorder: t3, t1, t2
        updated = queue.reorder_session(session_id, [t3.task_id, t1.task_id, t2.task_id])
        assert len(updated) == 3

        # Check new priorities
        assert queue.get(t3.task_id).priority == 100  # First
        assert queue.get(t1.task_id).priority == 99   # Second
        assert queue.get(t2.task_id).priority == 98   # Third

        # Verify order in next()
        next_task = queue.next(session_id)
        assert next_task.task_id == t3.task_id


class TestTaskQueueRealWorldScenarios:
    """Tests with real-world Claude Code task scenarios."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def queue(self, temp_dir: Path) -> TaskQueue:
        """Create a task queue with temp directory."""
        config = TaskQueueConfig(data_dir=temp_dir)
        return TaskQueue(config)

    def test_web_app_development_workflow(self, queue: TaskQueue):
        """Simulate a typical web application development workflow."""
        session_id = "webapp-project"

        # Add tasks in typical order for a web app feature
        setup = queue.add(
            session_id,
            "Set up project structure with TypeScript and Express",
            priority=TaskPriority.HIGH,
        )

        db_schema = queue.add(
            session_id,
            "Create database schema for user authentication",
            priority=TaskPriority.HIGH,
            depends_on=[setup.task_id],
        )

        api_routes = queue.add(
            session_id,
            "Implement REST API routes for /api/auth/*",
            priority=TaskPriority.NORMAL,
            depends_on=[db_schema.task_id],
            metadata={"endpoints": ["/login", "/register", "/logout"]},
        )

        frontend = queue.add(
            session_id,
            "Create React login and registration forms",
            priority=TaskPriority.NORMAL,
            depends_on=[api_routes.task_id],
        )

        tests = queue.add(
            session_id,
            "Write integration tests for authentication flow",
            priority=TaskPriority.NORMAL,
            depends_on=[api_routes.task_id, frontend.task_id],
        )

        docs = queue.add(
            session_id,
            "Update API documentation for auth endpoints",
            priority=TaskPriority.LOW,
            depends_on=[api_routes.task_id],
        )

        # Verify dependency chain execution order
        # First: setup (no deps)
        next_task = queue.next(session_id)
        assert next_task.task_id == setup.task_id
        queue.start(setup.task_id)
        queue.complete(setup.task_id)

        # Second: db_schema (setup complete)
        next_task = queue.next(session_id)
        assert next_task.task_id == db_schema.task_id
        queue.start(db_schema.task_id)
        queue.complete(db_schema.task_id)

        # Third: api_routes (db_schema complete)
        next_task = queue.next(session_id)
        assert next_task.task_id == api_routes.task_id
        queue.start(api_routes.task_id)
        queue.complete(api_routes.task_id)

        # Now both frontend and docs can run (api_routes complete)
        # frontend has higher priority than docs
        next_task = queue.next(session_id)
        assert next_task.task_id == frontend.task_id
        queue.start(frontend.task_id)
        queue.complete(frontend.task_id)

        # tests can now run (both deps complete)
        next_task = queue.next(session_id)
        assert next_task.task_id == tests.task_id
        queue.start(tests.task_id)
        queue.complete(tests.task_id)

        # Finally docs (lowest priority)
        next_task = queue.next(session_id)
        assert next_task.task_id == docs.task_id
        queue.start(docs.task_id)
        queue.complete(docs.task_id)

        # All tasks complete
        assert queue.next(session_id) is None
        stats = queue.get_statistics(session_id)
        assert stats["completed"] == 6

    def test_bug_fix_workflow_with_priority_escalation(self, queue: TaskQueue):
        """Simulate bug fix workflow with priority changes."""
        session_id = "bugfix-session"

        # Normal development tasks
        feature = queue.add(
            session_id,
            "Implement new dashboard widget",
            priority=TaskPriority.NORMAL,
        )

        refactor = queue.add(
            session_id,
            "Refactor authentication module",
            priority=TaskPriority.LOW,
        )

        # Start working on feature
        queue.start(feature.task_id)

        # Urgent bug comes in!
        hotfix = queue.add(
            session_id,
            "Fix critical XSS vulnerability in user input",
            priority=TaskPriority.URGENT,
            metadata={"severity": "critical", "cve": "CVE-2024-XXXX"},
        )

        # Feature is in progress, but hotfix should be visible
        pending = queue.list_tasks(session_id, status=TaskStatus.PENDING)
        assert len(pending) == 2

        # Highest priority pending is the hotfix
        assert pending[0].task_id == hotfix.task_id

        # Cancel feature work to focus on hotfix
        queue.cancel(feature.task_id)

        # Now hotfix is next
        next_task = queue.next(session_id)
        assert next_task.task_id == hotfix.task_id

    def test_ci_cd_pipeline_simulation(self, queue: TaskQueue):
        """Simulate a CI/CD pipeline with build stages."""
        session_id = "ci-pipeline"

        # Lint stage
        lint = queue.add(
            session_id,
            "Run ESLint and Prettier checks",
            priority=TaskPriority.HIGH,
            metadata={"stage": "lint", "parallel": True},
        )

        type_check = queue.add(
            session_id,
            "Run TypeScript type checking",
            priority=TaskPriority.HIGH,
            metadata={"stage": "lint", "parallel": True},
        )

        # Test stage (depends on lint)
        unit_tests = queue.add(
            session_id,
            "Run unit tests with Jest",
            priority=TaskPriority.NORMAL,
            depends_on=[lint.task_id, type_check.task_id],
            metadata={"stage": "test"},
        )

        integration_tests = queue.add(
            session_id,
            "Run integration tests",
            priority=TaskPriority.NORMAL,
            depends_on=[lint.task_id, type_check.task_id],
            metadata={"stage": "test"},
        )

        # Build stage (depends on tests)
        build = queue.add(
            session_id,
            "Build production bundle",
            priority=TaskPriority.NORMAL,
            depends_on=[unit_tests.task_id, integration_tests.task_id],
            metadata={"stage": "build"},
        )

        # Deploy stage (depends on build)
        deploy = queue.add(
            session_id,
            "Deploy to staging environment",
            priority=TaskPriority.LOW,
            depends_on=[build.task_id],
            metadata={"stage": "deploy", "environment": "staging"},
        )

        # Simulate pipeline execution
        # Lint and type_check can run in parallel (both are first)
        first_tasks = [queue.next(session_id)]
        queue.start(first_tasks[0].task_id)

        second_task = queue.next(session_id)
        assert second_task is not None  # Second parallel task available
        first_tasks.append(second_task)
        queue.start(second_task.task_id)

        # Complete lint stage
        for t in first_tasks:
            queue.complete(t.task_id)

        # Test stage now available
        test_tasks = []
        for _ in range(2):
            t = queue.next(session_id)
            if t:
                test_tasks.append(t)
                queue.start(t.task_id)

        assert len(test_tasks) == 2

        # Fail one test
        queue.complete(test_tasks[0].task_id)
        queue.complete(test_tasks[1].task_id, error_message="Integration test failed")

        # Build should still be blocked because integration tests failed
        # In a real scenario, you'd check if all deps succeeded
        # Here we just verify the pipeline state
        stats = queue.get_statistics(session_id)
        assert stats["completed"] == 3
        assert stats["failed"] == 1
        assert stats["pending"] == 2  # build and deploy still pending

    def test_refactoring_with_rollback(self, queue: TaskQueue):
        """Simulate refactoring workflow with potential rollback."""
        session_id = "refactor-session"

        # Add refactoring tasks
        backup = queue.add(
            session_id,
            "Create backup branch for rollback",
            priority=TaskPriority.HIGH,
        )

        refactor = queue.add(
            session_id,
            "Refactor user service to use repository pattern",
            depends_on=[backup.task_id],
        )

        tests = queue.add(
            session_id,
            "Update and run all affected tests",
            depends_on=[refactor.task_id],
        )

        # Start workflow
        queue.start(backup.task_id)
        queue.complete(backup.task_id)

        queue.start(refactor.task_id)
        queue.complete(refactor.task_id)

        queue.start(tests.task_id)

        # Tests fail - need rollback
        queue.complete(tests.task_id, error_message="5 tests failed after refactoring")

        # Add rollback task with highest priority
        rollback = queue.add(
            session_id,
            "Rollback to backup branch due to test failures",
            priority=TaskPriority.URGENT,
            metadata={"reason": "test_failures", "failed_tests": 5},
        )

        next_task = queue.next(session_id)
        assert next_task.task_id == rollback.task_id

    def test_multi_session_isolation(self, queue: TaskQueue):
        """Test that multiple sessions don't interfere with each other."""
        # Two different projects being worked on
        project_a = "ecommerce-app"
        project_b = "mobile-api"

        # Add tasks to both projects
        a1 = queue.add(project_a, "Build shopping cart")
        a2 = queue.add(project_a, "Add payment integration")

        b1 = queue.add(project_b, "Create REST endpoints")
        b2 = queue.add(project_b, "Add authentication")

        # Get next for project A
        next_a = queue.next(project_a)
        assert next_a.session_id == project_a

        # Get next for project B
        next_b = queue.next(project_b)
        assert next_b.session_id == project_b

        # Complete one from each
        queue.start(a1.task_id)
        queue.complete(a1.task_id)

        queue.start(b1.task_id)
        queue.complete(b1.task_id)

        # Stats should be separate
        stats_a = queue.get_statistics(project_a)
        stats_b = queue.get_statistics(project_b)

        assert stats_a["completed"] == 1
        assert stats_a["pending"] == 1
        assert stats_b["completed"] == 1
        assert stats_b["pending"] == 1

        # Clearing one shouldn't affect the other
        queue.clear_session(project_a)
        assert queue.count(session_id=project_a) == 0
        assert queue.count(session_id=project_b) == 2


class TestTaskQueuePersistence:
    """Tests for database persistence and recovery."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_persistence_across_instances(self, temp_dir: Path):
        """Data should persist across queue instances."""
        config = TaskQueueConfig(data_dir=temp_dir)

        # Create queue and add tasks
        queue1 = TaskQueue(config)
        task = queue1.add("session-123", "Persistent task")
        task_id = task.task_id

        # Create new queue instance
        queue2 = TaskQueue(config)
        retrieved = queue2.get(task_id)

        assert retrieved is not None
        assert retrieved.description == "Persistent task"

    def test_recovery_after_crash(self, temp_dir: Path):
        """Queue should recover task state after simulated crash."""
        config = TaskQueueConfig(data_dir=temp_dir)

        # Start task in first instance
        queue1 = TaskQueue(config)
        task = queue1.add("session-123", "Long running task")
        queue1.start(task.task_id)

        # Simulate crash - create new instance
        queue2 = TaskQueue(config)
        recovered = queue2.get(task.task_id)

        # Task should still be in_progress
        assert recovered is not None
        assert recovered.status == TaskStatus.IN_PROGRESS
        assert recovered.started_at is not None

    def test_concurrent_access_safety(self, temp_dir: Path):
        """Multiple instances should handle concurrent access."""
        config = TaskQueueConfig(data_dir=temp_dir)

        queue1 = TaskQueue(config)
        queue2 = TaskQueue(config)

        # Add from different instances
        t1 = queue1.add("session-1", "Task from queue1")
        t2 = queue2.add("session-1", "Task from queue2")

        # Both should be visible from either instance
        assert queue1.count() == 2
        assert queue2.count() == 2

        # Complete from queue1
        queue1.start(t1.task_id)
        queue1.complete(t1.task_id)

        # queue2 should see the update
        retrieved = queue2.get(t1.task_id)
        assert retrieved.status == TaskStatus.COMPLETED
