"""Unit tests for memory store (SQLite persistence).

Tests cover:
- Database initialization and schema creation
- Episode CRUD operations
- Procedure CRUD operations with confidence updates
- Fact CRUD operations
- Statistics and queries
- Real-world Claude Code scenarios
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from perpetualcc.memory.store import (
    MemoryStore,
    MemoryStoreConfig,
    StoredEpisode,
    StoredFact,
    StoredProcedure,
)


class TestMemoryStoreConfig:
    """Tests for MemoryStoreConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = MemoryStoreConfig()

        assert config.pool_size == 5
        assert config.busy_timeout == 5000
        assert config.journal_mode == "WAL"
        assert "perpetualcc" in str(config.db_path)

    def test_custom_path(self, tmp_path: Path):
        """Config should accept custom database path."""
        db_path = tmp_path / "custom.db"
        config = MemoryStoreConfig(db_path=db_path)

        assert config.db_path == db_path

    def test_immutable(self):
        """Config should be immutable."""
        config = MemoryStoreConfig()
        with pytest.raises(AttributeError):
            config.pool_size = 10


class TestMemoryStoreInitialization:
    """Tests for MemoryStore initialization."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> Path:
        """Create a temporary database path."""
        return tmp_path / "test.db"

    @pytest.fixture
    async def store(self, temp_db_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=temp_db_path)
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, temp_db_path: Path):
        """initialize() should create the database file."""
        config = MemoryStoreConfig(db_path=temp_db_path)
        store = MemoryStore(config)

        assert not temp_db_path.exists()
        await store.initialize()
        assert temp_db_path.exists()

        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, store: MemoryStore):
        """initialize() should create all required tables."""
        # Tables exist if we can query them
        episodes = await store.query_episodes(limit=1)
        procedures = await store.query_procedures(limit=1)
        facts = await store.query_facts(limit=1)

        assert isinstance(episodes, list)
        assert isinstance(procedures, list)
        assert isinstance(facts, list)

    @pytest.mark.asyncio
    async def test_double_initialize_safe(self, store: MemoryStore):
        """Calling initialize() twice should be safe."""
        # Already initialized in fixture
        await store.initialize()  # Should not raise

        # Should still work
        episodes = await store.query_episodes(limit=1)
        assert isinstance(episodes, list)

    @pytest.mark.asyncio
    async def test_operations_without_initialize_fail(self, temp_db_path: Path):
        """Operations before initialize() should raise."""
        config = MemoryStoreConfig(db_path=temp_db_path)
        store = MemoryStore(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await store.query_episodes()

    @pytest.mark.asyncio
    async def test_close_and_reopen(self, temp_db_path: Path):
        """Should be able to close and reopen store."""
        config = MemoryStoreConfig(db_path=temp_db_path)

        # First connection - insert data
        store1 = MemoryStore(config)
        await store1.initialize()
        episode_id = await store1.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="test",
            context="test context",
            action_taken="test",
            action_reason="test",
        )
        await store1.close()

        # Second connection - verify data
        store2 = MemoryStore(config)
        await store2.initialize()
        episode = await store2.get_episode_by_id(episode_id)
        assert episode is not None
        assert episode.session_id == "test"
        await store2.close()


class TestEpisodeOperations:
    """Tests for episode CRUD operations."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_episode_basic(self, store: MemoryStore):
        """Should insert an episode and return its ID."""
        now = datetime.now()
        episode_id = await store.insert_episode(
            session_id="session-123",
            timestamp=now,
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation in src/",
        )

        assert episode_id > 0

        # Verify it was inserted
        episode = await store.get_episode_by_id(episode_id)
        assert episode is not None
        assert episode.session_id == "session-123"
        assert episode.event_type == "permission_request"
        assert episode.action_taken == "approve_tool"

    @pytest.mark.asyncio
    async def test_insert_episode_with_metadata(self, store: MemoryStore):
        """Should insert episode with metadata."""
        metadata = {
            "tool_name": "Write",
            "file_path": "/src/main.py",
            "risk_level": "low",
        }
        episode_id = await store.insert_episode(
            session_id="session-123",
            timestamp=datetime.now(),
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
            metadata=metadata,
        )

        episode = await store.get_episode_by_id(episode_id)
        assert episode.metadata == metadata

    @pytest.mark.asyncio
    async def test_insert_episode_with_embedding(self, store: MemoryStore):
        """Should insert episode with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        episode_id = await store.insert_episode(
            session_id="session-123",
            timestamp=datetime.now(),
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
            embedding=embedding,
        )

        episode = await store.get_episode_by_id(episode_id)
        assert episode.embedding == embedding

    @pytest.mark.asyncio
    async def test_query_episodes_by_session(self, store: MemoryStore):
        """Should query episodes by session ID."""
        # Insert episodes for different sessions
        await store.insert_episode(
            session_id="session-a",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
        )
        await store.insert_episode(
            session_id="session-b",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
        )
        await store.insert_episode(
            session_id="session-a",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
        )

        episodes = await store.query_episodes(session_id="session-a")
        assert len(episodes) == 2
        assert all(e.session_id == "session-a" for e in episodes)

    @pytest.mark.asyncio
    async def test_query_episodes_by_event_type(self, store: MemoryStore):
        """Should query episodes by event type."""
        await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="permission_request",
            context="test",
            action_taken="test",
            action_reason="test",
        )
        await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="question",
            context="test",
            action_taken="test",
            action_reason="test",
        )

        permission_episodes = await store.query_episodes(event_type="permission_request")
        question_episodes = await store.query_episodes(event_type="question")

        assert len(permission_episodes) == 1
        assert len(question_episodes) == 1

    @pytest.mark.asyncio
    async def test_query_episodes_by_outcome(self, store: MemoryStore):
        """Should query episodes by outcome."""
        await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
            outcome="success",
        )
        await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
            outcome="failure",
        )

        success_episodes = await store.query_episodes(outcome="success")
        failure_episodes = await store.query_episodes(outcome="failure")

        assert len(success_episodes) == 1
        assert len(failure_episodes) == 1

    @pytest.mark.asyncio
    async def test_update_episode_outcome(self, store: MemoryStore):
        """Should update episode outcome."""
        episode_id = await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
            outcome="pending",
        )

        result = await store.update_episode_outcome(episode_id, "success")
        assert result is True

        episode = await store.get_episode_by_id(episode_id)
        assert episode.outcome == "success"

    @pytest.mark.asyncio
    async def test_update_episode_embedding(self, store: MemoryStore):
        """Should update episode embedding."""
        episode_id = await store.insert_episode(
            session_id="test",
            timestamp=datetime.now(),
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
        )

        embedding = [0.1, 0.2, 0.3]
        result = await store.update_episode_embedding(episode_id, embedding)
        assert result is True

        episode = await store.get_episode_by_id(episode_id)
        assert episode.embedding == embedding

    @pytest.mark.asyncio
    async def test_get_recent_episodes(self, store: MemoryStore):
        """Should get most recent episodes."""
        # Insert episodes with different timestamps
        base_time = datetime.now()
        for i in range(10):
            await store.insert_episode(
                session_id="test",
                timestamp=base_time - timedelta(hours=i),
                event_type="test",
                context=f"context-{i}",
                action_taken="test",
                action_reason="test",
            )

        recent = await store.get_recent_episodes(limit=3)
        assert len(recent) == 3
        # Most recent first (context-0 has latest timestamp)
        assert recent[0].context == "context-0"


class TestProcedureOperations:
    """Tests for procedure CRUD operations."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_procedure(self, store: MemoryStore):
        """Should insert a procedure."""
        proc_id = await store.insert_procedure(
            trigger_type="tool_use",
            trigger_pattern="Read:*",
            action="approve",
            confidence=0.5,
        )

        assert proc_id > 0

        proc = await store.get_procedure("tool_use", "Read:*")
        assert proc is not None
        assert proc.trigger_type == "tool_use"
        assert proc.action == "approve"
        assert proc.confidence == 0.5

    @pytest.mark.asyncio
    async def test_query_procedures_by_type(self, store: MemoryStore):
        """Should query procedures by trigger type."""
        await store.insert_procedure("tool_use", "Read:*", "approve", 0.5)
        await store.insert_procedure("tool_use", "Write:*", "approve", 0.6)
        await store.insert_procedure("question", "proceed?", "answer_yes", 0.7)

        tool_procs = await store.query_procedures(trigger_type="tool_use")
        question_procs = await store.query_procedures(trigger_type="question")

        assert len(tool_procs) == 2
        assert len(question_procs) == 1

    @pytest.mark.asyncio
    async def test_query_procedures_by_confidence(self, store: MemoryStore):
        """Should query procedures with minimum confidence."""
        await store.insert_procedure("test", "pattern1", "action1", 0.3)
        await store.insert_procedure("test", "pattern2", "action2", 0.6)
        await store.insert_procedure("test", "pattern3", "action3", 0.9)

        high_conf = await store.query_procedures(min_confidence=0.7)
        assert len(high_conf) == 1
        assert high_conf[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_update_procedure_outcome_success(self, store: MemoryStore):
        """Successful outcome should increase confidence."""
        proc_id = await store.insert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="action",
            confidence=0.5,
        )

        updated = await store.update_procedure_outcome(proc_id, success=True)

        assert updated is not None
        assert updated.confidence == 0.55  # 0.5 + 0.05
        assert updated.success_count == 1
        assert updated.failure_count == 0

    @pytest.mark.asyncio
    async def test_update_procedure_outcome_failure(self, store: MemoryStore):
        """Failed outcome should decrease confidence."""
        proc_id = await store.insert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="action",
            confidence=0.5,
        )

        updated = await store.update_procedure_outcome(proc_id, success=False)

        assert updated is not None
        assert updated.confidence == 0.4  # 0.5 - 0.1
        assert updated.success_count == 0
        assert updated.failure_count == 1

    @pytest.mark.asyncio
    async def test_confidence_caps_at_max(self, store: MemoryStore):
        """Confidence should cap at 0.99."""
        proc_id = await store.insert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="action",
            confidence=0.96,
        )

        # Multiple successes
        for _ in range(5):
            await store.update_procedure_outcome(proc_id, success=True)

        proc = await store.get_procedure("test", "pattern")
        assert proc.confidence == 0.99

    @pytest.mark.asyncio
    async def test_confidence_floors_at_min(self, store: MemoryStore):
        """Confidence should floor at 0.1."""
        proc_id = await store.insert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="action",
            confidence=0.2,
        )

        # Multiple failures
        for _ in range(5):
            await store.update_procedure_outcome(proc_id, success=False)

        proc = await store.get_procedure("test", "pattern")
        assert proc.confidence == 0.1

    @pytest.mark.asyncio
    async def test_upsert_procedure_insert(self, store: MemoryStore):
        """upsert_procedure should insert new procedure."""
        proc_id = await store.upsert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="action",
            confidence=0.5,
        )

        assert proc_id > 0
        proc = await store.get_procedure("test", "pattern")
        assert proc is not None

    @pytest.mark.asyncio
    async def test_upsert_procedure_update(self, store: MemoryStore):
        """upsert_procedure should update existing procedure."""
        await store.insert_procedure("test", "pattern", "old_action", 0.5)

        proc_id = await store.upsert_procedure(
            trigger_type="test",
            trigger_pattern="pattern",
            action="new_action",
            confidence=0.8,
        )

        proc = await store.get_procedure("test", "pattern")
        assert proc.action == "new_action"
        assert proc.confidence == 0.8


class TestFactOperations:
    """Tests for fact CRUD operations."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_fact(self, store: MemoryStore):
        """Should insert a fact."""
        fact_id = await store.insert_fact(
            project_path="/path/to/project",
            category="convention",
            fact="Uses pytest for testing",
            source="CLAUDE.md",
            confidence=1.0,
        )

        assert fact_id > 0

        fact = await store.get_fact_by_id(fact_id)
        assert fact is not None
        assert fact.project_path == "/path/to/project"
        assert fact.category == "convention"
        assert fact.fact == "Uses pytest for testing"

    @pytest.mark.asyncio
    async def test_query_facts_by_project(self, store: MemoryStore):
        """Should query facts by project path."""
        await store.insert_fact("/project/a", "convention", "fact1")
        await store.insert_fact("/project/b", "convention", "fact2")
        await store.insert_fact("/project/a", "convention", "fact3")

        facts_a = await store.query_facts(project_path="/project/a")
        facts_b = await store.query_facts(project_path="/project/b")

        assert len(facts_a) == 2
        assert len(facts_b) == 1

    @pytest.mark.asyncio
    async def test_query_facts_by_category(self, store: MemoryStore):
        """Should query facts by category."""
        await store.insert_fact("/project", "convention", "fact1")
        await store.insert_fact("/project", "architecture", "fact2")
        await store.insert_fact("/project", "convention", "fact3")

        conventions = await store.query_facts(
            project_path="/project", category="convention"
        )
        architecture = await store.query_facts(
            project_path="/project", category="architecture"
        )

        assert len(conventions) == 2
        assert len(architecture) == 1

    @pytest.mark.asyncio
    async def test_update_fact(self, store: MemoryStore):
        """Should update a fact."""
        fact_id = await store.insert_fact("/project", "convention", "old fact")

        result = await store.update_fact(fact_id, "new fact", confidence=0.8)

        assert result is True
        fact = await store.get_fact_by_id(fact_id)
        assert fact.fact == "new fact"
        assert fact.confidence == 0.8

    @pytest.mark.asyncio
    async def test_delete_fact(self, store: MemoryStore):
        """Should delete a fact."""
        fact_id = await store.insert_fact("/project", "convention", "fact")

        result = await store.delete_fact(fact_id)

        assert result is True
        fact = await store.get_fact_by_id(fact_id)
        assert fact is None


class TestStatistics:
    """Tests for statistics operations."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_statistics_empty(self, store: MemoryStore):
        """Statistics should work with empty database."""
        stats = await store.get_statistics()

        assert stats["total_episodes"] == 0
        assert stats["total_procedures"] == 0
        assert stats["total_facts"] == 0

    @pytest.mark.asyncio
    async def test_statistics_with_data(self, store: MemoryStore):
        """Statistics should reflect stored data."""
        # Add episodes
        await store.insert_episode(
            "sess", datetime.now(), "test", "ctx", "act", "reason", outcome="success"
        )
        await store.insert_episode(
            "sess", datetime.now(), "test", "ctx", "act", "reason", outcome="failure"
        )

        # Add procedures
        await store.insert_procedure("type1", "pattern1", "action", 0.5)
        await store.insert_procedure("type1", "pattern2", "action", 0.8)

        # Add facts
        await store.insert_fact("/proj", "convention", "fact1")
        await store.insert_fact("/proj", "architecture", "fact2")

        stats = await store.get_statistics()

        assert stats["total_episodes"] == 2
        assert stats["episodes_by_outcome"]["success"] == 1
        assert stats["episodes_by_outcome"]["failure"] == 1
        assert stats["total_procedures"] == 2
        assert stats["avg_procedure_confidence"] == 0.65  # (0.5 + 0.8) / 2
        assert stats["total_facts"] == 2


class TestTransaction:
    """Tests for transaction support."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_transaction_commit(self, store: MemoryStore):
        """Transaction should commit on success."""
        async with store.transaction():
            await store.insert_episode(
                "sess", datetime.now(), "test", "ctx", "act", "reason"
            )
            await store.insert_fact("/proj", "convention", "fact")

        episodes = await store.query_episodes()
        facts = await store.query_facts()

        assert len(episodes) == 1
        assert len(facts) == 1


class TestRealWorldScenarios:
    """Tests simulating real-world Claude Code scenarios."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_scenario_tool_permission_learning(self, store: MemoryStore):
        """Scenario: Learning from repeated tool approvals."""
        # User approves Read tool multiple times
        for i in range(5):
            await store.insert_episode(
                session_id=f"session-{i}",
                timestamp=datetime.now(),
                event_type="permission_request",
                context="Tool use request: Read",
                action_taken="approve_tool",
                action_reason="Safe read operation",
                outcome="success",
                confidence=0.95,
                metadata={"tool_name": "Read", "file_path": f"/src/file{i}.py"},
            )

        # Query successful Read approvals
        episodes = await store.query_episodes(
            event_type="permission_request", outcome="success"
        )
        read_approvals = [e for e in episodes if e.metadata.get("tool_name") == "Read"]

        assert len(read_approvals) == 5
        # All had high confidence
        assert all(e.confidence > 0.9 for e in read_approvals)

    @pytest.mark.asyncio
    async def test_scenario_question_answering_patterns(self, store: MemoryStore):
        """Scenario: Learning patterns from answered questions."""
        # Create procedure for common question
        proc_id = await store.insert_procedure(
            trigger_type="question",
            trigger_pattern="Should I proceed?",
            action="answer_yes",
            confidence=0.5,
        )

        # Simulate successful uses
        for _ in range(5):
            await store.update_procedure_outcome(proc_id, success=True)

        proc = await store.get_procedure("question", "Should I proceed?")

        # Confidence should have increased
        assert proc.confidence >= 0.7
        assert proc.success_count == 5

    @pytest.mark.asyncio
    async def test_scenario_project_conventions(self, store: MemoryStore):
        """Scenario: Storing project-specific conventions."""
        project = "/Users/dev/myproject"

        # Store conventions learned from CLAUDE.md
        await store.insert_fact(
            project_path=project,
            category="convention",
            fact="Uses pytest with 80% coverage requirement",
            source="CLAUDE.md",
        )
        await store.insert_fact(
            project_path=project,
            category="convention",
            fact="All files must have type hints",
            source="CLAUDE.md",
        )
        await store.insert_fact(
            project_path=project,
            category="architecture",
            fact="Uses layered architecture: core, api, cli",
            source="ARCHITECTURE.md",
        )

        conventions = await store.query_facts(
            project_path=project, category="convention"
        )
        architecture = await store.query_facts(
            project_path=project, category="architecture"
        )

        assert len(conventions) == 2
        assert len(architecture) == 1

    @pytest.mark.asyncio
    async def test_scenario_dangerous_command_blocking(self, store: MemoryStore):
        """Scenario: Learning to block dangerous commands."""
        # Record a dangerous command that was escalated
        await store.insert_episode(
            session_id="session-1",
            timestamp=datetime.now(),
            event_type="permission_request",
            context="Tool use request: Bash(rm -rf /)",
            action_taken="deny_tool",
            action_reason="Dangerous destructive command",
            outcome="success",  # Success because we correctly blocked it
            confidence=0.99,
            metadata={"tool_name": "Bash", "command": "rm -rf /"},
        )

        # Create procedure to always block this
        await store.insert_procedure(
            trigger_type="bash_command",
            trigger_pattern="rm -rf *",
            action="deny",
            confidence=0.99,
        )

        # Query for high-confidence deny procedures
        deny_procs = await store.query_procedures(min_confidence=0.9)
        rm_procs = [p for p in deny_procs if "rm -rf" in p.trigger_pattern]

        assert len(rm_procs) == 1
        assert rm_procs[0].action == "deny"

    @pytest.mark.asyncio
    async def test_scenario_git_workflow_approval(self, store: MemoryStore):
        """Scenario: Learning to approve safe git commands."""
        safe_git_commands = [
            "git status",
            "git diff",
            "git log --oneline",
            "git branch -a",
            "git add .",
        ]

        # Record successful approvals
        for cmd in safe_git_commands:
            await store.insert_episode(
                session_id="session-git",
                timestamp=datetime.now(),
                event_type="permission_request",
                context=f"Tool use request: Bash({cmd})",
                action_taken="approve_tool",
                action_reason="Safe git read-only or stage command",
                outcome="success",
                metadata={"tool_name": "Bash", "command": cmd},
            )

        # Verify all stored
        episodes = await store.query_episodes(session_id="session-git")
        assert len(episodes) == 5
        assert all(e.action_taken == "approve_tool" for e in episodes)

    @pytest.mark.asyncio
    async def test_scenario_confidence_evolution(self, store: MemoryStore):
        """Scenario: Confidence evolves based on outcomes."""
        # Start with neutral confidence
        proc_id = await store.insert_procedure(
            trigger_type="tool_use",
            trigger_pattern="Write:src/*.py",
            action="approve",
            confidence=0.5,
        )

        # Mix of successes and failures
        outcomes = [True, True, False, True, True, True, False, True]

        for success in outcomes:
            await store.update_procedure_outcome(proc_id, success)

        proc = await store.get_procedure("tool_use", "Write:src/*.py")

        # 6 successes (+0.3), 2 failures (-0.2) = 0.5 + 0.1 = 0.6
        # But with caps, should be around 0.6
        assert 0.5 <= proc.confidence <= 0.7
        assert proc.success_count == 6
        assert proc.failure_count == 2

    @pytest.mark.asyncio
    async def test_clear_all(self, store: MemoryStore):
        """clear_all should remove all data."""
        # Add data
        await store.insert_episode(
            "sess", datetime.now(), "test", "ctx", "act", "reason"
        )
        await store.insert_procedure("type", "pattern", "action", 0.5)
        await store.insert_fact("/proj", "cat", "fact")

        # Verify data exists
        stats_before = await store.get_statistics()
        assert stats_before["total_episodes"] == 1

        # Clear all
        await store.clear_all()

        # Verify empty
        stats_after = await store.get_statistics()
        assert stats_after["total_episodes"] == 0
        assert stats_after["total_procedures"] == 0
        assert stats_after["total_facts"] == 0
