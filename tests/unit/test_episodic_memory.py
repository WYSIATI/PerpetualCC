"""Unit tests for episodic memory.

Tests cover:
- Episode recording
- Similarity search (with and without embeddings)
- Outcome tracking
- Statistics
- Real-world Claude Code scenarios
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from perpetualcc.memory.episodic import (
    Episode,
    EpisodicMemory,
    EpisodicMemoryConfig,
    SimilarEpisode,
)
from perpetualcc.memory.store import MemoryStore, MemoryStoreConfig


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_basic(self):
        """Episode should store event data."""
        episode = Episode(
            timestamp=datetime.now(),
            session_id="session-123",
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
        )

        assert episode.session_id == "session-123"
        assert episode.event_type == "permission_request"
        assert episode.action_taken == "approve_tool"
        assert episode.outcome == "pending"  # Default
        assert episode.confidence == 1.0  # Default

    def test_episode_with_metadata(self):
        """Episode should store metadata."""
        metadata = {"tool_name": "Write", "file_path": "/src/main.py"}
        episode = Episode(
            timestamp=datetime.now(),
            session_id="session-123",
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
            metadata=metadata,
        )

        assert episode.metadata == metadata

    def test_episode_with_embedding(self):
        """Episode should store embedding."""
        embedding = [0.1, 0.2, 0.3]
        episode = Episode(
            timestamp=datetime.now(),
            session_id="session-123",
            event_type="test",
            context="test",
            action_taken="test",
            action_reason="test",
            embedding=embedding,
        )

        assert episode.embedding == embedding


class TestEpisodicMemoryConfig:
    """Tests for EpisodicMemoryConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = EpisodicMemoryConfig()

        assert config.max_episodes == 10000
        assert config.similarity_threshold == 0.5
        assert config.default_top_k == 5
        assert config.auto_embed is True

    def test_custom_values(self):
        """Config should accept custom values."""
        config = EpisodicMemoryConfig(
            max_episodes=5000,
            similarity_threshold=0.7,
            default_top_k=10,
            auto_embed=False,
        )

        assert config.max_episodes == 5000
        assert config.similarity_threshold == 0.7
        assert config.auto_embed is False


class TestEpisodicMemoryRecording:
    """Tests for episode recording."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory without embedding provider."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_record_episode(self, episodic_memory: EpisodicMemory):
        """Should record an episode."""
        episode = Episode(
            timestamp=datetime.now(),
            session_id="session-123",
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
        )

        episode_id = await episodic_memory.record(episode)

        assert episode_id > 0

        # Verify recorded
        recent = await episodic_memory.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].session_id == "session-123"

    @pytest.mark.asyncio
    async def test_record_from_master_agent(self, episodic_memory: EpisodicMemory):
        """Should record episode from MasterAgent-style parameters."""
        episode_id = await episodic_memory.record_from_master_agent(
            timestamp=datetime.now(),
            session_id="session-456",
            event_type="question",
            context="Question: Should I proceed?",
            action_taken="answer",
            action_reason="Confirmation pattern matched",
            outcome="success",
            confidence=0.85,
            metadata={"question": "Should I proceed?", "answer": "Yes"},
        )

        assert episode_id > 0

        recent = await episodic_memory.get_recent(limit=1)
        assert recent[0].event_type == "question"
        assert recent[0].outcome == "success"

    @pytest.mark.asyncio
    async def test_record_with_auto_embed(
        self, store: MemoryStore
    ):
        """Should auto-generate embedding when provider available."""
        # Create mock embedding provider
        mock_provider = AsyncMock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]

        memory = EpisodicMemory(store, embedding_provider=mock_provider)

        episode = Episode(
            timestamp=datetime.now(),
            session_id="session-123",
            event_type="test",
            context="Test context",
            action_taken="test",
            action_reason="test",
        )

        await memory.record(episode)

        # Verify embedding was generated
        mock_provider.embed.assert_called_once()


class TestEpisodicMemorySimilarity:
    """Tests for similarity search."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_find_similar_by_keywords(self, episodic_memory: EpisodicMemory):
        """Should find similar episodes by keyword matching."""
        # Record episodes
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s1",
                event_type="permission_request",
                context="Tool use request: Write file to src/main.py",
                action_taken="approve_tool",
                action_reason="Safe write",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s2",
                event_type="permission_request",
                context="Tool use request: Read file from config.json",
                action_taken="approve_tool",
                action_reason="Safe read",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s3",
                event_type="permission_request",
                context="Tool use request: Write file to tests/test_main.py",
                action_taken="approve_tool",
                action_reason="Safe write",
            )
        )

        # Search for Write operations
        similar = await episodic_memory.find_similar("Write file")

        assert len(similar) >= 2
        # Both Write episodes should match
        contexts = [s.episode.context for s in similar]
        assert any("Write" in c for c in contexts)

    @pytest.mark.asyncio
    async def test_find_similar_with_embeddings(self, store: MemoryStore):
        """Should find similar episodes using embeddings."""
        # Create mock embedding provider
        mock_provider = AsyncMock()

        # Create distinct embeddings for different contexts
        def generate_embedding(text):
            if "Write" in text:
                return [0.9, 0.1, 0.0]
            elif "Read" in text:
                return [0.1, 0.9, 0.0]
            else:
                return [0.5, 0.5, 0.0]

        mock_provider.embed.side_effect = generate_embedding

        memory = EpisodicMemory(store, embedding_provider=mock_provider)

        # Record episodes with embeddings
        await memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s1",
                event_type="permission_request",
                context="Write operation",
                action_taken="approve",
                action_reason="test",
            )
        )
        await memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s2",
                event_type="permission_request",
                context="Read operation",
                action_taken="approve",
                action_reason="test",
            )
        )

        # Search for Write-like context
        similar = await memory.find_similar("Write something", min_similarity=0.3)

        # Should find the Write episode with higher similarity
        assert len(similar) >= 1

    @pytest.mark.asyncio
    async def test_find_similar_with_event_type_filter(
        self, episodic_memory: EpisodicMemory
    ):
        """Should filter by event type."""
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s1",
                event_type="permission_request",
                context="Tool request with keyword test",
                action_taken="approve",
                action_reason="test",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="s2",
                event_type="question",
                context="Question with keyword test",
                action_taken="answer",
                action_reason="test",
            )
        )

        permission_similar = await episodic_memory.find_similar(
            "test", event_type="permission_request"
        )
        question_similar = await episodic_memory.find_similar(
            "test", event_type="question"
        )

        assert len(permission_similar) == 1
        assert permission_similar[0].episode.event_type == "permission_request"
        assert len(question_similar) == 1
        assert question_similar[0].episode.event_type == "question"

    def test_cosine_similarity(self):
        """Cosine similarity should be computed correctly."""
        # Identical vectors
        assert EpisodicMemory._cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

        # Orthogonal vectors
        assert abs(EpisodicMemory._cosine_similarity([1, 0], [0, 1])) < 0.01

        # Opposite vectors
        assert EpisodicMemory._cosine_similarity([1, 0], [-1, 0]) == -1.0

        # Empty vectors
        assert EpisodicMemory._cosine_similarity([], []) == 0.0

        # Different length vectors
        assert EpisodicMemory._cosine_similarity([1, 0], [1, 0, 0]) == 0.0


class TestEpisodicMemoryQueries:
    """Tests for episode queries."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_get_recent(self, episodic_memory: EpisodicMemory):
        """Should get recent episodes."""
        for i in range(10):
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now() - timedelta(hours=i),
                    session_id="test",
                    event_type="test",
                    context=f"context-{i}",
                    action_taken="test",
                    action_reason="test",
                )
            )

        recent = await episodic_memory.get_recent(limit=3)

        assert len(recent) == 3
        # Most recent first
        assert recent[0].context == "context-0"

    @pytest.mark.asyncio
    async def test_get_by_event_type(self, episodic_memory: EpisodicMemory):
        """Should get episodes by event type."""
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="permission_request",
                context="test",
                action_taken="test",
                action_reason="test",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="question",
                context="test",
                action_taken="test",
                action_reason="test",
            )
        )

        permissions = await episodic_memory.get_by_event_type("permission_request")
        questions = await episodic_memory.get_by_event_type("question")

        assert len(permissions) == 1
        assert len(questions) == 1

    @pytest.mark.asyncio
    async def test_get_by_outcome(self, episodic_memory: EpisodicMemory):
        """Should get episodes by outcome."""
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="test",
                context="test",
                action_taken="test",
                action_reason="test",
                outcome="success",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="test",
                context="test",
                action_taken="test",
                action_reason="test",
                outcome="failure",
            )
        )

        successes = await episodic_memory.get_by_outcome("success")
        failures = await episodic_memory.get_by_outcome("failure")

        assert len(successes) == 1
        assert len(failures) == 1


class TestEpisodicMemoryStatistics:
    """Tests for success rate statistics."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_success_rate_empty(self, episodic_memory: EpisodicMemory):
        """Success rate should handle empty data."""
        stats = await episodic_memory.get_success_rate()

        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, episodic_memory: EpisodicMemory):
        """Should calculate success rate correctly."""
        # 3 successes, 2 failures
        for i in range(3):
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id="test",
                    event_type="permission_request",
                    context="test",
                    action_taken="approve_tool",
                    action_reason="test",
                    outcome="success",
                )
            )
        for i in range(2):
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id="test",
                    event_type="permission_request",
                    context="test",
                    action_taken="approve_tool",
                    action_reason="test",
                    outcome="failure",
                )
            )

        stats = await episodic_memory.get_success_rate()

        assert stats["total"] == 5
        assert stats["success"] == 3
        assert stats["failure"] == 2
        assert stats["success_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_success_rate_by_event_type(self, episodic_memory: EpisodicMemory):
        """Should filter by event type."""
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="permission_request",
                context="test",
                action_taken="approve_tool",
                action_reason="test",
                outcome="success",
            )
        )
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="question",
                context="test",
                action_taken="answer",
                action_reason="test",
                outcome="failure",
            )
        )

        perm_stats = await episodic_memory.get_success_rate(event_type="permission_request")
        q_stats = await episodic_memory.get_success_rate(event_type="question")

        assert perm_stats["success_rate"] == 1.0
        assert q_stats["success_rate"] == 0.0


class TestEpisodicMemoryOutcomes:
    """Tests for outcome updates."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_update_outcome(self, episodic_memory: EpisodicMemory):
        """Should update episode outcome."""
        episode_id = await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="test",
                event_type="test",
                context="test",
                action_taken="test",
                action_reason="test",
                outcome="pending",
            )
        )

        result = await episodic_memory.update_outcome(episode_id, "success")

        assert result is True

        recent = await episodic_memory.get_recent(limit=1)
        assert recent[0].outcome == "success"


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

    @pytest.fixture
    def episodic_memory(self, store: MemoryStore) -> EpisodicMemory:
        """Create episodic memory."""
        return EpisodicMemory(store)

    @pytest.mark.asyncio
    async def test_scenario_learning_from_file_writes(
        self, episodic_memory: EpisodicMemory
    ):
        """Scenario: Learning patterns from repeated file write approvals."""
        # User approves writes to src/ directory multiple times
        write_episodes = [
            ("src/main.py", "approve_tool", "success"),
            ("src/utils.py", "approve_tool", "success"),
            ("src/config.py", "approve_tool", "success"),
            (".env", "deny_tool", "success"),  # Denied write to .env
            ("src/auth.py", "approve_tool", "success"),
        ]

        for file_path, action, outcome in write_episodes:
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id="session-1",
                    event_type="permission_request",
                    context=f"Tool use request: Write to {file_path}",
                    action_taken=action,
                    action_reason="Based on file path risk",
                    outcome=outcome,
                    metadata={"tool_name": "Write", "file_path": file_path},
                )
            )

        # Search for similar Write operations
        similar = await episodic_memory.find_similar("Write to src/")

        # Should find src/ writes
        assert len(similar) >= 3

        # Check success rate for approvals
        stats = await episodic_memory.get_success_rate(action_taken="approve_tool")
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_scenario_question_answering_history(
        self, episodic_memory: EpisodicMemory
    ):
        """Scenario: Building history of answered questions."""
        questions = [
            ("Should I proceed with the refactoring?", "Yes", "success"),
            ("Run the test suite?", "Yes", "success"),
            ("Install this dependency?", "Yes", "failure"),  # Failed install
            ("Continue with the implementation?", "Yes", "success"),
            ("Which database should I use?", "PostgreSQL", "success"),
        ]

        for question, answer, outcome in questions:
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id="session-2",
                    event_type="question",
                    context=f"Question: {question}",
                    action_taken="answer",
                    action_reason=f"Answered: {answer}",
                    outcome=outcome,
                    metadata={"question": question, "answer": answer},
                )
            )

        # Search for similar "proceed" questions
        similar = await episodic_memory.find_similar(
            "proceed", event_type="question"
        )

        assert len(similar) >= 1
        assert any("proceed" in s.episode.context.lower() for s in similar)

    @pytest.mark.asyncio
    async def test_scenario_bash_command_patterns(
        self, episodic_memory: EpisodicMemory
    ):
        """Scenario: Learning safe vs dangerous bash commands."""
        commands = [
            ("npm install", "approve_tool", "success"),
            ("pytest tests/", "approve_tool", "success"),
            ("rm -rf /", "deny_tool", "success"),  # Correctly blocked
            ("git status", "approve_tool", "success"),
            ("sudo rm -rf", "deny_tool", "success"),  # Correctly blocked
            ("pip install -e .", "approve_tool", "success"),
        ]

        for command, action, outcome in commands:
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id="session-3",
                    event_type="permission_request",
                    context=f"Bash command: {command}",
                    action_taken=action,
                    action_reason="Command safety evaluation",
                    outcome=outcome,
                    metadata={"tool_name": "Bash", "command": command},
                )
            )

        # Query all permission requests
        all_perms = await episodic_memory.get_by_event_type("permission_request")

        approved = [e for e in all_perms if e.action_taken == "approve_tool"]
        denied = [e for e in all_perms if e.action_taken == "deny_tool"]

        assert len(approved) == 4
        assert len(denied) == 2

    @pytest.mark.asyncio
    async def test_scenario_session_recovery(
        self, episodic_memory: EpisodicMemory
    ):
        """Scenario: Using episodic memory for session recovery context."""
        session_id = "session-recovery"

        # Record a series of events in a session
        events = [
            ("permission_request", "Approved Write to src/auth.py", "approve_tool"),
            ("text_output", "Implementing authentication...", "no_action"),
            ("permission_request", "Approved Bash: pytest", "approve_tool"),
            ("tool_result", "Tests passed: 15/15", "no_action"),
            ("question", "Question: Should I commit?", "answer"),
        ]

        for event_type, context, action in events:
            await episodic_memory.record(
                Episode(
                    timestamp=datetime.now(),
                    session_id=session_id,
                    event_type=event_type,
                    context=context,
                    action_taken=action,
                    action_reason="Session progress",
                    outcome="success",
                )
            )

        # Recover session history
        session_history = await episodic_memory.get_recent(
            limit=10, session_id=session_id
        )

        assert len(session_history) == 5
        # Should be most recent first
        assert "commit" in session_history[0].context.lower()

    @pytest.mark.asyncio
    async def test_scenario_multi_session_learning(
        self, episodic_memory: EpisodicMemory
    ):
        """Scenario: Learning across multiple sessions."""
        # Session 1: Python project
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="python-project",
                event_type="permission_request",
                context="Bash test command: pytest tests/",
                action_taken="approve_tool",
                action_reason="Safe test command",
                outcome="success",
                metadata={"project_type": "python"},
            )
        )

        # Session 2: Node project
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="node-project",
                event_type="permission_request",
                context="Bash test command: npm test",
                action_taken="approve_tool",
                action_reason="Safe test command",
                outcome="success",
                metadata={"project_type": "node"},
            )
        )

        # Session 3: Another Python project
        await episodic_memory.record(
            Episode(
                timestamp=datetime.now(),
                session_id="another-python",
                event_type="permission_request",
                context="Bash test command: pytest tests/",
                action_taken="approve_tool",
                action_reason="Safe test command",
                outcome="success",
                metadata={"project_type": "python"},
            )
        )

        # Find similar test commands (search for "test command" which appears in all contexts)
        similar = await episodic_memory.find_similar("test command")

        assert len(similar) >= 2

        # Verify cross-session learning
        sessions = {s.episode.session_id for s in similar}
        assert len(sessions) >= 2
