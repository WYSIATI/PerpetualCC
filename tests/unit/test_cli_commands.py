"""Tests for pcc init and pcc logs CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from perpetualcc.cli import app


runner = CliRunner()


class TestInitCommand:
    """Tests for the pcc init command."""

    def test_init_shows_in_help(self):
        """Verify init command appears in help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "RAG" in result.output or "code understanding" in result.output

    def test_init_help(self):
        """Verify init command has proper help text."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output or "project" in result.output.lower()

    def test_init_nonexistent_path(self, tmp_path):
        """Verify error for non-existent path."""
        fake_path = tmp_path / "does_not_exist"
        result = runner.invoke(app, ["init", str(fake_path)])
        assert result.exit_code != 0
        assert "Error" in result.output or "does not exist" in result.output.lower()

    def test_init_file_instead_of_directory(self, tmp_path):
        """Verify error when path is a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        result = runner.invoke(app, ["init", str(file_path)])
        assert result.exit_code != 0
        assert "Error" in result.output or "not a directory" in result.output.lower()


class TestLogsCommand:
    """Tests for the pcc logs command."""

    def test_logs_shows_in_help(self):
        """Verify logs command appears in help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "logs" in result.output

    def test_logs_help(self):
        """Verify logs command has proper help text."""
        result = runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower() or "SESSION" in result.output

    def test_logs_nonexistent_session(self):
        """Verify error for non-existent session."""
        # Mock the session manager to return empty list
        with patch("perpetualcc.cli._get_session_manager") as mock_get_sm:
            mock_sm = MagicMock()
            mock_sm.list_sessions.return_value = []
            mock_get_sm.return_value = mock_sm

            result = runner.invoke(app, ["logs", "nonexistent123"])
            assert result.exit_code != 0
            assert "Error" in result.output or "not found" in result.output.lower()

    def test_logs_options(self):
        """Verify logs command accepts expected options."""
        result = runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0
        assert "--follow" in result.output or "-f" in result.output
        assert "--lines" in result.output or "-n" in result.output
        assert "--level" in result.output or "-l" in result.output


class TestMasterAgentMemoryAdapter:
    """Tests for the MasterAgentMemoryAdapter."""

    @pytest.fixture
    def mock_episodic_memory(self):
        """Create a mock episodic memory."""
        mock = AsyncMock()
        mock.record = AsyncMock(return_value=1)
        mock.find_similar = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def adapter(self, mock_episodic_memory):
        """Create an adapter with mock episodic memory."""
        from perpetualcc.memory import MasterAgentMemoryAdapter
        return MasterAgentMemoryAdapter(mock_episodic_memory)

    @pytest.mark.asyncio
    async def test_record_episode(self, adapter, mock_episodic_memory):
        """Test recording an episode through the adapter."""
        from datetime import datetime
        from dataclasses import dataclass, field
        from typing import Any

        # Create a mock episode similar to master_agent.Episode
        @dataclass
        class MockEpisode:
            timestamp: datetime
            session_id: str
            event_type: str
            context: str
            action_taken: str
            action_reason: str
            outcome: str = "pending"
            confidence: float = 1.0
            metadata: dict[str, Any] = field(default_factory=dict)

        episode = MockEpisode(
            timestamp=datetime.now(),
            session_id="test-session",
            event_type="permission_request",
            context="Tool use request: Write",
            action_taken="approve_tool",
            action_reason="Safe operation",
        )

        await adapter.record_episode(episode)

        # Verify episodic memory was called
        mock_episodic_memory.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar(self, adapter, mock_episodic_memory):
        """Test finding similar episodes through the adapter."""
        from datetime import datetime
        from dataclasses import dataclass

        # Mock the similar episode return value
        @dataclass
        class MockStoredEpisode:
            id: int
            session_id: str
            timestamp: datetime
            event_type: str
            context: str
            action_taken: str
            action_reason: str
            outcome: str
            confidence: float
            metadata: dict

        @dataclass
        class MockSimilarEpisode:
            episode: MockStoredEpisode
            similarity: float

        mock_stored = MockStoredEpisode(
            id=1,
            session_id="test",
            timestamp=datetime.now(),
            event_type="permission_request",
            context="Tool use: Write",
            action_taken="approve_tool",
            action_reason="Safe",
            outcome="success",
            confidence=0.9,
            metadata={},
        )
        mock_episodic_memory.find_similar.return_value = [
            MockSimilarEpisode(episode=mock_stored, similarity=0.85)
        ]

        results = await adapter.find_similar("Tool use request", top_k=3)

        mock_episodic_memory.find_similar.assert_called_once_with("Tool use request", top_k=3)
        assert len(results) == 1
        assert results[0].event_type == "permission_request"


class TestRAGContextIntegration:
    """Tests for RAG context integration in brain."""

    def test_question_context_has_rag_field(self):
        """Verify QuestionContext has rag_context field."""
        from perpetualcc.brain.base import QuestionContext

        context = QuestionContext(
            project_path="/test",
            question="Test question?",
            options=[],
            rag_context=[{"file_path": "test.py", "content": "code"}],
        )
        assert context.rag_context == [{"file_path": "test.py", "content": "code"}]

    def test_permission_context_has_rag_field(self):
        """Verify PermissionContext has rag_context field."""
        from perpetualcc.brain.base import PermissionContext

        context = PermissionContext(
            project_path="/test",
            rag_context=[{"file_path": "test.py", "name": "function"}],
        )
        assert context.rag_context == [{"file_path": "test.py", "name": "function"}]

    def test_default_rag_context_is_empty_list(self):
        """Verify default rag_context is empty list."""
        from perpetualcc.brain.base import QuestionContext, PermissionContext

        q_context = QuestionContext(
            project_path="/test",
            question="Test?",
            options=[],
        )
        assert q_context.rag_context == []

        p_context = PermissionContext(project_path="/test")
        assert p_context.rag_context == []
