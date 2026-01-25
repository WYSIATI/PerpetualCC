"""Unit tests for brain factory."""

from __future__ import annotations

import pytest

from perpetualcc.brain.base import Brain
from perpetualcc.brain.factory import create_brain, get_brain_status
from perpetualcc.brain.rule_based import RuleBasedBrain
from perpetualcc.config import BrainConfig, BrainType, GeminiConfig, OllamaConfig


class TestCreateBrain:
    """Tests for create_brain factory function."""

    def test_create_rule_based_brain(self):
        """Test creating rule-based brain."""
        config = BrainConfig(type=BrainType.RULE_BASED)
        brain = create_brain(config)

        assert isinstance(brain, RuleBasedBrain)
        assert isinstance(brain, Brain)

    def test_create_rule_based_with_custom_threshold(self):
        """Test creating rule-based brain with custom confidence threshold."""
        config = BrainConfig(
            type=BrainType.RULE_BASED,
            confidence_threshold=0.9,
        )
        brain = create_brain(config)

        assert isinstance(brain, RuleBasedBrain)
        assert brain.get_confidence_threshold() == 0.9

    def test_create_gemini_brain_without_key_falls_back(self, monkeypatch):
        """Test that Gemini brain without API key falls back to rule-based."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        config = BrainConfig(
            type=BrainType.GEMINI,
            gemini=GeminiConfig(api_key=None),
        )
        brain = create_brain(config)

        # Should fall back to rule-based
        assert isinstance(brain, RuleBasedBrain)

    def test_create_ollama_brain_unavailable_falls_back(self):
        """Test that Ollama brain falls back when server unavailable."""
        config = BrainConfig(
            type=BrainType.OLLAMA,
            ollama=OllamaConfig(
                host="http://localhost:99999",  # Invalid port
            ),
        )
        brain = create_brain(config)

        # Should fall back to rule-based
        assert isinstance(brain, RuleBasedBrain)


class TestGetBrainStatus:
    """Tests for get_brain_status function."""

    def test_rule_based_status(self):
        """Test status for rule-based brain."""
        config = BrainConfig(type=BrainType.RULE_BASED)
        status = get_brain_status(config)

        assert status["type"] == "rule_based"
        assert status["available"] is True
        assert "always available" in status["message"].lower()

    def test_gemini_status_without_key(self, monkeypatch):
        """Test status for Gemini brain without API key."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        config = BrainConfig(
            type=BrainType.GEMINI,
            gemini=GeminiConfig(api_key=None),
        )
        status = get_brain_status(config)

        assert status["type"] == "gemini"
        # Either not available (no key) or package not installed
        if status["available"]:
            # Key must be set from somewhere
            pass
        else:
            assert "api key" in status["message"].lower() or "not installed" in status["message"].lower()

    def test_gemini_status_with_key(self, monkeypatch):
        """Test status for Gemini brain with API key."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        config = BrainConfig(
            type=BrainType.GEMINI,
            gemini=GeminiConfig(api_key="env:GEMINI_API_KEY"),
        )
        status = get_brain_status(config)

        assert status["type"] == "gemini"
        # May or may not be available depending on google-genai installation
        if "not installed" not in status["message"].lower():
            assert status["details"].get("api_key_set") is True

    def test_ollama_status_unavailable(self):
        """Test status for Ollama brain when server unavailable."""
        config = BrainConfig(
            type=BrainType.OLLAMA,
            ollama=OllamaConfig(
                host="http://localhost:99999",
            ),
        )
        status = get_brain_status(config)

        assert status["type"] == "ollama"
        # Either not available or package not installed
        assert not status["available"] or "not installed" in status["message"].lower()

    def test_status_details_content(self):
        """Test that status contains expected details."""
        config = BrainConfig(
            type=BrainType.RULE_BASED,
            confidence_threshold=0.85,
        )
        status = get_brain_status(config)

        assert "details" in status
        assert status["details"]["confidence_threshold"] == 0.85


class TestBrainInterface:
    """Tests to verify brain implementations follow the interface."""

    @pytest.mark.asyncio
    async def test_rule_based_brain_answer_question(self):
        """Test rule-based brain can answer questions."""
        config = BrainConfig(type=BrainType.RULE_BASED)
        brain = create_brain(config)

        from perpetualcc.brain.base import QuestionContext

        context = QuestionContext(
            project_path="/test/project",
            question="Should I proceed?",
            options=[{"label": "Yes"}, {"label": "No"}],
        )

        answer = await brain.answer_question(
            question="Should I proceed?",
            options=[{"label": "Yes"}, {"label": "No"}],
            context=context,
        )

        assert answer.selected is not None or answer.confidence == 0.0
        assert 0.0 <= answer.confidence <= 1.0
        assert answer.reasoning is not None

    @pytest.mark.asyncio
    async def test_rule_based_brain_evaluate_permission(self):
        """Test rule-based brain can evaluate permissions."""
        config = BrainConfig(type=BrainType.RULE_BASED)
        brain = create_brain(config)

        from perpetualcc.brain.base import PermissionContext

        context = PermissionContext(
            project_path="/test/project",
            current_task="Test task",
        )

        decision = await brain.evaluate_permission(
            tool_name="Bash",
            tool_input={"command": "git status"},
            context=context,
        )

        assert isinstance(decision.approve, bool)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.reason is not None
