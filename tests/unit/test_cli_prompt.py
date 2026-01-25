"""Unit tests for the CLI prompt module.

These tests cover:
1. InteractivePrompt state management
2. Escalation prompt building
3. Signal handler setup
4. Session header/footer display
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest

from perpetualcc.human.cli_prompt import (
    InteractiveConfig,
    InteractiveContext,
    InteractivePrompt,
    SignalHandler,
    UserAction,
)
from perpetualcc.human.escalation import EscalationRequest, EscalationType


class TestInteractiveConfig:
    """Tests for InteractiveConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = InteractiveConfig()
        assert config.enable_mid_execution_review is False
        assert config.show_thinking is True
        assert config.show_tool_details is True

    def test_custom_config(self):
        """Config should accept custom values."""
        config = InteractiveConfig(
            enable_mid_execution_review=True,
            show_thinking=False,
            show_tool_details=False,
        )
        assert config.enable_mid_execution_review is True
        assert config.show_thinking is False


class TestInteractiveContext:
    """Tests for InteractiveContext."""

    def test_default_context(self):
        """Default context should be empty."""
        context = InteractiveContext()
        assert context.session_id is None
        assert context.project_path is None
        assert context.current_task is None
        assert context.modified_files == []
        assert context.pending_escalations == 0

    def test_context_with_values(self):
        """Context should store values."""
        context = InteractiveContext(
            session_id="session-123",
            project_path="/path/to/project",
            current_task="Implement feature",
            modified_files=["file1.py", "file2.py"],
            pending_escalations=2,
        )
        assert context.session_id == "session-123"
        assert len(context.modified_files) == 2


class TestInteractivePrompt:
    """Tests for InteractivePrompt class."""

    def test_initial_state(self):
        """Prompt should start with clean state."""
        prompt = InteractivePrompt()
        assert prompt.is_paused is False
        assert prompt.stop_requested is False

    def test_reset_clears_state(self):
        """reset should clear all state."""
        prompt = InteractivePrompt()
        prompt._paused = True
        prompt._stop_requested = True
        prompt._context.session_id = "test"

        prompt.reset()

        assert prompt.is_paused is False
        assert prompt.stop_requested is False
        assert prompt._context.session_id is None

    def test_set_context_updates_fields(self):
        """set_context should update context fields."""
        prompt = InteractivePrompt()
        prompt.set_context(
            session_id="session-123",
            project_path="/path/to/project",
            current_task="Task X",
        )

        assert prompt._context.session_id == "session-123"
        assert prompt._context.project_path == "/path/to/project"
        assert prompt._context.current_task == "Task X"

    def test_set_context_partial_update(self):
        """set_context should allow partial updates."""
        prompt = InteractivePrompt()
        prompt.set_context(session_id="session-123")
        prompt.set_context(project_path="/path")

        assert prompt._context.session_id == "session-123"
        assert prompt._context.project_path == "/path"

    def test_track_modified_file(self):
        """track_modified_file should add unique files."""
        prompt = InteractivePrompt()
        prompt.track_modified_file("file1.py")
        prompt.track_modified_file("file2.py")
        prompt.track_modified_file("file1.py")  # Duplicate

        assert len(prompt._context.modified_files) == 2
        assert "file1.py" in prompt._context.modified_files
        assert "file2.py" in prompt._context.modified_files


class TestInteractivePromptEscalation:
    """Tests for escalation prompt functionality."""

    def test_build_escalation_panel_basic(self):
        """_build_escalation_panel should format basic request."""
        prompt = InteractivePrompt()
        request = EscalationRequest.create(
            session_id="session-12345678",
            escalation_type=EscalationType.QUESTION,
            context="Working on feature",
            question="Which approach?",
        )

        panel = prompt._build_escalation_panel(request)

        # Panel should be a Rich Text object
        assert panel is not None
        text_str = str(panel)
        assert "session-" in text_str
        assert "question" in text_str.lower()

    def test_build_escalation_panel_with_suggestion(self):
        """_build_escalation_panel should include suggestion."""
        prompt = InteractivePrompt()
        request = EscalationRequest.create(
            session_id="session-12345678",
            escalation_type=EscalationType.QUESTION,
            context="",
            question="Proceed?",
            brain_suggestion="Yes",
            brain_confidence=0.85,
        )

        panel = prompt._build_escalation_panel(request)
        text_str = str(panel)

        assert "Yes" in text_str
        assert "85" in text_str  # 85% confidence


class TestUserAction:
    """Tests for UserAction enum."""

    def test_user_action_values(self):
        """UserAction should have expected values."""
        assert UserAction.CONTINUE.value == "continue"
        assert UserAction.PAUSE.value == "pause"
        assert UserAction.STOP.value == "stop"
        assert UserAction.RESPOND.value == "respond"
        assert UserAction.SKIP.value == "skip"


class TestSignalHandler:
    """Tests for SignalHandler class."""

    def test_setup_registers_handlers(self):
        """setup should register signal handlers."""
        handler = SignalHandler()
        handler.setup()

        # Should have saved original handlers
        assert handler._original_sigint is not None
        assert handler._original_sigterm is not None

        handler.restore()

    def test_restore_restores_handlers(self):
        """restore should restore original handlers."""
        handler = SignalHandler()
        original_sigint = signal.getsignal(signal.SIGINT)

        handler.setup()
        handler.restore()

        current_sigint = signal.getsignal(signal.SIGINT)
        assert current_sigint == original_sigint

    def test_callback_called_on_interrupt(self):
        """on_interrupt callback should be called."""
        results = []

        def callback() -> UserAction:
            results.append("called")
            return UserAction.CONTINUE

        handler = SignalHandler(on_interrupt=callback)
        handler.setup()

        # Simulate interrupt
        handler._handle_interrupt(signal.SIGINT, None)

        assert len(results) == 1
        assert results[0] == "called"

        handler.restore()

    def test_multiple_interrupts_increment_count(self):
        """Multiple interrupts should increment counter."""
        handler = SignalHandler(on_interrupt=lambda: UserAction.CONTINUE)
        handler.setup()

        handler._handle_interrupt(signal.SIGINT, None)
        assert handler._interrupt_count == 1

        handler._handle_interrupt(signal.SIGINT, None)
        assert handler._interrupt_count == 2

        handler.restore()


class TestInteractivePromptOutput:
    """Tests for output display methods."""

    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_session_header(self, mock_console: MagicMock):
        """show_session_header should print header panel."""
        prompt = InteractivePrompt()
        prompt.show_session_header(
            session_id="session-12345678",
            project_path="/path/to/project",
            task="Test task",
        )

        mock_console.print.assert_called()

    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_session_summary_success(self, mock_console: MagicMock):
        """show_session_summary should display success summary."""
        prompt = InteractivePrompt()
        prompt._context.modified_files = ["file1.py", "file2.py"]

        prompt.show_session_summary(
            success=True,
            turns=25,
            cost_usd=0.0534,
            offer_review=False,
        )

        mock_console.print.assert_called()

    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_session_summary_failure(self, mock_console: MagicMock):
        """show_session_summary should display failure summary."""
        prompt = InteractivePrompt()

        prompt.show_session_summary(
            success=False,
            turns=10,
            cost_usd=0.0234,
            offer_review=False,
        )

        mock_console.print.assert_called()

    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_pending_escalations_empty(self, mock_console: MagicMock):
        """show_pending_escalations should handle empty list."""
        prompt = InteractivePrompt()
        prompt.show_pending_escalations([])

        mock_console.print.assert_called()

    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_pending_escalations_with_items(self, mock_console: MagicMock):
        """show_pending_escalations should display table."""
        prompt = InteractivePrompt()
        escalations = [
            EscalationRequest.create(
                session_id="session-123",
                escalation_type=EscalationType.QUESTION,
                context="",
                question="Q1?",
            ),
            EscalationRequest.create(
                session_id="session-456",
                escalation_type=EscalationType.PERMISSION,
                context="",
                question="Allow X?",
            ),
        ]

        prompt.show_pending_escalations(escalations)

        mock_console.print.assert_called()


class TestInteractivePromptConfirm:
    """Tests for confirmation methods."""

    @patch("perpetualcc.human.cli_prompt.Confirm")
    def test_confirm_action_returns_true(self, mock_confirm: MagicMock):
        """confirm_action should return confirmation result."""
        mock_confirm.ask.return_value = True

        prompt = InteractivePrompt()
        result = prompt.confirm_action("Proceed?", default=True)

        assert result is True
        mock_confirm.ask.assert_called_once()

    @patch("perpetualcc.human.cli_prompt.Confirm")
    def test_confirm_action_handles_interrupt(self, mock_confirm: MagicMock):
        """confirm_action should handle keyboard interrupt."""
        mock_confirm.ask.side_effect = KeyboardInterrupt()

        prompt = InteractivePrompt()
        result = prompt.confirm_action("Proceed?")

        assert result is False

    @patch("perpetualcc.human.cli_prompt.Confirm")
    def test_confirm_action_handles_eof(self, mock_confirm: MagicMock):
        """confirm_action should handle EOF."""
        mock_confirm.ask.side_effect = EOFError()

        prompt = InteractivePrompt()
        result = prompt.confirm_action("Proceed?")

        assert result is False


class TestFileReview:
    """Tests for file review functionality."""

    @patch("perpetualcc.human.cli_prompt.Prompt")
    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_file_for_review_continue(
        self, mock_console: MagicMock, mock_prompt: MagicMock
    ):
        """show_file_for_review should return CONTINUE on 'n'."""
        mock_prompt.ask.return_value = "n"

        prompt = InteractivePrompt()
        result = prompt.show_file_for_review("test.py", "# Test content")

        assert result == UserAction.CONTINUE

    @patch("perpetualcc.human.cli_prompt.Prompt")
    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_file_for_review_done(
        self, mock_console: MagicMock, mock_prompt: MagicMock
    ):
        """show_file_for_review should return STOP on 'd'."""
        mock_prompt.ask.return_value = "d"

        prompt = InteractivePrompt()
        result = prompt.show_file_for_review("test.py")

        assert result == UserAction.STOP

    @patch("perpetualcc.human.cli_prompt.Prompt")
    @patch("perpetualcc.human.cli_prompt.console")
    def test_show_file_for_review_handles_interrupt(
        self, mock_console: MagicMock, mock_prompt: MagicMock
    ):
        """show_file_for_review should handle interrupt."""
        mock_prompt.ask.side_effect = KeyboardInterrupt()

        prompt = InteractivePrompt()
        result = prompt.show_file_for_review("test.py")

        assert result == UserAction.STOP
