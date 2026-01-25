"""Interactive CLI prompts for human intervention.

This module provides Rich-based interactive prompts for:
- Responding to escalation requests
- Graceful pause/exit handling
- Session summaries and code change reviews (at end of execution)

Note: By default, code changes are NOT reviewed during execution.
The user can enable mid-execution review with --review-changes flag.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from perpetualcc.human.escalation import EscalationRequest

console = Console()


class UserAction(Enum):
    """Actions the user can take during interactive mode."""

    CONTINUE = "continue"
    PAUSE = "pause"
    STOP = "stop"
    RESPOND = "respond"
    SKIP = "skip"


@dataclass
class InteractiveConfig:
    """Configuration for interactive prompts.

    Attributes:
        enable_mid_execution_review: Show code changes during execution
        show_thinking: Show Claude's thinking output
        show_tool_details: Show detailed tool input/output
    """

    enable_mid_execution_review: bool = False
    show_thinking: bool = True
    show_tool_details: bool = True


@dataclass
class InteractiveContext:
    """Context for interactive session.

    Attributes:
        session_id: Current session ID
        project_path: Project path
        current_task: Current task description
        modified_files: List of modified files (tracked for end summary)
        pending_escalations: Pending escalation count
    """

    session_id: str | None = None
    project_path: str | None = None
    current_task: str | None = None
    modified_files: list[str] = field(default_factory=list)
    pending_escalations: int = 0


class InteractivePrompt:
    """Interactive prompt handler for CLI.

    Provides Rich-based prompts for human interaction during sessions.

    Usage:
        prompt = InteractivePrompt()

        # Show escalation and get response
        response = prompt.prompt_escalation(request)

        # At end of session, show summary
        prompt.show_session_summary(success=True, modified_files=files)
    """

    def __init__(self, config: InteractiveConfig | None = None):
        """Initialize the interactive prompt."""
        self.config = config or InteractiveConfig()
        self._paused = False
        self._stop_requested = False
        self._context = InteractiveContext()

    @property
    def is_paused(self) -> bool:
        """Check if session is paused."""
        return self._paused

    @property
    def stop_requested(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested

    def reset(self) -> None:
        """Reset state for new session."""
        self._paused = False
        self._stop_requested = False
        self._context = InteractiveContext()

    def set_context(
        self,
        session_id: str | None = None,
        project_path: str | None = None,
        current_task: str | None = None,
    ) -> None:
        """Update the interactive context."""
        if session_id:
            self._context.session_id = session_id
        if project_path:
            self._context.project_path = project_path
        if current_task:
            self._context.current_task = current_task

    def track_modified_file(self, file_path: str) -> None:
        """Track a modified file for end-of-session summary."""
        if file_path not in self._context.modified_files:
            self._context.modified_files.append(file_path)

    def prompt_escalation(self, request: EscalationRequest) -> str | None:
        """Prompt user to respond to an escalation.

        Args:
            request: The escalation request

        Returns:
            User's response, or None if skipped/cancelled
        """
        console.print()

        panel_content = self._build_escalation_panel(request)
        console.print(
            Panel(
                panel_content,
                title="[bold yellow]Human Input Needed[/bold yellow]",
                border_style="yellow",
            )
        )

        if request.options:
            return self._prompt_with_options(request)
        else:
            return self._prompt_freeform(request)

    def _build_escalation_panel(self, request: EscalationRequest) -> Text:
        """Build the content for escalation panel."""
        text = Text()

        text.append("Session: ", style="bold")
        text.append(f"{request.session_id[:8]}\n")

        text.append("Type: ", style="bold")
        text.append(f"{request.escalation_type.value}\n")

        if request.context:
            text.append("Context: ", style="bold")
            text.append(f"{request.context}\n")

        text.append("\n")
        text.append("Question: ", style="bold cyan")
        text.append(f"{request.question}\n")

        if request.brain_suggestion:
            text.append("\n")
            text.append("AI Suggestion: ", style="bold green")
            text.append(f"{request.brain_suggestion} ")
            text.append(
                f"({int(request.brain_confidence * 100)}% confidence)",
                style="dim",
            )
            text.append("\n")

        return text

    def _prompt_with_options(self, request: EscalationRequest) -> str | None:
        """Prompt user to select from options."""
        console.print("\n[bold]Options:[/bold]")
        for i, option in enumerate(request.options, 1):
            if (
                request.brain_suggestion
                and option.lower() == request.brain_suggestion.lower()
            ):
                console.print(f"  [{i}] {option} [green](suggested)[/green]")
            else:
                console.print(f"  [{i}] {option}")

        if request.brain_suggestion:
            console.print(f"  [a] Accept suggestion: {request.brain_suggestion}")
        console.print("  [c] Custom response")
        console.print("  [s] Skip (use AI suggestion if available)")
        console.print()

        try:
            choice = Prompt.ask(
                "Your choice",
                default="a" if request.brain_suggestion else "1",
            )

            if choice.lower() == "a" and request.brain_suggestion:
                return request.brain_suggestion

            if choice.lower() == "c":
                return Prompt.ask("Enter your response")

            if choice.lower() == "s":
                return request.brain_suggestion

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(request.options):
                    return request.options[idx]
            except ValueError:
                pass

            console.print("[red]Invalid choice[/red]")
            return self._prompt_with_options(request)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Skipped[/yellow]")
            return request.brain_suggestion

    def _prompt_freeform(self, request: EscalationRequest) -> str | None:
        """Prompt user for freeform response."""
        try:
            default = request.brain_suggestion or ""
            response = Prompt.ask("Your response", default=default)
            return response if response else None

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Skipped[/yellow]")
            return request.brain_suggestion

    def show_pending_escalations(
        self, escalations: list[EscalationRequest]
    ) -> None:
        """Display a table of pending escalations.

        Args:
            escalations: List of pending escalation requests
        """
        if not escalations:
            console.print("[dim]No pending escalations.[/dim]")
            return

        table = Table(title="Pending Human Decisions")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Session", style="dim", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Question")
        table.add_column("Age", justify="right")

        for i, esc in enumerate(escalations, 1):
            age_secs = esc.age_seconds
            if age_secs < 60:
                age = f"{age_secs}s"
            elif age_secs < 3600:
                age = f"{age_secs // 60}m"
            else:
                age = f"{age_secs // 3600}h"

            question = esc.question[:50]
            if len(esc.question) > 50:
                question += "..."

            table.add_row(
                str(i),
                esc.session_id[:8],
                esc.escalation_type.value,
                question,
                age,
            )

        console.print(table)
        console.print()
        console.print("[dim]Respond with: pcc respond <#> <response>[/dim]")
        console.print("[dim]Or interactive: pcc respond <#> --interactive[/dim]")

    def prompt_pause_or_continue(self) -> UserAction:
        """Prompt user to pause or continue.

        Called when Ctrl+C is pressed to give user options.

        Returns:
            UserAction indicating what to do next
        """
        console.print()
        console.print(
            "[yellow]Session interrupted. What would you like to do?[/yellow]"
        )
        console.print("  [c] Continue session")
        console.print("  [p] Pause session (can resume later)")
        console.print("  [s] Stop session")
        console.print()

        try:
            choice = Prompt.ask(
                "Choice",
                choices=["c", "p", "s"],
                default="c",
            )

            if choice == "c":
                return UserAction.CONTINUE
            elif choice == "p":
                self._paused = True
                return UserAction.PAUSE
            else:
                self._stop_requested = True
                return UserAction.STOP

        except (KeyboardInterrupt, EOFError):
            self._stop_requested = True
            return UserAction.STOP

    def confirm_action(
        self,
        message: str,
        default: bool = True,
    ) -> bool:
        """Prompt user to confirm an action."""
        try:
            return Confirm.ask(message, default=default)
        except (KeyboardInterrupt, EOFError):
            return False

    def show_session_header(
        self,
        session_id: str,
        project_path: str,
        task: str | None = None,
    ) -> None:
        """Display session header."""
        from pathlib import Path

        project_name = Path(project_path).name

        content = Text()
        content.append("Session: ", style="bold")
        content.append(f"{session_id[:8]}\n")
        content.append("Project: ", style="bold")
        content.append(f"{project_name}\n")
        if task:
            content.append("Task: ", style="bold")
            content.append(f"{task[:60]}")
            if len(task) > 60:
                content.append("...")

        console.print(
            Panel(
                content,
                title="[bold cyan]PerpetualCC Session[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()
        console.print("[dim]Press Ctrl+C to pause/exit[/dim]")
        console.print()

        self.set_context(session_id, project_path, task)

    def show_session_summary(
        self,
        success: bool = True,
        turns: int | None = None,
        cost_usd: float | None = None,
        modified_files: list[str] | None = None,
        offer_review: bool = True,
    ) -> bool:
        """Display session summary at end of execution.

        Args:
            success: Whether session completed successfully
            turns: Number of turns
            cost_usd: Total cost
            modified_files: List of modified files
            offer_review: Whether to offer code review option

        Returns:
            True if user wants to review changes, False otherwise
        """
        console.print()

        if success:
            status_text = "[bold green]Session Completed[/bold green]"
        else:
            status_text = "[bold red]Session Failed[/bold red]"

        details = []
        if turns:
            details.append(f"Turns: {turns}")
        if cost_usd:
            details.append(f"Cost: ${cost_usd:.4f}")

        files = modified_files or self._context.modified_files
        if files:
            details.append(f"Files modified: {len(files)}")

        detail_str = " | ".join(details) if details else ""

        console.print(
            Panel(
                f"{status_text}\n{detail_str}",
                border_style="green" if success else "red",
            )
        )

        if files:
            console.print()
            console.print(f"[bold]Modified files ({len(files)}):[/bold]")
            for f in files[:20]:
                console.print(f"  - {f}")
            if len(files) > 20:
                remaining = len(files) - 20
                console.print(f"  ... and {remaining} more")

            if offer_review and len(files) > 0:
                console.print()
                return self.confirm_action(
                    "Would you like to review the changes?",
                    default=False,
                )

        return False

    def show_file_for_review(
        self,
        file_path: str,
        content: str | None = None,
    ) -> UserAction:
        """Show a file for review (only when explicitly enabled).

        Args:
            file_path: Path to the file
            content: Optional content to show

        Returns:
            UserAction for what to do next
        """
        console.print()
        console.print(f"[bold cyan]File: {file_path}[/bold cyan]")

        if content:
            console.print()
            # Show truncated content
            lines = content.split("\n")
            for i, line in enumerate(lines[:50]):
                console.print(f"[dim]{i+1:4}[/dim] {line}")
            if len(lines) > 50:
                console.print(f"  ... ({len(lines) - 50} more lines)")

        console.print()
        console.print("  [n] Next file")
        console.print("  [d] Done reviewing")
        console.print("  [q] Quit review")

        try:
            choice = Prompt.ask("Choice", choices=["n", "d", "q"], default="n")
            if choice == "d" or choice == "q":
                return UserAction.STOP
            return UserAction.CONTINUE
        except (KeyboardInterrupt, EOFError):
            return UserAction.STOP


class SignalHandler:
    """Handler for graceful pause/exit on signals.

    Sets up signal handlers for SIGINT (Ctrl+C) and SIGTERM
    to allow graceful session pause/exit.

    Usage:
        handler = SignalHandler(on_interrupt=my_callback)
        handler.setup()
        # ... run session ...
        handler.restore()
    """

    def __init__(
        self,
        on_interrupt: Callable[[], UserAction] | None = None,
    ):
        """Initialize the signal handler."""
        self.on_interrupt = on_interrupt
        self._original_sigint = None
        self._original_sigterm = None
        self._interrupt_count = 0

    def setup(self) -> None:
        """Set up signal handlers."""
        self._original_sigint = signal.signal(
            signal.SIGINT, self._handle_interrupt
        )
        self._original_sigterm = signal.signal(
            signal.SIGTERM, self._handle_interrupt
        )
        self._interrupt_count = 0

    def restore(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal."""
        self._interrupt_count += 1

        if self._interrupt_count >= 3:
            console.print("\n[red]Force exit[/red]")
            sys.exit(1)

        if self.on_interrupt:
            action = self.on_interrupt()
            if action == UserAction.STOP:
                sys.exit(0)
        else:
            console.print("\n[yellow]Interrupt received[/yellow]")
