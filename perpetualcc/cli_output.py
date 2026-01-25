"""Rich CLI output for PerpetualCC sessions.

This module provides Claude Code CLI-like output with:
- Live status spinner during thinking/processing
- File change display with content diffs
- Verbose tool output
- Progress tracking
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from perpetualcc.claude.types import (
    AskQuestionEvent,
    ClaudeEvent,
    InitEvent,
    RateLimitEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)


class OutputVerbosity(Enum):
    """Output verbosity levels."""

    QUIET = "quiet"  # Minimal output - only errors and final result
    NORMAL = "normal"  # Show tools, Claude text, file changes summary
    VERBOSE = "verbose"  # Show everything including file contents, full tool output


@dataclass
class OutputConfig:
    """Configuration for CLI output.

    Attributes:
        verbosity: Output verbosity level
        show_file_changes: Show file content changes (Write/Edit)
        show_thinking: Show Claude's thinking process
        show_tool_output: Show tool results
        show_timestamps: Show timestamps on each line
        max_file_preview_lines: Max lines to show for file changes
        max_tool_output_lines: Max lines to show for tool output
        color_enabled: Enable colored output
        use_markdown: Render Claude's text as markdown
    """

    verbosity: OutputVerbosity = OutputVerbosity.NORMAL
    show_file_changes: bool = True
    show_thinking: bool = True
    show_tool_output: bool = True
    show_timestamps: bool = True
    max_file_preview_lines: int = 50
    max_tool_output_lines: int = 30
    color_enabled: bool = True
    use_markdown: bool = True

    @classmethod
    def quiet(cls) -> OutputConfig:
        """Create a quiet configuration."""
        return cls(
            verbosity=OutputVerbosity.QUIET,
            show_file_changes=False,
            show_thinking=False,
            show_tool_output=False,
        )

    @classmethod
    def verbose(cls) -> OutputConfig:
        """Create a verbose configuration."""
        return cls(
            verbosity=OutputVerbosity.VERBOSE,
            show_file_changes=True,
            show_thinking=True,
            show_tool_output=True,
            max_file_preview_lines=100,
            max_tool_output_lines=100,
        )


@dataclass
class FileChange:
    """Tracks a file change during a session."""

    file_path: str
    operation: str  # "write", "edit", "delete"
    old_content: str | None = None
    new_content: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_new_file(self) -> bool:
        """Check if this is a new file creation."""
        return self.operation == "write" and self.old_content is None

    @property
    def lines_added(self) -> int:
        """Count lines added."""
        if not self.new_content:
            return 0
        if not self.old_content:
            return len(self.new_content.splitlines())
        return max(0, len(self.new_content.splitlines()) - len(self.old_content.splitlines()))

    @property
    def lines_removed(self) -> int:
        """Count lines removed."""
        if not self.old_content:
            return 0
        if not self.new_content:
            return len(self.old_content.splitlines())
        return max(0, len(self.old_content.splitlines()) - len(self.new_content.splitlines()))


class SessionOutputManager:
    """Manages rich output for a PerpetualCC session.

    Provides Claude Code CLI-like output with live status updates,
    file change tracking, and verbose tool output.
    """

    def __init__(
        self,
        config: OutputConfig | None = None,
        console: Console | None = None,
    ):
        """Initialize the output manager.

        Args:
            config: Output configuration
            console: Rich console to use (creates new one if not provided)
        """
        self.config = config or OutputConfig()
        self.console = console or Console()

        # State tracking
        self._live: Live | None = None
        self._current_status: str = ""
        self._is_processing: bool = False
        self._file_changes: list[FileChange] = []
        self._pending_tool_use: dict[str, ToolUseEvent] = {}
        self._session_id: str | None = None
        self._start_time: datetime | None = None
        self._turn_count: int = 0
        self._last_file_contents: dict[str, str] = {}  # Cache for file contents before edit

    def _timestamp(self) -> str:
        """Get formatted timestamp."""
        if not self.config.show_timestamps:
            return ""
        return f"[dim][{datetime.now().strftime('%H:%M:%S')}][/dim] "

    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension for syntax highlighting."""
        ext = Path(file_path).suffix.lower()
        # Map common extensions to lexer names
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        return ext_map.get(ext, "text")

    def _get_brain_info(self) -> str:
        """Get brain configuration info for display."""
        try:
            from perpetualcc.brain.factory import get_brain_status
            from perpetualcc.config import get_config

            config = get_config()
            status = get_brain_status(config.brain)

            brain_type = status["type"]
            available = status["available"]

            if brain_type == "rule_based":
                return "rule_based (pattern matching)"
            elif brain_type == "ollama":
                model = status.get("details", {}).get("model", "unknown")
                if available:
                    return f"ollama ({model})"
                else:
                    return f"ollama ({model}) [yellow]unavailable[/yellow]"
            elif brain_type == "gemini":
                model = status.get("details", {}).get("model", "unknown")
                if available:
                    return f"gemini ({model})"
                else:
                    return "gemini [yellow]no API key[/yellow]"
            else:
                return brain_type
        except Exception:
            return "rule_based (default)"

    def start_session(
        self,
        session_id: str,
        project_path: str | Path,
        task: str,
    ) -> None:
        """Display session start header.

        Args:
            session_id: Session identifier
            project_path: Project path
            task: Task description
        """
        self._session_id = session_id
        self._start_time = datetime.now()
        self._file_changes = []
        self._turn_count = 0

        project_name = Path(project_path).name

        # Get brain config info
        brain_info = self._get_brain_info()

        content = Text()
        content.append("Session: ", style="bold")
        content.append(f"{session_id[:8]}\n")
        content.append("Project: ", style="bold")
        content.append(f"{project_name}\n")
        content.append("Brain: ", style="bold")
        content.append(f"{brain_info}\n")
        content.append("Task: ", style="bold")
        task_display = task[:60] + "..." if len(task) > 60 else task
        content.append(task_display)

        self.console.print(
            Panel(
                content,
                title="[bold cyan]PerpetualCC Session[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()
        self.console.print("[dim]Press Ctrl+C to pause/exit[/dim]")
        self.console.print()

    def start_processing(self, status: str = "Connecting to Claude...") -> None:
        """Start showing a processing spinner.

        Args:
            status: Status text to show
        """
        if self.config.verbosity == OutputVerbosity.QUIET:
            return

        self._is_processing = True
        self._current_status = status

        # Use Live display for spinner
        spinner = Spinner("dots", text=Text(f" {status}", style="cyan"))
        self._live = Live(spinner, console=self.console, refresh_per_second=10)
        self._live.start()

    def update_status(self, status: str) -> None:
        """Update the processing status.

        Args:
            status: New status text
        """
        if not self._is_processing or not self._live:
            return

        self._current_status = status
        spinner = Spinner("dots", text=Text(f" {status}", style="cyan"))
        self._live.update(spinner)

    def stop_processing(self) -> None:
        """Stop the processing spinner."""
        if self._live:
            self._live.stop()
            self._live = None
        self._is_processing = False

    def handle_event(self, event: ClaudeEvent) -> None:
        """Handle a Claude event and display appropriate output.

        Args:
            event: The event to handle
        """
        # Stop spinner before printing
        if self._is_processing and not isinstance(event, ThinkingEvent):
            self.stop_processing()

        if isinstance(event, InitEvent):
            self._handle_init(event)
        elif isinstance(event, TextEvent):
            self._handle_text(event)
        elif isinstance(event, ThinkingEvent):
            self._handle_thinking(event)
        elif isinstance(event, ToolUseEvent):
            self._handle_tool_use(event)
        elif isinstance(event, ToolResultEvent):
            self._handle_tool_result(event)
        elif isinstance(event, AskQuestionEvent):
            self._handle_question(event)
        elif isinstance(event, RateLimitEvent):
            self._handle_rate_limit(event)
        elif isinstance(event, ResultEvent):
            self._handle_result(event)

    def _handle_init(self, event: InitEvent) -> None:
        """Handle session init event."""
        ts = self._timestamp()
        self.console.print(f"{ts}Session started: [cyan]{event.session_id}[/cyan]")

    def _handle_text(self, event: TextEvent) -> None:
        """Handle Claude text output."""
        if self.config.verbosity == OutputVerbosity.QUIET:
            return

        text = event.text.strip()
        if not text:
            return

        ts = self._timestamp()

        if self.config.use_markdown:
            # Render as markdown for rich formatting
            self.console.print(f"{ts}[bold]Claude:[/bold]")
            md = Markdown(text)
            self.console.print(Padding(md, (0, 0, 0, 2)))
        else:
            self.console.print(f"{ts}[bold]Claude:[/bold] {text}")

    def _handle_thinking(self, event: ThinkingEvent) -> None:
        """Handle Claude thinking event."""
        if not self.config.show_thinking:
            return

        if self.config.verbosity == OutputVerbosity.QUIET:
            return

        thinking = event.thinking.strip()
        if not thinking:
            return

        # Update status if we have a live spinner
        if self._is_processing:
            # Show abbreviated thinking in spinner
            short_thinking = thinking[:50] + "..." if len(thinking) > 50 else thinking
            self.update_status(f"Thinking: {short_thinking}")
        else:
            # Show full thinking
            ts = self._timestamp()
            if self.config.verbosity == OutputVerbosity.VERBOSE:
                self.console.print(f"{ts}[dim italic]Thinking: {thinking}[/dim italic]")
            else:
                # Truncate for normal verbosity
                display = thinking[:200] + "..." if len(thinking) > 200 else thinking
                self.console.print(f"{ts}[dim italic]Thinking: {display}[/dim italic]")

    def _handle_tool_use(self, event: ToolUseEvent) -> None:
        """Handle tool use event."""
        if self.config.verbosity == OutputVerbosity.QUIET:
            return

        # Store for matching with result
        self._pending_tool_use[event.tool_use_id] = event

        ts = self._timestamp()
        tool_detail = self._format_tool_detail(event.tool_name, event.tool_input)

        # Color based on tool type
        tool_colors = {
            "Read": "blue",
            "Write": "green",
            "Edit": "yellow",
            "Bash": "magenta",
            "Glob": "cyan",
            "Grep": "cyan",
            "WebFetch": "blue",
            "WebSearch": "blue",
            "Task": "yellow",
            "TodoWrite": "dim",
        }
        color = tool_colors.get(event.tool_name, "white")

        self.console.print(f"{ts}[{color}]▶ {event.tool_name}[/{color}] {tool_detail}")

        # For Write/Edit, cache file content for diff
        if event.tool_name in ("Write", "Edit"):
            file_path = event.tool_input.get("file_path", "")
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        self._last_file_contents[file_path] = f.read()
                except Exception:
                    pass

        # Show file content preview for Write
        if event.tool_name == "Write" and self.config.show_file_changes:
            content = event.tool_input.get("content", "")
            file_path = event.tool_input.get("file_path", "")
            if content:
                self._show_file_content(file_path, content, is_new=True)

        # Start processing spinner for potentially slow operations
        if event.tool_name in ("Bash", "WebFetch", "WebSearch", "Task"):
            self.start_processing(f"Running {event.tool_name}...")

    def _handle_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool result event."""
        self.stop_processing()

        if self.config.verbosity == OutputVerbosity.QUIET:
            return

        # Get the corresponding tool use
        tool_use = self._pending_tool_use.pop(event.tool_use_id, None)
        tool_name = tool_use.tool_name if tool_use else "Unknown"

        ts = self._timestamp()

        # Show error/success status
        if event.is_error:
            self.console.print(f"{ts}  [red]✗ Error:[/red] {event.content[:200]}")
        else:
            # Show result based on tool type
            if tool_name in ("Write", "Edit"):
                self._handle_file_change_result(tool_use, event)
            elif tool_name == "Bash":
                self._handle_bash_result(tool_use, event)
            elif tool_name in ("Read", "Glob", "Grep"):
                self._handle_read_result(tool_name, event)
            else:
                # Generic result
                if event.content and self.config.show_tool_output:
                    preview = event.content[:100]
                    if len(event.content) > 100:
                        preview += "..."
                    self.console.print(f"{ts}  [green]✓[/green] {preview}")
                else:
                    self.console.print(f"{ts}  [green]✓[/green]")

    def _handle_file_change_result(
        self, tool_use: ToolUseEvent | None, event: ToolResultEvent
    ) -> None:
        """Handle Write/Edit tool result."""
        if not tool_use:
            return

        ts = self._timestamp()
        file_path = tool_use.tool_input.get("file_path", "")

        # Track the file change
        old_content = self._last_file_contents.pop(file_path, None)
        new_content = None

        if tool_use.tool_name == "Write":
            new_content = tool_use.tool_input.get("content", "")
            operation = "write"
        else:  # Edit
            operation = "edit"
            # Read the new content after edit
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        new_content = f.read()
                except Exception:
                    pass

        change = FileChange(
            file_path=file_path,
            operation=operation,
            old_content=old_content,
            new_content=new_content,
        )
        self._file_changes.append(change)

        # Display the change
        if change.is_new_file:
            lines = len(new_content.splitlines()) if new_content else 0
            self.console.print(f"{ts}  [green]✓ Created[/green] {file_path} ({lines} lines)")
        else:
            added = change.lines_added
            removed = change.lines_removed
            self.console.print(
                f"{ts}  [green]✓ Modified[/green] {file_path} "
                f"([green]+{added}[/green]/[red]-{removed}[/red] lines)"
            )

        # Show diff for edits if verbose
        if (
            operation == "edit"
            and self.config.verbosity == OutputVerbosity.VERBOSE
            and old_content
            and new_content
        ):
            self._show_diff(file_path, old_content, new_content)

    def _handle_bash_result(self, tool_use: ToolUseEvent | None, event: ToolResultEvent) -> None:
        """Handle Bash tool result."""
        ts = self._timestamp()

        if not event.content:
            self.console.print(f"{ts}  [green]✓[/green] (no output)")
            return

        if not self.config.show_tool_output:
            self.console.print(f"{ts}  [green]✓[/green]")
            return

        lines = event.content.strip().splitlines()
        max_lines = self.config.max_tool_output_lines

        if len(lines) <= 3:
            # Short output - show inline
            for line in lines:
                self.console.print(f"{ts}  [dim]{escape(line)}[/dim]")
        elif len(lines) <= max_lines:
            # Medium output - show in box
            self.console.print(f"{ts}  [dim]Output ({len(lines)} lines):[/dim]")
            for line in lines:
                self.console.print(f"    [dim]{escape(line)}[/dim]")
        else:
            # Long output - truncate
            self.console.print(f"{ts}  [dim]Output ({len(lines)} lines, truncated):[/dim]")
            for line in lines[:max_lines]:
                self.console.print(f"    [dim]{escape(line)}[/dim]")
            self.console.print(f"    [dim]... ({len(lines) - max_lines} more lines)[/dim]")

    def _handle_read_result(self, tool_name: str, event: ToolResultEvent) -> None:
        """Handle Read/Glob/Grep result."""
        ts = self._timestamp()

        if not event.content:
            self.console.print(f"{ts}  [green]✓[/green]")
            return

        lines = event.content.strip().splitlines()

        if tool_name == "Glob":
            # Show file count
            self.console.print(f"{ts}  [green]✓[/green] Found {len(lines)} files")
        elif tool_name == "Grep":
            # Show match count
            self.console.print(f"{ts}  [green]✓[/green] {len(lines)} matches")
        else:
            # Read - show line count
            self.console.print(f"{ts}  [green]✓[/green] {len(lines)} lines")

    def _handle_question(self, event: AskQuestionEvent) -> None:
        """Handle question event."""
        self.console.print()
        ts = self._timestamp()

        for q in event.questions:
            self.console.print(f"{ts}[bold yellow]❓ Question:[/bold yellow] {q.question}")
            for i, opt in enumerate(q.options, 1):
                desc = f" - {opt.description}" if opt.description else ""
                self.console.print(f"       [{i}] {opt.label}{desc}")

        self.console.print()

    def _handle_rate_limit(self, event: RateLimitEvent) -> None:
        """Handle rate limit event."""
        ts = self._timestamp()
        self.console.print(
            f"{ts}[bold red]⚠ Rate Limited[/bold red] "
            f"Retry after: {event.retry_after}s - {event.message}"
        )

    def _handle_result(self, event: ResultEvent) -> None:
        """Handle session result event."""
        self.stop_processing()
        self.console.print()

        ts = self._timestamp()

        if event.is_error:
            self.console.print(
                f"{ts}[bold red]✗ Session ended with error:[/bold red] {event.result}"
            )
        else:
            self.console.print(f"{ts}[bold green]✓ Session completed[/bold green]")

            # Show result summary
            if event.result:
                result_text = event.result
                if len(result_text) > 500 and self.config.verbosity != OutputVerbosity.VERBOSE:
                    result_text = result_text[:500] + "..."
                self.console.print(f"       Result: {result_text}")

        # Show cost and turns
        if event.total_cost_usd is not None:
            cost = f"${event.total_cost_usd:.4f}"
            self.console.print(f"       Cost: {cost} | Turns: {event.num_turns}")

        # Show file changes summary
        if self._file_changes:
            self._show_file_changes_summary()

    def _show_file_content(
        self,
        file_path: str,
        content: str,
        is_new: bool = False,
    ) -> None:
        """Show file content with syntax highlighting."""
        if not self.config.show_file_changes:
            return

        lines = content.splitlines()
        max_lines = self.config.max_file_preview_lines

        if len(lines) > max_lines:
            # Truncate content
            truncated = "\n".join(lines[:max_lines])
            truncated += f"\n... ({len(lines) - max_lines} more lines)"
            content = truncated

        lexer = self._get_file_extension(file_path)
        syntax = Syntax(
            content,
            lexer,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )

        title = "[green]New file[/green]" if is_new else "[yellow]Modified[/yellow]"
        self.console.print(
            Panel(
                syntax,
                title=f"{title}: {Path(file_path).name}",
                border_style="green" if is_new else "yellow",
            )
        )

    def _show_diff(self, file_path: str, old_content: str, new_content: str) -> None:
        """Show a unified diff of changes."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{Path(file_path).name}",
                tofile=f"b/{Path(file_path).name}",
                lineterm="",
            )
        )

        if not diff:
            return

        # Build colored diff output
        diff_text = Text()
        for line in diff[:100]:  # Limit diff lines
            line = line.rstrip("\n")
            if line.startswith("+") and not line.startswith("+++"):
                diff_text.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                diff_text.append(line + "\n", style="red")
            elif line.startswith("@@"):
                diff_text.append(line + "\n", style="cyan")
            else:
                diff_text.append(line + "\n", style="dim")

        if len(diff) > 100:
            diff_text.append(f"... ({len(diff) - 100} more lines)\n", style="dim")

        self.console.print(
            Panel(
                diff_text,
                title=f"[yellow]Diff: {Path(file_path).name}[/yellow]",
                border_style="yellow",
            )
        )

    def _show_file_changes_summary(self) -> None:
        """Show summary of all file changes."""
        if not self._file_changes:
            return

        self.console.print()
        self.console.print(f"[bold]Files modified ({len(self._file_changes)}):[/bold]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Op", justify="center")
        table.add_column("+", justify="right", style="green")
        table.add_column("-", justify="right", style="red")

        for change in self._file_changes:
            op_style = "green" if change.is_new_file else "yellow"
            op_text = "NEW" if change.is_new_file else "MOD"
            table.add_row(
                Path(change.file_path).name,
                f"[{op_style}]{op_text}[/{op_style}]",
                str(change.lines_added),
                str(change.lines_removed),
            )

        self.console.print(table)

    def _format_tool_detail(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Format tool input for display."""
        match tool_name:
            case "Read":
                path = tool_input.get("file_path", "?")
                return f"[dim]{Path(path).name}[/dim]"
            case "Write":
                path = tool_input.get("file_path", "?")
                content = tool_input.get("content", "")
                lines = len(content.splitlines())
                return f"[dim]{Path(path).name}[/dim] ({lines} lines)"
            case "Edit":
                path = tool_input.get("file_path", "?")
                old = tool_input.get("old_string", "")
                new = tool_input.get("new_string", "")
                return f"[dim]{Path(path).name}[/dim] ({len(old)}→{len(new)} chars)"
            case "Bash":
                cmd = tool_input.get("command", "?")
                if len(cmd) > 60:
                    cmd = cmd[:60] + "..."
                return f"[dim]$ {cmd}[/dim]"
            case "Glob":
                return f"[dim]pattern={tool_input.get('pattern', '?')}[/dim]"
            case "Grep":
                return f"[dim]/{tool_input.get('pattern', '?')}/[/dim]"
            case "WebFetch":
                url = tool_input.get("url", "?")
                if len(url) > 50:
                    url = url[:50] + "..."
                return f"[dim]{url}[/dim]"
            case "WebSearch":
                return f'[dim]"{tool_input.get("query", "?")}"[/dim]'
            case "Task":
                desc = tool_input.get("description", "?")
                if len(desc) > 40:
                    desc = desc[:40] + "..."
                return f"[dim]{desc}[/dim]"
            case "TodoWrite":
                todos = tool_input.get("todos", [])
                return f"[dim]{len(todos)} items[/dim]"
            case _:
                return ""

    def show_summary(
        self,
        success: bool,
        turns: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Show final session summary.

        Args:
            success: Whether session completed successfully
            turns: Number of turns
            cost_usd: Total cost in USD
        """
        self.console.print()

        if success:
            status = "[bold green]Session Completed[/bold green]"
        else:
            status = "[bold red]Session Failed[/bold red]"
        border = "green" if success else "red"

        details = []
        if turns:
            details.append(f"Turns: {turns}")
        if cost_usd:
            details.append(f"Cost: ${cost_usd:.4f}")
        if self._file_changes:
            details.append(f"Files: {len(self._file_changes)}")

        detail_str = " | ".join(details) if details else ""

        self.console.print(
            Panel(
                f"{status}\n{detail_str}" if detail_str else status,
                border_style=border,
            )
        )

    @property
    def file_changes(self) -> list[FileChange]:
        """Get list of file changes from this session."""
        return self._file_changes.copy()
