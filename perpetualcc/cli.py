"""CLI commands for PerpetualCC."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from perpetualcc import __version__
from perpetualcc.claude.adapter import ClaudeCodeAdapter
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
from perpetualcc.cli_output import OutputConfig, OutputVerbosity, SessionOutputManager
from perpetualcc.core.rate_limit import RateLimitDetector, RateLimitMonitor
from perpetualcc.core.session_manager import (
    ManagedSession,
    SessionManager,
    SessionManagerConfig,
    SessionStatus,
)
from perpetualcc.core.task_queue import TaskStatus

# Configure logging to suppress SDK internal errors that don't affect functionality
logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)


def _suppress_sdk_cancel_scope_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Custom exception handler to suppress known SDK cancel scope errors.

    The Claude Agent SDK has a known issue where async generator cleanup
    happens in a background task, causing "cancel scope in different task"
    errors. These don't affect functionality but pollute the output.

    Args:
        loop: The event loop
        context: Exception context dict with 'message' and optionally 'exception'
    """
    exception = context.get("exception")
    message = context.get("message", "")

    # Check if this is the known SDK cancel scope error
    if exception and isinstance(exception, RuntimeError):
        error_str = str(exception)
        if "cancel scope in a different task" in error_str:
            # Suppress this specific error - it's a known SDK issue
            return

    # For all other exceptions, use the default handler
    loop.default_exception_handler(context)


# Global output manager for use across commands
_output_manager: SessionOutputManager | None = None


def _get_output_manager(
    quiet: bool = False,
    verbose: bool = False,
) -> SessionOutputManager:
    """Get or create the global output manager."""
    global _output_manager
    if _output_manager is None or quiet or verbose:
        if quiet:
            config = OutputConfig.quiet()
        elif verbose:
            config = OutputConfig.verbose()
        else:
            config = OutputConfig()
        _output_manager = SessionOutputManager(config=config, console=console)
    return _output_manager


# Rich help formatting
MAIN_HELP = """
[bold cyan]PerpetualCC[/] - Your AI coding companion that runs 24/7

PerpetualCC supervises Claude Code sessions automatically. It handles permissions,
answers routine questions, manages rate limits, and knows when to ask for your help.

[bold yellow]Quick Start:[/]
  pcc start ./myproject -t "Add user authentication"
  pcc start . -r requirements.md

[bold yellow]Common Workflows:[/]
  [dim]Start a new session:[/]     pcc start <path> -t "your task"
  [dim]Check on sessions:[/]       pcc list
  [dim]Resume after a break:[/]    pcc resume <session-id>
  [dim]Respond to questions:[/]    pcc pending && pcc respond 1

[bold yellow]Session Lifecycle:[/]
  start -> (pause/attach) -> resume -> stop -> delete

Run [bold]pcc <command> --help[/] for detailed help on any command.
"""

app = typer.Typer(
    name="pcc",
    help=MAIN_HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Global session manager instance (lazy initialized)
_session_manager: SessionManager | None = None


def _ensure_config_exists() -> None:
    """Ensure config file exists, creating default if needed.

    This is called automatically when PCC commands run to provide
    a seamless first-run experience.
    """
    from perpetualcc.config import (
        DEFAULT_CONFIG_PATH,
        ensure_config_dir,
        generate_default_config,
    )

    if not DEFAULT_CONFIG_PATH.exists():
        ensure_config_dir()
        config_content = generate_default_config()
        DEFAULT_CONFIG_PATH.write_text(config_content)
        console.print(f"[dim]Created default config: {DEFAULT_CONFIG_PATH}[/dim]")
        console.print("[dim]Customize with: pcc config show[/dim]")
        console.print()


def _get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def _timestamp() -> str:
    """Get a formatted timestamp for log output."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def _print_event(event: ClaudeEvent, output_manager: SessionOutputManager | None = None) -> None:
    """Print a Claude event to the console.

    Args:
        event: The event to print
        output_manager: Optional output manager for enhanced display
    """
    if output_manager:
        # Use the enhanced output manager
        output_manager.handle_event(event)
        return

    # Fallback to basic output (for backwards compatibility)
    ts = f"[dim][{_timestamp()}][/dim]"

    if isinstance(event, InitEvent):
        console.print(f"{ts} Session started: [cyan]{event.session_id}[/cyan]")

    elif isinstance(event, TextEvent):
        if event.text.strip():
            console.print(f"{ts} [bold]Claude:[/bold] {event.text.strip()}")

    elif isinstance(event, ThinkingEvent):
        if event.thinking.strip():
            # Show truncated thinking
            thinking = event.thinking.strip()
            if len(thinking) > 200:
                thinking = thinking[:200] + "..."
            console.print(f"{ts} [dim italic]Thinking: {thinking}[/dim italic]")

    elif isinstance(event, ToolUseEvent):
        tool_detail = _format_tool_detail(event.tool_name, event.tool_input)
        console.print(f"{ts} [yellow][TOOL][/yellow] {event.tool_name} {tool_detail}")

    elif isinstance(event, ToolResultEvent):
        status = "[red]ERROR[/red]" if event.is_error else "[green]OK[/green]"
        content_preview = event.content[:100] + "..." if len(event.content) > 100 else event.content
        if content_preview.strip():
            console.print(f"{ts}   {status} {content_preview.strip()}")

    elif isinstance(event, AskQuestionEvent):
        console.print()
        for q in event.questions:
            console.print(f"{ts} [bold red]QUESTION:[/bold red] {q.question}")
            for i, opt in enumerate(q.options, 1):
                desc = f" - {opt.description}" if opt.description else ""
                console.print(f"       [{i}] {opt.label}{desc}")
        console.print()

    elif isinstance(event, RateLimitEvent):
        console.print(
            f"{ts} [bold red]RATE LIMITED[/bold red] "
            f"Retry after: {event.retry_after}s - {event.message}"
        )

    elif isinstance(event, ResultEvent):
        console.print()
        if event.is_error:
            console.print(f"{ts} [bold red]Session ended with error:[/bold red] {event.result}")
        else:
            result_text = event.result or "Task completed"
            if len(result_text) > 500:
                result_text = result_text[:500] + "..."
            console.print(f"{ts} [bold green]Session completed[/bold green]")
            if event.result:
                console.print(f"       Result: {result_text}")
        if event.total_cost_usd is not None:
            console.print(f"       Cost: ${event.total_cost_usd:.4f} | Turns: {event.num_turns}")


def _format_tool_detail(tool_name: str, tool_input: dict) -> str:
    """Format tool input for display."""
    match tool_name:
        case "Read":
            return f"-> {tool_input.get('file_path', '?')}"
        case "Write":
            return f"-> {tool_input.get('file_path', '?')}"
        case "Edit":
            return f"-> {tool_input.get('file_path', '?')}"
        case "Bash":
            cmd = tool_input.get("command", "?")
            if len(cmd) > 80:
                cmd = cmd[:80] + "..."
            return f"$ {cmd}"
        case "Glob":
            return f"pattern={tool_input.get('pattern', '?')}"
        case "Grep":
            return f"/{tool_input.get('pattern', '?')}/"
        case "WebFetch":
            return f"-> {tool_input.get('url', '?')}"
        case "WebSearch":
            return f'"{tool_input.get("query", "?")}"'
        case _:
            return ""


def _format_status(status: SessionStatus) -> str:
    """Format session status with color."""
    match status:
        case SessionStatus.IDLE:
            return "[dim]Idle[/dim]"
        case SessionStatus.PROCESSING:
            return "[green]Running[/green]"
        case SessionStatus.WAITING_INPUT:
            return "[yellow]Waiting[/yellow]"
        case SessionStatus.RATE_LIMITED:
            return "[red]Rate Limited[/red]"
        case SessionStatus.PAUSED:
            return "[yellow]Paused[/yellow]"
        case SessionStatus.COMPLETED:
            return "[green]Completed[/green]"
        case SessionStatus.ERROR:
            return "[red]Error[/red]"
        case _:
            return str(status.value)


def _format_task_status(status: TaskStatus) -> str:
    """Format task status with color."""
    match status:
        case TaskStatus.PENDING:
            return "[dim]Pending[/dim]"
        case TaskStatus.IN_PROGRESS:
            return "[green]Running[/green]"
        case TaskStatus.COMPLETED:
            return "[green]Done[/green]"
        case TaskStatus.FAILED:
            return "[red]Failed[/red]"
        case TaskStatus.CANCELLED:
            return "[yellow]Cancelled[/yellow]"
        case _:
            return str(status.value)


def _truncate(text: str | None, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return "-"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _event_callback(session_id: str, event: ClaudeEvent) -> None:
    """Event callback for session manager to print events."""
    _print_event(event)


async def _run_session(
    project_path: Path,
    task: str,
    model: str | None,
    max_turns: int | None,
    quiet: bool = False,
    verbose: bool = False,
) -> None:
    """Run a Claude Code session and stream output.

    Args:
        project_path: Path to the project
        task: Task description
        model: Optional model name
        max_turns: Optional max turns limit
        quiet: Minimal output mode
        verbose: Verbose output mode
    """
    # Install custom exception handler to suppress SDK cancel scope errors
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_suppress_sdk_cancel_scope_errors)

    adapter = ClaudeCodeAdapter(
        project_path=project_path,
        model=model,
        max_turns=max_turns,
    )
    rate_monitor = RateLimitMonitor()
    rate_detector = RateLimitDetector()

    # Create output manager with appropriate verbosity
    output_manager = _get_output_manager(quiet=quiet, verbose=verbose)

    # Show session header
    output_manager.start_session(
        session_id="pending",
        project_path=project_path,
        task=task,
    )

    session_id: str | None = None
    current_task = task
    result_event: ResultEvent | None = None

    while True:
        try:
            # Show connecting status
            output_manager.start_processing("Connecting to Claude...")

            # Connect and stream events using ClaudeSDKClient
            await adapter.connect(current_task)

            # Update status
            output_manager.update_status("Waiting for response...")

            rate_limit_info = None

            async for event in adapter.receive_events():
                _print_event(event, output_manager)

                # Capture session ID for potential resume
                if isinstance(event, InitEvent):
                    session_id = event.session_id

                # Capture result for summary
                if isinstance(event, ResultEvent):
                    result_event = event
                    # Check for rate limit in result
                    if event.is_error:
                        rate_limit_info = rate_detector.detect_from_message(event.result or "")

                # Check for explicit rate limit event
                if isinstance(event, RateLimitEvent):
                    rate_limit_info = rate_monitor.detect(event)

            # Ensure processing stopped
            output_manager.stop_processing()

            # Disconnect after streaming completes
            await adapter.disconnect()

            # If we got a rate limit, wait and retry
            if rate_limit_info:
                await _wait_for_rate_limit_reset(rate_limit_info)
                # Continue the loop to resume the session
                continue

            # Session completed normally - show summary
            if result_event:
                output_manager.show_summary(
                    success=not result_event.is_error,
                    turns=result_event.num_turns,
                    cost_usd=result_event.total_cost_usd,
                )

            break

        except KeyboardInterrupt:
            output_manager.stop_processing()
            console.print("\n[yellow]Session interrupted by user.[/yellow]")
            if adapter.session_id:
                console.print(
                    f"Resume with: pcc start {project_path} --resume {adapter.session_id}"
                )
            await adapter.disconnect()
            break

        except Exception as e:
            output_manager.stop_processing()
            error_msg = str(e)
            # Check if exception message indicates rate limit
            rate_limit_info = rate_detector.detect_from_message(error_msg)
            if rate_limit_info:
                await adapter.disconnect()
                await _wait_for_rate_limit_reset(rate_limit_info)
                # Continue the loop to resume the session
                continue

            # Ensure disconnect on error
            await adapter.disconnect()
            # Re-raise non-rate-limit exceptions
            raise


async def _wait_for_rate_limit_reset(rate_limit_info) -> None:
    """Wait for rate limit reset with progress display."""
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    reset_time = rate_limit_info.reset_time
    retry_seconds = rate_limit_info.retry_after_seconds

    console.print()
    console.print(f"[bold yellow]Rate limit detected:[/bold yellow] {rate_limit_info.message}")
    if reset_time:
        console.print(
            f"[dim]Reset time: {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(waiting {retry_seconds} seconds)[/dim]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        wait_task = progress.add_task(
            "[yellow]Waiting for rate limit reset...",
            total=retry_seconds,
        )

        for i in range(retry_seconds):
            await asyncio.sleep(1)
            progress.update(wait_task, advance=1)

    console.print("[green]Rate limit reset. Resuming session...[/green]")
    console.print()


async def _run_managed_session(
    session: ManagedSession,
    quiet: bool = False,
    verbose: bool = False,
) -> None:
    """Run a managed session with the session manager.

    Uses interactive mode with proper signal handling for pause/stop.

    Args:
        session: The managed session to run
        quiet: Minimal output mode
        verbose: Verbose output mode
    """
    # Install custom exception handler to suppress SDK cancel scope errors
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_suppress_sdk_cancel_scope_errors)

    from perpetualcc.human import InteractivePrompt, SignalHandler, UserAction

    manager = _get_session_manager()

    # Create output manager with appropriate verbosity
    output_manager = _get_output_manager(quiet=quiet, verbose=verbose)

    # Create event callback that uses our output manager
    def managed_event_callback(session_id: str, event: ClaudeEvent) -> None:
        _print_event(event, output_manager)

    manager.event_callback = managed_event_callback

    # Show session header using output manager
    output_manager.start_session(session.id, session.project_path, session.current_task or "")

    prompt = InteractivePrompt()

    # Set up signal handler for pause/stop options
    action_result = UserAction.CONTINUE

    def on_interrupt() -> UserAction:
        nonlocal action_result
        action_result = prompt.prompt_pause_or_continue()
        return action_result

    handler = SignalHandler(on_interrupt=on_interrupt)
    handler.setup()

    try:
        await manager.start_session(session.id)

        # Wait for session to complete or be interrupted
        while True:
            updated = manager.get_session(session.id)
            if not updated or updated.is_completed:
                break

            if action_result == UserAction.PAUSE:
                await manager.pause_session(session.id)
                console.print("[yellow]Session paused.[/yellow]")
                console.print(f"Resume with: pcc resume {session.id[:8]}")
                return

            if action_result == UserAction.STOP:
                await manager.stop_session(session.id)
                console.print("[red]Session stopped.[/red]")
                return

            await asyncio.sleep(0.5)

        # Show session summary using output manager
        if updated:
            output_manager.show_summary(
                success=updated.status == SessionStatus.COMPLETED,
                turns=updated.turn_count,
                cost_usd=updated.total_cost_usd,
            )

    finally:
        output_manager.stop_processing()
        handler.restore()


@app.command()
def start(
    project_path: Path = typer.Argument(
        ...,
        help="Project folder to work in (created if needed)",
        file_okay=False,
        resolve_path=True,
        metavar="PATH",
    ),
    task: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help='What you want Claude to do. Can specify multiple: -t "Task 1" -t "Task 2"',
        metavar="DESCRIPTION",
    ),
    tasks_file: list[Path] | None = typer.Option(
        None,
        "--tasks-file",
        "-f",
        help="Load tasks from file(s). Can specify multiple: -f reqs1.md -f reqs2.md",
        exists=True,
        dir_okay=False,
        metavar="FILE",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Claude Code model: sonnet (balanced), opus (smartest), haiku (fastest)",
        metavar="NAME",
    ),
    simple: bool = typer.Option(
        False,
        "--simple",
        help="Disable pause/resume features (run in simple non-interactive mode)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Minimal output - only errors and final result",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output - show file contents, diffs, full tool output",
    ),
    plan_first: bool = typer.Option(
        False,
        "--plan-first",
        help="Run planning session first, review with Brain before execution",
    ),
    human_review: bool = typer.Option(
        False,
        "--human-review",
        help="Force human review for each plan iteration (requires --plan-first)",
    ),
) -> None:
    """
    Start a new Claude Code session.

    Claude will work on your task autonomously. PerpetualCC handles routine
    decisions and only asks you when something important needs your input.

    [bold yellow]Examples:[/]

      [dim]# Simple task - just tell Claude what to do[/]
      pcc start ./myapp -t "Add a dark mode toggle"

      [dim]# Multiple tasks - runs concurrently[/]
      pcc start ./myapp -t "Add tests" -t "Fix bug #123"

      [dim]# Use requirements files[/]
      pcc start ./myapp -f requirements.md
      pcc start ./myapp -f reqs1.md -f reqs2.md

      [dim]# Mix inline tasks and files[/]
      pcc start ./myapp -t "Code review" -f bugfixes.md

    [bold yellow]Tips:[/]

      - Use [bold].[/] for current directory: pcc start . -t "..."
      - Press [bold]Ctrl+C[/] to pause or stop the session
      - Multiple tasks run as separate sessions
    """
    # Ensure config file exists (auto-init on first run)
    _ensure_config_exists()

    # Create project directory if it doesn't exist
    if not project_path.exists():
        try:
            project_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created project directory:[/green] {project_path}")
        except OSError as e:
            console.print(f"[red]Error:[/red] Could not create directory: {e}")
            raise typer.Exit(1)
    elif not project_path.is_dir():
        console.print(f"[red]Error:[/red] Path exists but is not a directory: {project_path}")
        raise typer.Exit(1)

    # Collect all tasks from options
    all_tasks: list[str] = []
    
    # Add inline tasks
    if task:
        all_tasks.extend(task)
    
    # Add tasks from files
    if tasks_file:
        for file_path in tasks_file:
            try:
                file_content = file_path.read_text().strip()
                if file_content:
                    all_tasks.append(file_content)
            except OSError as e:
                console.print(f"[red]Error reading {file_path}:[/red] {e}")
                raise typer.Exit(1)
    
    if not all_tasks:
        console.print("[red]Error:[/red] Either --task or --tasks-file must be provided.")
        raise typer.Exit(1)

    # Validate quiet and verbose are mutually exclusive
    if quiet and verbose:
        console.print("[red]Error:[/red] --quiet and --verbose cannot be used together.")
        raise typer.Exit(1)

    # Validate --human-review requires --plan-first
    if human_review and not plan_first:
        console.print("[red]Error:[/red] --human-review requires --plan-first flag.")
        raise typer.Exit(1)

    try:
        if simple:
            # Simple mode: no pause/resume, no session management
            # Run all tasks sequentially in simple mode
            for task_desc in all_tasks:
                asyncio.run(_run_session(project_path, task_desc, model, None, quiet, verbose))
        else:
            # Default: managed mode with pause/resume and session tracking
            manager = _get_session_manager()
            
            if len(all_tasks) == 1:
                # Single task: run with interactive mode
                session = asyncio.run(manager.create_session(project_path, all_tasks[0]))
                asyncio.run(_run_managed_session(session, quiet, verbose))
            else:
                # Multiple tasks: create sessions and run concurrently
                console.print(f"[bold]Starting {len(all_tasks)} sessions...[/bold]")
                
                async def run_multiple_sessions():
                    # Create all sessions first
                    sessions = []
                    for i, task_desc in enumerate(all_tasks, 1):
                        session = await manager.create_session(project_path, task_desc)
                        sessions.append(session)
                        console.print(f"  [{i}/{len(all_tasks)}] Created session {session.id[:8]}")
                    
                    # Run sessions concurrently (without interactive signal handling)
                    console.print("[bold]Running sessions...[/bold]")
                    for session in sessions:
                        await manager.start_session(session.id)
                    
                    # Wait for all sessions to complete
                    while True:
                        all_done = True
                        for session in sessions:
                            updated = manager.get_session(session.id)
                            if updated and not updated.is_completed:
                                all_done = False
                                break
                        if all_done:
                            break
                        await asyncio.sleep(0.5)
                    
                    # Show summary for all sessions
                    console.print("\n[bold]Session Summary:[/bold]")
                    for session in sessions:
                        updated = manager.get_session(session.id)
                        if updated:
                            is_done = updated.status == SessionStatus.COMPLETED
                            status_color = "green" if is_done else "red"
                            task_preview = session.current_task[:40]
                            console.print(
                                f"  {session.id[:8]}: "
                                f"[{status_color}]{updated.status.value}[/{status_color}] "
                                f"- {task_preview}..."
                            )
                
                asyncio.run(run_multiple_sessions())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting.[/yellow]")


@app.command(name="list")
def list_sessions(
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter: idle, processing, paused, rate_limited, completed, error",
        metavar="STATUS",
    ),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Show only sessions for this project folder",
        metavar="PATH",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Include finished sessions (completed and errored)",
    ),
) -> None:
    """
    Show all your sessions at a glance.

    By default, shows only active sessions. Use -a to see completed ones too.

    [bold yellow]Examples:[/]

      [dim]# See what's running right now[/]
      pcc list

      [dim]# See all sessions including finished[/]
      pcc list --all

      [dim]# Filter by status[/]
      pcc list -s paused
      pcc list -s rate_limited

    [bold yellow]Status meanings:[/]
      idle          - Created but not started yet
      processing    - Actively working on a task
      paused        - Waiting for you to resume
      rate_limited  - Waiting for API cooldown
      completed     - Finished successfully
      error         - Something went wrong
    """
    manager = _get_session_manager()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid status: {status}")
            raise typer.Exit(1)

    sessions = manager.list_sessions(
        status=status_filter,
        project_path=str(project) if project else None,
    )

    # Filter out completed/error unless --all
    if not show_all:
        sessions = [s for s in sessions if not s.is_completed]

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    # Create table
    table = Table(title="Managed Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Project", style="dim")
    table.add_column("Current Task")
    table.add_column("Tasks", justify="right")
    table.add_column("Cost", justify="right")

    for session in sessions:
        # Get task count
        pending = manager.task_queue.count(session.id, TaskStatus.PENDING)
        in_progress = manager.task_queue.count(session.id, TaskStatus.IN_PROGRESS)
        task_count = f"{in_progress}/{pending + in_progress}"

        # Format cost
        cost = f"${session.total_cost_usd:.4f}" if session.total_cost_usd else "-"

        # Truncate project path
        project_display = Path(session.project_path).name

        table.add_row(
            session.id[:8],
            _format_status(session.status),
            project_display,
            _truncate(session.current_task, 40),
            task_count,
            cost,
        )

    console.print(table)


@app.command()
def status(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list' (first 8 chars is enough)",
        metavar="SESSION",
    ),
    show_tasks: bool = typer.Option(
        False,
        "--tasks",
        "-t",
        help="Also show the task queue",
    ),
) -> None:
    """
    Get detailed info about a specific session.

    Shows timing, cost, current task, and any errors. Use the short ID
    from 'pcc list' - you don't need the full ID.

    [bold yellow]Examples:[/]

      [dim]# Check on a session (use ID from 'pcc list')[/]
      pcc status a1b2c3d4

      [dim]# See the task queue too[/]
      pcc status a1b2c3d4 --tasks
    """
    manager = _get_session_manager()

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    # Build status panel
    status_lines = [
        f"[bold]Session ID:[/bold] {session.id}",
        f"[bold]Claude Session:[/bold] {session.claude_session_id or 'Not started'}",
        f"[bold]Status:[/bold] {_format_status(session.status)}",
        f"[bold]Project:[/bold] {session.project_path}",
        f"[bold]Current Task:[/bold] {session.current_task or '-'}",
        "",
        f"[bold]Created:[/bold] {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"[bold]Started:[/bold] {session.started_at.strftime('%Y-%m-%d %H:%M:%S') if session.started_at else '-'}",
        f"[bold]Last Activity:[/bold] {session.last_activity.strftime('%Y-%m-%d %H:%M:%S') if session.last_activity else '-'}",
        "",
        f"[bold]Turns:[/bold] {session.turn_count}",
        f"[bold]Cost:[/bold] ${session.total_cost_usd:.4f}"
        if session.total_cost_usd
        else "[bold]Cost:[/bold] -",
    ]

    if session.error_message:
        status_lines.append(f"\n[bold red]Error:[/bold red] {session.error_message}")

    if session.rate_limit_info:
        status_lines.append(
            f"\n[bold yellow]Rate Limited:[/bold yellow] {session.rate_limit_info.message}"
        )

    console.print(
        Panel(
            Text.from_markup("\n".join(status_lines)),
            title=f"[bold cyan]Session {session.id[:8]}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Show tasks if requested
    if show_tasks:
        tasks = manager.task_queue.list_tasks(session.id)
        if tasks:
            console.print()
            table = Table(title="Task Queue")
            table.add_column("ID", style="dim", no_wrap=True)
            table.add_column("Status", no_wrap=True)
            table.add_column("Priority", justify="right")
            table.add_column("Description")

            for task in tasks:
                table.add_row(
                    task.task_id[:8],
                    _format_task_status(task.status),
                    str(task.priority),
                    _truncate(task.description, 50),
                )

            console.print(table)
        else:
            console.print("\n[dim]No tasks in queue.[/dim]")


@app.command()
def add(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
    task: str = typer.Argument(
        ...,
        help="What Claude should do next",
        metavar="TASK",
    ),
    priority: int = typer.Option(
        5,
        "--priority",
        "-p",
        help="1-10 (higher runs first). Default: 5",
        metavar="N",
    ),
) -> None:
    """
    Queue up another task for an existing session.

    Tasks run in priority order (higher number = sooner).
    Perfect for adding follow-up work while a session is running.

    [bold yellow]Examples:[/]

      [dim]# Add a follow-up task[/]
      pcc add a1b2c3d4 "Add unit tests for the new feature"

      [dim]# Add a high-priority task (runs next)[/]
      pcc add a1b2c3d4 "Fix critical bug" -p 10
    """
    manager = _get_session_manager()

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    try:
        new_task = asyncio.run(manager.add_task(session.id, task, priority))
        console.print(f"[green]Added task:[/green] {new_task.task_id[:8]} (priority={priority})")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def pause(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
) -> None:
    """
    Pause a running session to pick it up later.

    The session state is saved so you can resume exactly where it left off.
    Useful when you need to step away or want to review progress.

    [bold yellow]Example:[/]
      pcc pause a1b2c3d4

    [bold yellow]Tip:[/] You can also press Ctrl+C while attached to a session.
    """
    manager = _get_session_manager()

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    if not session.is_active:
        console.print(f"[yellow]Session is not active:[/yellow] {session.status.value}")
        return

    try:
        asyncio.run(manager.pause_session(session.id))
        console.print(f"[green]Paused session:[/green] {session.id[:8]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def resume(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
) -> None:
    """
    Continue a paused session.

    Picks up exactly where the session left off.

    [bold yellow]Note:[/] Rate-limited sessions auto-resume when the limit resets.
    You only need this for sessions you manually paused.

    [bold yellow]Example:[/]
      pcc resume a1b2c3d4

    [bold yellow]Tip:[/] Use 'pcc list -s paused' to see sessions waiting to resume.
    """
    # Ensure config file exists
    _ensure_config_exists()

    manager = _get_session_manager()
    manager.event_callback = _event_callback

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    if not session.can_resume:
        console.print(f"[yellow]Session cannot be resumed:[/yellow] {session.status.value}")
        return

    console.print(f"[green]Resuming session:[/green] {session.id[:8]}")

    try:
        asyncio.run(_run_managed_session(session))
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")


@app.command()
def attach(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
) -> None:
    """
    Watch a session's progress in real-time.

    Connects to a running session so you can see what Claude is doing.
    Press Ctrl+C to pause or stop the session.

    [bold yellow]Example:[/]
      pcc attach a1b2c3d4

    [bold yellow]Controls:[/]
      Ctrl+C  - Opens pause/continue/stop menu
    """
    # Ensure config file exists
    _ensure_config_exists()

    from perpetualcc.human import InteractivePrompt, SignalHandler, UserAction

    manager = _get_session_manager()
    manager.event_callback = _event_callback

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    if not session.is_active:
        console.print(f"[yellow]Session is not active:[/yellow] {session.status.value}")
        console.print("Use 'pcc resume' to restart the session.")
        return

    prompt = InteractivePrompt()
    prompt.show_session_header(session.id, session.project_path, session.current_task)

    # Set up signal handler for pause/stop options
    action_result = UserAction.CONTINUE

    def on_interrupt() -> UserAction:
        nonlocal action_result
        action_result = prompt.prompt_pause_or_continue()
        return action_result

    handler = SignalHandler(on_interrupt=on_interrupt)
    handler.setup()

    try:
        # Wait for session to complete or be interrupted
        while True:
            updated = manager.get_session(session.id)
            if not updated or not updated.is_active:
                break

            if action_result == UserAction.PAUSE:
                asyncio.run(manager.pause_session(session.id))
                console.print("[yellow]Session paused.[/yellow]")
                console.print(f"Resume with: pcc resume {session.id[:8]}")
                break

            if action_result == UserAction.STOP:
                asyncio.run(manager.stop_session(session.id))
                console.print("[red]Session stopped.[/red]")
                break

            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))

        if updated and updated.is_completed:
            prompt.show_session_summary(
                success=updated.status == SessionStatus.COMPLETED,
                turns=updated.turn_count,
                cost_usd=updated.total_cost_usd,
            )

    finally:
        handler.restore()


@app.command()
def stop(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """
    Stop a session permanently.

    Ends the session and marks it as completed. Unlike pause, you can't
    continue from where it left off.

    [bold yellow]Examples:[/]

      [dim]# Stop with confirmation[/]
      pcc stop a1b2c3d4

      [dim]# Stop immediately without asking[/]
      pcc stop a1b2c3d4 --force
    """
    manager = _get_session_manager()

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    if session.is_completed:
        console.print(f"[yellow]Session already completed:[/yellow] {session.status.value}")
        return

    if not force:
        confirm = typer.confirm(f"Stop session {session.id[:8]}?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        asyncio.run(manager.stop_session(session.id))
        console.print(f"[green]Stopped session:[/green] {session.id[:8]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def delete(
    session_id: str = typer.Argument(
        ...,
        help="Session ID from 'pcc list'",
        metavar="SESSION",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """
    Remove a session completely.

    Deletes all session data including history, checkpoints, and task queue.
    Use this to clean up old sessions you no longer need.

    [bold yellow]Examples:[/]

      [dim]# Delete with confirmation[/]
      pcc delete a1b2c3d4

      [dim]# Delete immediately[/]
      pcc delete a1b2c3d4 --force
    """
    manager = _get_session_manager()

    # Find session by partial ID
    session = None
    for s in manager.list_sessions():
        if s.id.startswith(session_id):
            session = s
            break

    if not session:
        console.print(f"[red]Error:[/red] Session not found: {session_id}")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete session {session.id[:8]} and all associated data?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        asyncio.run(manager.delete_session(session.id))
        console.print(f"[green]Deleted session:[/green] {session.id[:8]}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """
    See overall usage statistics.

    Shows total sessions, costs, and breakdown by status.
    Useful for tracking usage and costs over time.

    [bold yellow]Example:[/]
      pcc stats
    """
    manager = _get_session_manager()
    stats = manager.get_statistics()

    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Total Sessions:[/bold] {stats['total_sessions']}\n"
                f"[bold]Active Sessions:[/bold] {stats['active_sessions']}\n"
                f"[bold]Total Cost:[/bold] ${stats['total_cost_usd']:.4f}\n"
                f"[bold]Total Turns:[/bold] {stats['total_turns']}\n"
                f"\n[bold]By Status:[/bold]\n"
                + "\n".join(f"  {status}: {count}" for status, count in stats["by_status"].items())
                + f"\n\n[bold]Task Queue:[/bold]\n"
                f"  Pending: {stats['task_queue']['pending']}\n"
                f"  In Progress: {stats['task_queue']['in_progress']}\n"
                f"  Completed: {stats['task_queue']['completed']}\n"
                f"  Failed: {stats['task_queue']['failed']}"
            ),
            title="[bold cyan]Session Manager Statistics[/bold cyan]",
            border_style="cyan",
        )
    )


@app.command()
def pending(
    session_id: str | None = typer.Option(
        None,
        "--session",
        "-s",
        help="Show only questions from this session",
        metavar="SESSION",
    ),
) -> None:
    """
    See what Claude needs your help with.

    Lists questions and decisions that Claude couldn't handle automatically.
    These are things that need your input before the session can continue.

    [bold yellow]Examples:[/]

      [dim]# See all pending questions[/]
      pcc pending

      [dim]# Filter to specific session[/]
      pcc pending -s a1b2c3d4

    [bold yellow]Next step:[/] Use 'pcc respond' to answer questions.
    """
    from perpetualcc.human import EscalationQueue, InteractivePrompt

    queue = EscalationQueue()

    async def _get_pending():
        await queue.initialize()
        try:
            # Find session if partial ID given
            resolved_session_id = None
            if session_id:
                manager = _get_session_manager()
                for s in manager.list_sessions():
                    if s.id.startswith(session_id):
                        resolved_session_id = s.id
                        break
            return await queue.get_pending(session_id=resolved_session_id)
        finally:
            await queue.close()

    escalations = asyncio.run(_get_pending())

    if not escalations:
        console.print("[dim]No pending escalations.[/dim]")
        return

    prompt = InteractivePrompt()
    prompt.show_pending_escalations(escalations)


@app.command()
def respond(
    request_id: str = typer.Argument(
        ...,
        help="Number from 'pcc pending' or the request ID",
        metavar="ID",
    ),
    response: str | None = typer.Argument(
        None,
        help="Your answer (or omit to type interactively)",
        metavar="ANSWER",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Force interactive mode even if answer provided",
    ),
) -> None:
    """
    Answer a question Claude asked.

    Use the number from 'pcc pending' to respond to questions.
    You can provide your answer directly or use interactive mode.

    [bold yellow]Examples:[/]

      [dim]# Answer the first pending question[/]
      pcc respond 1 "Yes, proceed with the refactoring"

      [dim]# Use interactive mode for complex answers[/]
      pcc respond 1 -i

      [dim]# Using the full request ID[/]
      pcc respond a1b2c3d4 "Use PostgreSQL"

    [bold yellow]Workflow:[/]
      1. pcc pending        # See what's waiting
      2. pcc respond 1 ...  # Answer by number
    """
    from perpetualcc.human import EscalationQueue, InteractivePrompt

    queue = EscalationQueue()

    async def _respond():
        await queue.initialize()
        try:
            # If request_id is a number, get from pending list
            resolved_id = request_id
            if request_id.isdigit():
                pending = await queue.get_pending()
                idx = int(request_id) - 1
                if idx < 0 or idx >= len(pending):
                    console.print(
                        f"[red]Error:[/red] Invalid index {request_id}. "
                        f"Use 'pcc pending' to see available escalations."
                    )
                    return
                resolved_id = pending[idx].id

            # Get the escalation
            escalation = await queue.get_request(resolved_id)
            if not escalation:
                # Try partial match
                pending = await queue.get_pending()
                for esc in pending:
                    if esc.id.startswith(resolved_id):
                        escalation = esc
                        break

            if not escalation:
                console.print(f"[red]Error:[/red] Escalation not found: {request_id}")
                return

            if not escalation.is_pending:
                console.print(
                    f"[yellow]Escalation already responded:[/yellow] {escalation.status.value}"
                )
                return

            # Get response
            final_response = response
            if not final_response or interactive:
                prompt = InteractivePrompt()
                final_response = prompt.prompt_escalation(escalation)

            if not final_response:
                console.print("[yellow]No response provided. Skipping.[/yellow]")
                return

            # Submit response
            result = await queue.respond(escalation.id, final_response)
            if result:
                console.print(f"[green]Responded:[/green] {final_response}")
            else:
                console.print("[red]Failed to submit response.[/red]")

        finally:
            await queue.close()

    asyncio.run(_respond())


# Config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage PerpetualCC configuration",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command(name="show")
def config_show() -> None:
    """
    Show current configuration.

    Displays all configuration settings from ~/.perpetualcc/config.toml
    or shows defaults if no config file exists.

    [bold yellow]Example:[/]
      pcc config show
    """
    from perpetualcc.config import DEFAULT_CONFIG_PATH, get_config

    config = get_config()

    # Check if config file exists
    config_exists = DEFAULT_CONFIG_PATH.exists()

    lines = [
        f"[bold]Config file:[/bold] {DEFAULT_CONFIG_PATH}",
        f"[bold]Status:[/bold] {'[green]exists[/green]' if config_exists else '[yellow]using defaults[/yellow]'}",
        "",
        "[bold cyan]Brain Settings[/bold cyan]",
        f"  Type: {config.brain.type.value}",
        f"  Confidence threshold: {config.brain.confidence_threshold}",
        "",
        "[bold cyan]Gemini[/bold cyan]",
        f"  API key: {'[green]set[/green]' if config.brain.gemini.get_api_key() else '[yellow]not set[/yellow]'}",
        f"  Model: {config.brain.gemini.model}",
        "",
        "[bold cyan]Ollama[/bold cyan]",
        f"  Host: {config.brain.ollama.host}",
        f"  Model: {config.brain.ollama.model}",
        f"  Timeout: {config.brain.ollama.timeout}s",
        "",
        "[bold cyan]Sessions[/bold cyan]",
        f"  Max concurrent: {config.sessions.max_concurrent}",
        f"  Auto resume: {config.sessions.auto_resume}",
        "",
        "[bold cyan]Permissions[/bold cyan]",
        f"  Auto approve low risk: {config.permissions.auto_approve_low_risk}",
        f"  Safe directories: {', '.join(config.permissions.safe_directories)}",
        "",
        "[bold cyan]Output[/bold cyan]",
        f"  Verbosity: {config.output.verbosity.value}",
        f"  Show timestamps: {config.output.show_timestamps}",
    ]

    console.print(
        Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold cyan]PerpetualCC Configuration[/bold cyan]",
            border_style="cyan",
        )
    )


@config_app.command(name="init")
def config_init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file",
    ),
) -> None:
    """
    Create a default configuration file.

    Creates ~/.perpetualcc/config.toml with default settings.
    You can then edit this file to customize PerpetualCC behavior.

    [bold yellow]Examples:[/]

      [dim]# Create default config[/]
      pcc config init

      [dim]# Overwrite existing config[/]
      pcc config init --force

    [bold yellow]Next steps:[/]
      1. Edit ~/.perpetualcc/config.toml
      2. Set brain type to "gemini" or "ollama" if desired
      3. Configure API keys or Ollama settings
    """
    from perpetualcc.config import (
        DEFAULT_CONFIG_PATH,
        ensure_config_dir,
        generate_default_config,
    )

    ensure_config_dir()

    if DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]Config file already exists:[/yellow] {DEFAULT_CONFIG_PATH}")
        console.print("Use --force to overwrite.")
        return

    config_content = generate_default_config()
    DEFAULT_CONFIG_PATH.write_text(config_content)

    console.print(f"[green]Created config file:[/green] {DEFAULT_CONFIG_PATH}")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Edit the config file to customize settings")
    console.print("  2. Set [cyan]brain.type[/cyan] to 'gemini' or 'ollama' for AI-powered answers")
    console.print("  3. Configure API keys or local model settings")
    console.print()
    console.print(f"[dim]Open in editor: code {DEFAULT_CONFIG_PATH}[/dim]")


@config_app.command(name="brain")
def config_brain(
    brain_type: str = typer.Argument(
        None,
        help="Brain type: rule_based, gemini, or ollama",
        metavar="TYPE",
    ),
) -> None:
    """
    Configure or check brain settings.

    Without arguments, shows current brain status.
    With a brain type, updates the configuration.

    [bold yellow]Brain types:[/]
      rule_based  - Pattern matching, no external AI (default)
      gemini      - Google Gemini API (requires API key)
      ollama      - Local LLM via Ollama (requires Ollama running)

    [bold yellow]Examples:[/]

      [dim]# Check brain status[/]
      pcc config brain

      [dim]# Switch to Gemini[/]
      pcc config brain gemini

      [dim]# Switch to Ollama[/]
      pcc config brain ollama

      [dim]# Switch back to rule-based[/]
      pcc config brain rule_based
    """
    from perpetualcc.brain.factory import get_brain_status
    from perpetualcc.config import (
        BrainType,
        DEFAULT_CONFIG_PATH,
        get_config,
        load_config,
        save_config,
    )

    config = get_config()

    if brain_type is None:
        # Show current brain status
        status = get_brain_status(config.brain)

        status_color = "green" if status["available"] else "yellow"
        lines = [
            f"[bold]Brain Type:[/bold] {status['type']}",
            f"[bold]Status:[/bold] [{status_color}]{status['message']}[/{status_color}]",
        ]

        if status["details"]:
            lines.append("")
            lines.append("[bold]Details:[/bold]")
            for key, value in status["details"].items():
                if key == "hint":
                    lines.append(f"  [yellow]Hint:[/yellow] {value}")
                elif isinstance(value, list):
                    lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"  {key}: {value}")

        console.print(
            Panel(
                Text.from_markup("\n".join(lines)),
                title="[bold cyan]Brain Status[/bold cyan]",
                border_style="cyan",
            )
        )
        return

    # Update brain type
    try:
        new_type = BrainType(brain_type)
    except ValueError:
        console.print(f"[red]Invalid brain type:[/red] {brain_type}")
        console.print("Valid types: rule_based, gemini, ollama")
        raise typer.Exit(1)

    # Load current config, update, and save
    current_config = load_config()
    current_config.brain.type = new_type
    save_config(current_config)

    console.print(f"[green]Updated brain type to:[/green] {new_type.value}")
    console.print(f"[dim]Saved to: {DEFAULT_CONFIG_PATH}[/dim]")

    # Show status of the new brain
    from perpetualcc.config import reload_config

    reload_config()
    status = get_brain_status(current_config.brain)

    if not status["available"]:
        console.print()
        console.print(f"[yellow]Note:[/yellow] {status['message']}")
        if "hint" in status.get("details", {}):
            console.print(f"[dim]{status['details']['hint']}[/dim]")


@config_app.command(name="path")
def config_path() -> None:
    """
    Show configuration file path.

    Useful for opening the config in an editor.

    [bold yellow]Example:[/]
      code $(pcc config path)
    """
    from perpetualcc.config import DEFAULT_CONFIG_PATH

    console.print(str(DEFAULT_CONFIG_PATH))


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Configuration key (e.g., brain.confidence_threshold)",
        metavar="KEY",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set",
        metavar="VALUE",
    ),
) -> None:
    """
    Set a configuration value.

    Supports dot-notation for nested values.

    [bold yellow]Examples:[/]

      [dim]# Set brain confidence threshold[/]
      pcc config set brain.confidence_threshold 0.8

      [dim]# Set Ollama model[/]
      pcc config set brain.ollama.model "deepseek-coder:33b"

      [dim]# Set output verbosity[/]
      pcc config set output.verbosity verbose

    [bold yellow]Available keys:[/]
      brain.type                    - rule_based, gemini, ollama
      brain.confidence_threshold    - 0.0 to 1.0
      brain.gemini.model           - Gemini model name
      brain.ollama.host            - Ollama server URL
      brain.ollama.model           - Ollama model name
      output.verbosity             - quiet, normal, verbose
      sessions.max_concurrent      - Max parallel sessions
      notifications.enabled        - true/false
    """
    from perpetualcc.config import (
        DEFAULT_CONFIG_PATH,
        BrainType,
        OutputVerbosity,
        load_config,
        save_config,
    )

    config = load_config()

    # Parse the key path
    parts = key.split(".")

    # Convert value to appropriate type
    def parse_value(val: str, target_type: str | None = None):
        # Boolean
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        # Number
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
        # String
        return val

    try:
        if len(parts) == 2:
            section, attr = parts
            obj = getattr(config, section)

            # Handle enums
            if section == "brain" and attr == "type":
                setattr(obj, attr, BrainType(value))
            elif section == "output" and attr == "verbosity":
                setattr(obj, attr, OutputVerbosity(value))
            else:
                setattr(obj, attr, parse_value(value))

        elif len(parts) == 3:
            section, subsection, attr = parts
            obj = getattr(getattr(config, section), subsection)
            setattr(obj, attr, parse_value(value))

        else:
            console.print(f"[red]Invalid key format:[/red] {key}")
            console.print("Use dot notation: section.key or section.subsection.key")
            raise typer.Exit(1)

        save_config(config)
        console.print(f"[green]Set {key} = {value}[/green]")
        console.print(f"[dim]Saved to: {DEFAULT_CONFIG_PATH}[/dim]")

    except AttributeError:
        console.print(f"[red]Unknown configuration key:[/red] {key}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Invalid value:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version number."""
    console.print(f"PerpetualCC v{__version__}")


if __name__ == "__main__":
    app()
