"""Claude Agent SDK wrapper for PerpetualCC."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
)

from perpetualcc.claude.types import (
    AskQuestionEvent,
    ClaudeEvent,
    InitEvent,
    Question,
    QuestionOption,
    RateLimitEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)

logger = logging.getLogger(__name__)

# Type aliases for permission callbacks
PermissionResult = PermissionResultAllow | PermissionResultDeny
PermissionCallback = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult],
]

# Type alias for hooks
HookCallback = Callable[
    [dict[str, Any], str | None, Any],  # HookContext
    Awaitable[dict[str, Any]],
]


class ClaudeCodeAdapter:
    """Wraps Claude Agent SDK for PerpetualCC session management.

    This adapter provides:
    - Session lifecycle management (connect/disconnect)
    - Continuous conversation support via ClaudeSDKClient
    - Interrupt capability for stopping execution
    - Custom permission handling via can_use_tool callback
    - Event stream processing converting SDK messages to ClaudeEvents

    Implementation Note:
        Uses ClaudeSDKClient with async context manager pattern as recommended
        by the official documentation to properly handle asyncio cleanup and
        avoid "cancel scope in different task" errors.

        From the docs: "When iterating over messages, avoid using break to exit
        early as this can cause asyncio cleanup issues."
    """

    def __init__(
        self,
        project_path: str | Path,
        allowed_tools: list[str] | None = None,
        permission_mode: str | None = None,
        permission_callback: PermissionCallback | None = None,
        hooks: dict[str, list[HookMatcher]] | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        resume_session_id: str | None = None,
    ):
        """
        Initialize the Claude Code adapter.

        Args:
            project_path: Path to the project directory
            allowed_tools: List of allowed tool names
            permission_mode: Permission mode for SDK ("bypassPermissions", "default", etc.)
            permission_callback: Async callback for permission decisions
                                 Signature: (tool_name, input_data, context) -> PermissionResult
            hooks: Hook configurations for intercepting events
            model: Model to use (sonnet, opus, haiku)
            max_turns: Maximum number of turns before stopping
            resume_session_id: Optional Claude session ID to resume from
        """
        self.project_path = Path(project_path).resolve()
        self.allowed_tools = allowed_tools or [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "Task",
        ]
        self.permission_mode = permission_mode
        self.permission_callback = permission_callback
        self.hooks = hooks
        self.model = model
        self.max_turns = max_turns

        # Client state
        self._client: ClaudeSDKClient | None = None
        self._connected: bool = False
        self._current_prompt: str | None = None
        self.session_id: str | None = resume_session_id

    def _build_options(self, resume: str | None = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from adapter config."""
        kwargs: dict[str, Any] = {
            "allowed_tools": self.allowed_tools,
            "cwd": str(self.project_path),
        }

        if self.permission_mode:
            kwargs["permission_mode"] = self.permission_mode

        if self.permission_callback:
            kwargs["can_use_tool"] = self.permission_callback

        if self.hooks:
            kwargs["hooks"] = self.hooks

        if self.model:
            kwargs["model"] = self.model

        if self.max_turns:
            kwargs["max_turns"] = self.max_turns

        if resume:
            kwargs["resume"] = resume

        return ClaudeAgentOptions(**kwargs)

    @property
    def connected(self) -> bool:
        """Check if the adapter is connected to a session."""
        return self._connected

    async def connect(self, initial_prompt: str | None = None) -> None:
        """Connect to Claude with an optional initial prompt.

        Uses ClaudeSDKClient for proper async context management.

        Args:
            initial_prompt: Optional prompt to send on connect

        Raises:
            RuntimeError: If already connected
        """
        if self._connected:
            await self.disconnect()

        self._current_prompt = initial_prompt
        self._connected = True

        # Create client with options
        options = self._build_options(resume=self.session_id)
        self._client = ClaudeSDKClient(options=options)

        # Connect the client
        await self._client.connect()

        logger.debug("Adapter connected with prompt: %s", initial_prompt[:50] if initial_prompt else None)

    async def disconnect(self) -> None:
        """Disconnect from Claude."""
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                # Log but don't raise - cleanup errors shouldn't break the flow
                logger.debug("Client disconnect error (ignored): %s", e)
            self._client = None

        self._connected = False
        self._current_prompt = None
        logger.debug("Adapter disconnected")

    async def query(self, prompt: str) -> None:
        """Send a query to the connected session.

        Uses ClaudeSDKClient.query() for proper async handling.

        Args:
            prompt: The prompt to send

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected. Call connect() first.")
        self._current_prompt = prompt
        await self._client.query(prompt)

    async def receive_events(self) -> AsyncIterator[ClaudeEvent]:
        """Receive and convert messages to ClaudeEvents.

        Uses ClaudeSDKClient.receive_response() as recommended by the
        official documentation to avoid asyncio cleanup issues.

        Yields:
            ClaudeEvent: Converted events from the session

        Raises:
            RuntimeError: If not connected or no prompt set
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        if not self._current_prompt:
            raise RuntimeError("No prompt set. Call connect(prompt) or query(prompt) first.")

        logger.debug("Starting query with prompt: %s", self._current_prompt[:50])

        # Send the query if not already sent
        # (connect() doesn't send the initial prompt automatically)
        await self._client.query(self._current_prompt)

        # Use receive_response() which properly handles the async iteration
        # and waits for ResultMessage. Per docs: "avoid using break to exit
        # early as this can cause asyncio cleanup issues"
        async for message in self._client.receive_response():
            for event in self._convert_message(message):
                yield event

        # Clear prompt after processing
        self._current_prompt = None

    async def interrupt(self) -> None:
        """Interrupt the current execution.

        Uses ClaudeSDKClient.interrupt() for proper signal handling.
        """
        if self._client:
            logger.info("Sending interrupt signal to Claude")
            await self._client.interrupt()
        else:
            logger.warning("Interrupt requested but no client connected")
            await self.disconnect()

    async def send_response(self, response: str) -> AsyncIterator[ClaudeEvent]:
        """Send a response to a question and receive events.

        This is used to answer AskQuestionEvent questions within
        a continuous conversation.

        Note: With query() approach, this starts a new query with the response
        and the resume session ID. True session continuity requires ClaudeSDKClient.

        Args:
            response: The response to send

        Yields:
            ClaudeEvent: Events from the continued session
        """
        if not self._connected:
            raise RuntimeError("Not connected.")

        # Set the response as the next prompt
        await self.query(response)

        # Receive events from the continued session
        async for event in self.receive_events():
            yield event

    def _convert_message(self, message: Any) -> list[ClaudeEvent]:
        """Convert a single SDK message to one or more ClaudeEvents."""
        events: list[ClaudeEvent] = []

        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                session_id = getattr(message, "session_id", "") or message.data.get(
                    "session_id", ""
                )
                self.session_id = session_id
                events.append(InitEvent(session_id=session_id, raw=message.data))

        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    events.append(TextEvent(text=block.text))
                elif isinstance(block, ThinkingBlock):
                    events.append(ThinkingEvent(thinking=block.thinking))
                elif isinstance(block, ToolUseBlock):
                    if block.name == "AskUserQuestion":
                        events.append(self._parse_ask_question(block))
                    else:
                        events.append(
                            ToolUseEvent(
                                tool_use_id=block.id,
                                tool_name=block.name,
                                tool_input=block.input,
                            )
                        )
                elif isinstance(block, ToolResultBlock):
                    content = (
                        block.content if isinstance(block.content, str) else str(block.content)
                    )
                    events.append(
                        ToolResultEvent(
                            tool_use_id=block.tool_use_id,
                            content=content,
                            is_error=block.is_error or False,
                        )
                    )

        elif isinstance(message, ResultMessage):
            if self._is_rate_limit(message):
                events.append(
                    RateLimitEvent(
                        retry_after=self._parse_retry_after(message),
                        message=message.result or "Rate limited",
                    )
                )
            else:
                self.session_id = getattr(message, "session_id", self.session_id)
                events.append(
                    ResultEvent(
                        is_error=message.is_error,
                        result=message.result,
                        session_id=getattr(message, "session_id", "") or "",
                        duration_ms=getattr(message, "duration_ms", 0) or 0,
                        total_cost_usd=getattr(message, "total_cost_usd", None),
                        num_turns=getattr(message, "num_turns", 0) or 0,
                    )
                )

        return events

    def _parse_ask_question(self, block: ToolUseBlock) -> AskQuestionEvent:
        """Parse an AskUserQuestion tool use into an AskQuestionEvent."""
        questions: list[Question] = []
        for q_data in block.input.get("questions", []):
            options = [
                QuestionOption(label=o.get("label", ""), description=o.get("description", ""))
                for o in q_data.get("options", [])
            ]
            questions.append(
                Question(
                    question=q_data.get("question", ""),
                    header=q_data.get("header", ""),
                    options=options,
                    multi_select=q_data.get("multiSelect", False),
                )
            )
        return AskQuestionEvent(
            questions=questions,
            tool_use_id=block.id,
            tool_input=block.input,
        )

    def _is_rate_limit(self, message: ResultMessage) -> bool:
        """Check if a result message indicates a rate limit."""
        if message.is_error and message.result:
            from perpetualcc.core.rate_limit_utils import is_rate_limit_message

            return is_rate_limit_message(message.result)
        return False

    def _parse_retry_after(self, message: ResultMessage) -> int:
        """Parse retry-after seconds from a rate limit message."""
        if message.result:
            from perpetualcc.core.rate_limit_utils import parse_rate_limit_message

            parsed = parse_rate_limit_message(message.result)
            if parsed.is_rate_limit:
                return parsed.retry_after_seconds
        # Default to 60 seconds if we can't parse
        return 60
