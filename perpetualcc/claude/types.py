"""Event types and session state for Claude Code integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SessionState(Enum):
    """State of a Claude Code session."""

    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_INPUT = "waiting_input"
    RATE_LIMITED = "rate_limited"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ClaudeEvent:
    """Base event from a Claude Code session."""

    type: str
    timestamp: datetime = field(default_factory=datetime.now)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitEvent(ClaudeEvent):
    """Session initialization event with session ID."""

    type: str = "init"
    session_id: str = ""


@dataclass
class TextEvent(ClaudeEvent):
    """Text output from Claude."""

    type: str = "text"
    text: str = ""


@dataclass
class ThinkingEvent(ClaudeEvent):
    """Thinking/reasoning output from Claude."""

    type: str = "thinking"
    thinking: str = ""


@dataclass
class ToolUseEvent(ClaudeEvent):
    """Claude wants to use a tool."""

    type: str = "tool_use"
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent(ClaudeEvent):
    """Result from a tool execution."""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class QuestionOption:
    """An option in a question from Claude."""

    label: str
    description: str = ""


@dataclass
class Question:
    """A question from Claude to the user."""

    question: str
    header: str = ""
    options: list[QuestionOption] = field(default_factory=list)
    multi_select: bool = False


@dataclass
class AskQuestionEvent(ClaudeEvent):
    """Claude is asking the user a question."""

    type: str = "ask_question"
    questions: list[Question] = field(default_factory=list)
    tool_use_id: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultEvent(ClaudeEvent):
    """Session result (completion or error)."""

    type: str = "result"
    is_error: bool = False
    result: str | None = None
    session_id: str = ""
    duration_ms: int = 0
    total_cost_usd: float | None = None
    num_turns: int = 0


@dataclass
class RateLimitEvent(ClaudeEvent):
    """Rate limit detected."""

    type: str = "rate_limit"
    retry_after: int = 0
    reset_time: datetime | None = None
    message: str = ""
