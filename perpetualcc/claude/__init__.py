"""Claude Agent SDK integration for PerpetualCC."""

from perpetualcc.claude.adapter import ClaudeCodeAdapter, PermissionCallback, PermissionResult
from perpetualcc.claude.hooks import (
    create_default_hooks,
    create_logging_hooks,
    create_strict_hooks,
)
from perpetualcc.claude.types import (
    AskQuestionEvent,
    ClaudeEvent,
    InitEvent,
    Question,
    QuestionOption,
    RateLimitEvent,
    ResultEvent,
    SessionState,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)

__all__ = [
    # Adapter
    "ClaudeCodeAdapter",
    "PermissionCallback",
    "PermissionResult",
    # Hooks
    "create_default_hooks",
    "create_logging_hooks",
    "create_strict_hooks",
    # Event types
    "ClaudeEvent",
    "InitEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolUseEvent",
    "ToolResultEvent",
    "AskQuestionEvent",
    "Question",
    "QuestionOption",
    "ResultEvent",
    "RateLimitEvent",
    "SessionState",
]
