"""Hook implementations for Claude Agent SDK integration.

Hooks allow intercepting and modifying tool execution at various points:
- PreToolUse: Before tool execution (validation, logging, blocking)
- PostToolUse: After tool execution (logging, auditing)

These hooks integrate with ClaudeSDKClient's hook system.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookContext

logger = logging.getLogger(__name__)


async def pre_tool_use_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Pre-tool-use hook for logging and validation.

    This hook runs before each tool execution and can:
    - Log tool usage for auditing
    - Validate inputs
    - Block dangerous operations

    Args:
        input_data: Hook input data containing tool_name and tool_input
        tool_use_id: Optional tool use identifier
        context: Hook context

    Returns:
        Hook output dict (empty for pass-through)
    """
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})

    logger.debug("PreToolUse: %s (id=%s)", tool_name, tool_use_id)

    # Example: Block extremely dangerous commands
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Block rm -rf / or similar destructive patterns
        if "rm -rf /" in command or "rm -rf /*" in command:
            logger.warning("Blocked dangerous command: %s", command[:100])
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Dangerous command blocked by hook",
                }
            }

    return {}


async def post_tool_use_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Post-tool-use hook for logging and auditing.

    This hook runs after each tool execution and can:
    - Log tool results for auditing
    - Track state changes
    - Trigger follow-up actions

    Args:
        input_data: Hook input data containing tool_name, tool_input, tool_response
        tool_use_id: Optional tool use identifier
        context: Hook context

    Returns:
        Hook output dict (empty for pass-through)
    """
    tool_name = input_data.get("tool_name", "unknown")
    tool_response = input_data.get("tool_response", {})

    # Log completion with truncated response
    response_str = str(tool_response)
    if len(response_str) > 200:
        response_str = response_str[:200] + "..."

    logger.debug("PostToolUse: %s -> %s", tool_name, response_str)

    return {}


async def file_change_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Hook for tracking file changes (Write, Edit tools).

    This hook runs after Write or Edit operations to track
    which files have been modified during the session.

    Args:
        input_data: Hook input data
        tool_use_id: Optional tool use identifier
        context: Hook context

    Returns:
        Hook output dict
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    if tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        if file_path:
            logger.info("File modified: %s", file_path)

    return {}


def create_default_hooks() -> dict[str, list[HookMatcher]]:
    """Create the default hook configuration for PerpetualCC.

    Returns:
        Dictionary mapping hook events to HookMatcher lists
    """
    return {
        "PreToolUse": [
            HookMatcher(hooks=[pre_tool_use_hook]),
        ],
        "PostToolUse": [
            HookMatcher(hooks=[post_tool_use_hook]),
            HookMatcher(matcher="Write|Edit", hooks=[file_change_hook]),
        ],
    }


def create_logging_hooks() -> dict[str, list[HookMatcher]]:
    """Create hooks focused on logging (minimal intervention).

    Returns:
        Dictionary mapping hook events to HookMatcher lists
    """
    return {
        "PreToolUse": [
            HookMatcher(hooks=[pre_tool_use_hook]),
        ],
        "PostToolUse": [
            HookMatcher(hooks=[post_tool_use_hook]),
        ],
    }


def create_strict_hooks() -> dict[str, list[HookMatcher]]:
    """Create strict hooks with additional safety checks.

    Returns:
        Dictionary mapping hook events to HookMatcher lists
    """

    async def strict_bash_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: HookContext,
    ) -> dict[str, Any]:
        """Strict validation for Bash commands."""
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "")

        # Block various dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "mkfs.",
            "dd if=/dev/zero",
            "> /dev/sda",
            "chmod 777 /",
            ":(){ :|:& };:",  # Fork bomb
        ]

        for pattern in dangerous_patterns:
            if pattern in command:
                logger.warning("Strict hook blocked: %s", command[:100])
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Blocked dangerous pattern: {pattern}",
                    }
                }

        return {}

    return {
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[strict_bash_hook]),
            HookMatcher(hooks=[pre_tool_use_hook]),
        ],
        "PostToolUse": [
            HookMatcher(hooks=[post_tool_use_hook]),
            HookMatcher(matcher="Write|Edit", hooks=[file_change_hook]),
        ],
    }
