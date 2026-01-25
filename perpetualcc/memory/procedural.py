"""Procedural memory - learned rules with confidence levels.

Procedural memory stores learned patterns for handling specific situations.
Each procedure has a trigger pattern and an action to take when the pattern matches.
Confidence levels adjust based on success/failure feedback.

This enables the system to:
- Learn patterns from successful decisions
- Auto-approve recurring situations with high confidence
- Gracefully degrade confidence when patterns fail
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from typing import Any

from perpetualcc.memory.store import MemoryStore, StoredProcedure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcedureMatch:
    """A matched procedure with match details.

    Attributes:
        procedure: The matched procedure
        match_type: How it was matched (exact/regex/glob)
        match_score: Match quality score (1.0 = perfect match)
    """

    procedure: StoredProcedure
    match_type: str
    match_score: float = 1.0


@dataclass(frozen=True)
class ProceduralMemoryConfig:
    """Configuration for procedural memory.

    Attributes:
        confidence_increase: Amount to increase confidence on success
        confidence_decrease: Amount to decrease confidence on failure
        min_confidence: Minimum confidence (floor)
        max_confidence: Maximum confidence (ceiling)
        auto_approve_threshold: Confidence level for automatic approval
        initial_confidence: Default confidence for new procedures
    """

    confidence_increase: float = 0.05
    confidence_decrease: float = 0.1
    min_confidence: float = 0.1
    max_confidence: float = 0.99
    auto_approve_threshold: float = 0.7
    initial_confidence: float = 0.5


# Common trigger types
class TriggerType:
    """Standard trigger types for procedures."""

    TOOL_USE = "tool_use"  # Tool name + input pattern
    QUESTION = "question"  # Question pattern
    BASH_COMMAND = "bash_command"  # Specific bash command pattern
    FILE_PATH = "file_path"  # File path pattern
    GIT_COMMAND = "git_command"  # Git command pattern
    ERROR = "error"  # Error message pattern


# Common actions
class ActionType:
    """Standard action types for procedures."""

    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"
    ANSWER_YES = "answer_yes"
    ANSWER_NO = "answer_no"
    ANSWER_FIRST = "answer_first"  # Select first option


class ProceduralMemory:
    """Procedural memory system for learned rules.

    This class manages learned procedures that map trigger patterns to actions.
    Procedures have confidence levels that adjust based on feedback.

    Usage:
        memory = ProceduralMemory(store)

        # Match a procedure
        match = await memory.match("tool_use", "Read:/src/main.py")
        if match and match.procedure.confidence >= 0.7:
            # Use the learned action
            action = match.procedure.action

        # Record outcome to update confidence
        await memory.record_outcome(match.procedure.id, success=True)

        # Add a new procedure
        await memory.add_procedure(
            trigger_type="question",
            trigger_pattern="Should I proceed?",
            action="answer_yes",
        )
    """

    def __init__(
        self,
        store: MemoryStore,
        config: ProceduralMemoryConfig | None = None,
    ):
        """Initialize procedural memory.

        Args:
            store: The underlying memory store
            config: Optional configuration
        """
        self.store = store
        self.config = config or ProceduralMemoryConfig()

    async def match(
        self,
        trigger_type: str,
        input_value: str,
    ) -> ProcedureMatch | None:
        """Find a matching procedure for the given trigger.

        Matching is tried in order:
        1. Exact match
        2. Glob pattern match (e.g., "*.py")
        3. Regex pattern match (patterns starting with "^" or containing ".*")

        Args:
            trigger_type: Type of trigger (tool_use, question, etc.)
            input_value: Value to match against trigger patterns

        Returns:
            ProcedureMatch if found, None otherwise
        """
        # Get all procedures of this type
        procedures = await self.store.query_procedures(trigger_type=trigger_type, limit=1000)

        if not procedures:
            return None

        # Try exact match first
        for proc in procedures:
            if proc.trigger_pattern == input_value:
                return ProcedureMatch(procedure=proc, match_type="exact", match_score=1.0)

        # Try glob pattern match
        for proc in procedures:
            if self._is_glob_pattern(proc.trigger_pattern):
                if fnmatch(input_value, proc.trigger_pattern):
                    return ProcedureMatch(procedure=proc, match_type="glob", match_score=0.9)

        # Try regex pattern match
        for proc in procedures:
            if self._is_regex_pattern(proc.trigger_pattern):
                try:
                    if re.match(proc.trigger_pattern, input_value):
                        return ProcedureMatch(procedure=proc, match_type="regex", match_score=0.8)
                except re.error:
                    logger.warning(
                        "Invalid regex pattern in procedure %d: %s",
                        proc.id,
                        proc.trigger_pattern,
                    )

        return None

    async def match_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> ProcedureMatch | None:
        """Match a tool use request against procedures.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters

        Returns:
            ProcedureMatch if found, None otherwise
        """
        # Create a matchable string from tool use
        # Format: "ToolName:key=value,key=value"
        if tool_name == "Bash":
            # Special handling for bash commands
            command = tool_input.get("command", "")
            input_str = f"Bash:{command}"
            match = await self.match(TriggerType.BASH_COMMAND, input_str)
            if match:
                return match
            # Also try just the command
            return await self.match(TriggerType.BASH_COMMAND, command)
        elif tool_name in ("Write", "Edit", "Read"):
            # Special handling for file operations
            file_path = tool_input.get("file_path", "")
            input_str = f"{tool_name}:{file_path}"
            match = await self.match(TriggerType.FILE_PATH, input_str)
            if match:
                return match
            # Also try just the file path
            return await self.match(TriggerType.FILE_PATH, file_path)
        else:
            # Generic tool matching
            input_str = f"{tool_name}:{','.join(f'{k}={v}' for k, v in sorted(tool_input.items()))}"
            return await self.match(TriggerType.TOOL_USE, input_str)

    async def match_question(self, question_text: str) -> ProcedureMatch | None:
        """Match a question against procedures.

        Args:
            question_text: The question text

        Returns:
            ProcedureMatch if found, None otherwise
        """
        return await self.match(TriggerType.QUESTION, question_text)

    async def match_bash_command(self, command: str) -> ProcedureMatch | None:
        """Match a bash command against procedures.

        Args:
            command: The bash command

        Returns:
            ProcedureMatch if found, None otherwise
        """
        return await self.match(TriggerType.BASH_COMMAND, command)

    async def record_outcome(self, procedure_id: int, success: bool) -> StoredProcedure | None:
        """Record the outcome of applying a procedure.

        This updates the procedure's confidence based on the outcome.

        Args:
            procedure_id: The procedure ID
            success: Whether the application was successful

        Returns:
            The updated procedure, or None if not found
        """
        return await self.store.update_procedure_outcome(procedure_id, success)

    async def add_procedure(
        self,
        trigger_type: str,
        trigger_pattern: str,
        action: str,
        confidence: float | None = None,
    ) -> int:
        """Add a new procedure.

        If a procedure with the same trigger already exists, it is updated.

        Args:
            trigger_type: Type of trigger
            trigger_pattern: Pattern to match
            action: Action to take when matched
            confidence: Initial confidence (defaults to config.initial_confidence)

        Returns:
            The procedure ID
        """
        confidence = confidence or self.config.initial_confidence
        return await self.store.upsert_procedure(
            trigger_type=trigger_type,
            trigger_pattern=trigger_pattern,
            action=action,
            confidence=confidence,
        )

    async def get_high_confidence_procedures(
        self, min_confidence: float | None = None
    ) -> list[StoredProcedure]:
        """Get procedures with high confidence.

        Args:
            min_confidence: Minimum confidence threshold
                           (defaults to config.auto_approve_threshold)

        Returns:
            List of high-confidence procedures
        """
        threshold = min_confidence or self.config.auto_approve_threshold
        return await self.store.query_procedures(min_confidence=threshold)

    async def get_procedures_by_type(
        self, trigger_type: str, limit: int = 100
    ) -> list[StoredProcedure]:
        """Get procedures by trigger type.

        Args:
            trigger_type: Type of trigger to filter by
            limit: Maximum number of results

        Returns:
            List of matching procedures
        """
        return await self.store.query_procedures(trigger_type=trigger_type, limit=limit)

    async def learn_from_episode(
        self,
        event_type: str,
        context: str,
        action: str,
        outcome: str,
    ) -> int | None:
        """Learn a procedure from an episode.

        This extracts a trigger pattern from the context and creates
        or updates a procedure.

        Args:
            event_type: Type of event
            context: Episode context (used to extract trigger pattern)
            action: The action that was taken
            outcome: The outcome (success/failure)

        Returns:
            Procedure ID if a procedure was created/updated, None otherwise
        """
        # Map event types to trigger types
        trigger_type = self._event_to_trigger_type(event_type)
        if not trigger_type:
            return None

        # Extract a pattern from the context
        trigger_pattern = self._extract_pattern(context)
        if not trigger_pattern:
            return None

        # Map action to procedure action
        procedure_action = self._action_to_procedure_action(action)
        if not procedure_action:
            return None

        # Check if procedure exists
        existing = await self.store.get_procedure(trigger_type, trigger_pattern)

        if existing:
            # Update confidence based on outcome
            await self.record_outcome(existing.id, success=(outcome == "success"))
            return existing.id
        else:
            # Create new procedure with initial confidence
            # Successful outcomes get higher initial confidence
            initial_conf = (
                self.config.initial_confidence + 0.1
                if outcome == "success"
                else self.config.initial_confidence - 0.05
            )
            initial_conf = max(
                self.config.min_confidence,
                min(self.config.max_confidence, initial_conf),
            )

            return await self.add_procedure(
                trigger_type=trigger_type,
                trigger_pattern=trigger_pattern,
                action=procedure_action,
                confidence=initial_conf,
            )

    async def decay_unused_procedures(
        self, days_unused: int = 30, decay_amount: float = 0.05
    ) -> int:
        """Decay confidence for procedures not used recently.

        This prevents old, potentially outdated procedures from having
        inappropriately high confidence.

        Args:
            days_unused: Number of days without use to trigger decay
            decay_amount: Amount to decrease confidence

        Returns:
            Number of procedures decayed
        """
        procedures = await self.store.query_procedures(limit=10000)
        decayed = 0
        cutoff = datetime.now().timestamp() - (days_unused * 24 * 60 * 60)

        for proc in procedures:
            if proc.last_used_at:
                if proc.last_used_at.timestamp() < cutoff:
                    # Decay by recording a "failure" outcome
                    await self.store.update_procedure_outcome(proc.id, success=False)
                    decayed += 1
            elif proc.created_at:
                if proc.created_at.timestamp() < cutoff:
                    await self.store.update_procedure_outcome(proc.id, success=False)
                    decayed += 1

        if decayed > 0:
            logger.info("Decayed %d unused procedures", decayed)

        return decayed

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about procedural memory.

        Returns:
            Dictionary with procedure statistics
        """
        procedures = await self.store.query_procedures(limit=10000)

        if not procedures:
            return {
                "total": 0,
                "by_trigger_type": {},
                "by_action": {},
                "avg_confidence": 0.0,
                "high_confidence_count": 0,
            }

        by_type: dict[str, int] = {}
        by_action: dict[str, int] = {}
        total_confidence = 0.0
        high_conf_count = 0

        for proc in procedures:
            by_type[proc.trigger_type] = by_type.get(proc.trigger_type, 0) + 1
            by_action[proc.action] = by_action.get(proc.action, 0) + 1
            total_confidence += proc.confidence
            if proc.confidence >= self.config.auto_approve_threshold:
                high_conf_count += 1

        return {
            "total": len(procedures),
            "by_trigger_type": by_type,
            "by_action": by_action,
            "avg_confidence": total_confidence / len(procedures),
            "high_confidence_count": high_conf_count,
        }

    @staticmethod
    def _is_glob_pattern(pattern: str) -> bool:
        """Check if a pattern contains glob characters."""
        return "*" in pattern or "?" in pattern or "[" in pattern

    @staticmethod
    def _is_regex_pattern(pattern: str) -> bool:
        """Check if a pattern appears to be a regex."""
        return (
            pattern.startswith("^")
            or pattern.endswith("$")
            or ".*" in pattern
            or "\\d" in pattern
            or "\\w" in pattern
        )

    @staticmethod
    def _event_to_trigger_type(event_type: str) -> str | None:
        """Map event types to trigger types."""
        mapping = {
            "permission_request": TriggerType.TOOL_USE,
            "question": TriggerType.QUESTION,
            "error": TriggerType.ERROR,
        }
        return mapping.get(event_type)

    @staticmethod
    def _extract_pattern(context: str) -> str | None:
        """Extract a matchable pattern from context."""
        # Try to extract meaningful patterns
        # "Tool use request: Write" -> "Write"
        # "Question: Should I proceed?" -> "Should I proceed?"

        if context.startswith("Tool use request: "):
            return context[18:].strip()
        if context.startswith("Question: "):
            return context[10:].strip()
        if context.startswith("Error: "):
            return context[7:].strip()

        # Return the context itself if no pattern extraction possible
        return context.strip() if context.strip() else None

    @staticmethod
    def _action_to_procedure_action(action: str) -> str | None:
        """Map action types to procedure actions."""
        mapping = {
            "approve_tool": ActionType.APPROVE,
            "deny_tool": ActionType.DENY,
            "escalate": ActionType.ESCALATE,
            "answer": ActionType.ANSWER_YES,  # Assumes "yes" type answers
        }
        return mapping.get(action)
