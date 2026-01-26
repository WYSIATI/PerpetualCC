"""Rule-based brain implementation using pattern matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from perpetualcc.brain.base import (
    Brain,
    BrainAnswer,
    PermissionContext,
    PermissionDecision,
    QuestionContext,
)


@dataclass(frozen=True)
class QuestionPattern:
    """Pattern for matching questions."""

    pattern: re.Pattern[str]
    answer: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class PermissionPattern:
    """Pattern for matching permission requests."""

    tool_name: str | None  # None matches any tool
    input_pattern: re.Pattern[str] | None  # Pattern to match against input string
    approve: bool
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class RuleBasedConfig:
    """Configuration for rule-based brain."""

    # Question patterns - order matters, first match wins
    question_patterns: tuple[QuestionPattern, ...] = field(default_factory=tuple)

    # Permission patterns - order matters, first match wins
    permission_patterns: tuple[PermissionPattern, ...] = field(default_factory=tuple)

    # Default confidence threshold for auto-answer
    confidence_threshold: float = 0.7


def default_question_patterns() -> tuple[QuestionPattern, ...]:
    """Create default question patterns."""
    return (
        # Confirmation questions - high confidence approval
        QuestionPattern(
            pattern=re.compile(r"proceed\??$", re.IGNORECASE),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation to proceed",
        ),
        QuestionPattern(
            pattern=re.compile(r"continue\??$", re.IGNORECASE),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation to continue",
        ),
        QuestionPattern(
            pattern=re.compile(r"(should|shall|can)\s+i\s+proceed", re.IGNORECASE),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation to proceed",
        ),
        QuestionPattern(
            pattern=re.compile(r"(should|shall|can)\s+i\s+continue", re.IGNORECASE),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation to continue",
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(is|does)\s+this\s+(look|seem)\s+(ok|good|correct)", re.IGNORECASE
            ),
            answer="Yes",
            confidence=0.75,
            reasoning="Affirmative response to verification question",
        ),
        QuestionPattern(
            pattern=re.compile(r"ready\s+to\s+(start|begin|proceed)", re.IGNORECASE),
            answer="Yes",
            confidence=0.85,
            reasoning="Standard confirmation to start",
        ),
        # Standard operations - medium confidence
        QuestionPattern(
            pattern=re.compile(r"(run|execute)\s+(the\s+)?(tests?|build|lint)", re.IGNORECASE),
            answer="Yes",
            confidence=0.80,
            reasoning="Standard development operation",
        ),
        QuestionPattern(
            pattern=re.compile(r"(install|add)\s+(the\s+)?dependencies", re.IGNORECASE),
            answer="Yes",
            confidence=0.75,
            reasoning="Standard dependency installation",
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(create|add)\s+(a\s+)?(new\s+)?(file|directory|folder)", re.IGNORECASE
            ),
            answer="Yes",
            confidence=0.75,
            reasoning="Standard file creation operation",
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(update|modify|change|edit)\s+(the\s+)?(code|file)", re.IGNORECASE
            ),
            answer="Yes",
            confidence=0.75,
            reasoning="Standard code modification",
        ),
        # Git operations - slightly lower confidence due to potential impact
        QuestionPattern(
            pattern=re.compile(r"commit\s+(the\s+)?changes", re.IGNORECASE),
            answer="Yes",
            confidence=0.70,
            reasoning="Git commit operation",
        ),
        QuestionPattern(
            pattern=re.compile(r"push\s+(to\s+)?(remote|origin|main|master)", re.IGNORECASE),
            answer="Yes",
            confidence=0.65,
            reasoning="Git push operation - requires more caution",
        ),
        # Type/format selection - approve first option with low confidence
        QuestionPattern(
            pattern=re.compile(r"(which|what)\s+(type|format|option|approach)", re.IGNORECASE),
            answer="",  # Empty means select first option
            confidence=0.60,
            reasoning="Default to first option for type selection",
        ),
    )


def default_permission_patterns() -> tuple[PermissionPattern, ...]:
    """Create default permission patterns for medium-risk operations.

    IMPORTANT: These patterns are conservative and avoid "always allow" rules.
    The rule-based brain returns low confidence for most operations, deferring
    to LLM brains or human escalation. Only clearly dangerous operations are
    confidently denied.

    Pattern philosophy:
    - DENY patterns: High confidence denial for clearly dangerous operations
    - APPROVE patterns: Low confidence (below threshold) to suggest approval
      but still require LLM evaluation or human confirmation
    - No "always allow" - even safe-looking operations have modest confidence
    """
    return (
        # HIGH CONFIDENCE DENIALS - clearly dangerous operations
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"git\s+push\s+(--force|-f)", re.IGNORECASE),
            approve=False,
            confidence=0.95,
            reasoning="Force push can destroy remote history - requires human approval",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"docker\s+(rm|rmi|prune|system\s+prune)", re.IGNORECASE),
            approve=False,
            confidence=0.85,
            reasoning="Docker cleanup can remove important containers/images",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"kubectl\s+delete", re.IGNORECASE),
            approve=False,
            confidence=0.90,
            reasoning="Kubernetes delete operations are dangerous",
        ),

        # LOW CONFIDENCE SUGGESTIONS - let LLM or human decide
        # These return low confidence intentionally - they suggest approval
        # but don't auto-approve. LLM brains will evaluate these with full context.
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^git\s+(add|commit)\b", re.IGNORECASE),
            approve=True,
            confidence=0.55,  # Below typical threshold - needs LLM/human confirmation
            reasoning="Git commit - suggest approval but verify with context",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^git\s+push\b(?!\s+(--force|-f))", re.IGNORECASE),
            approve=True,
            confidence=0.50,  # Below threshold - pushing needs verification
            reasoning="Git push - suggest approval but verify target branch",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^curl\s+", re.IGNORECASE),
            approve=True,
            confidence=0.45,  # Low - network operations need review
            reasoning="Network operation - needs context verification",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^wget\s+", re.IGNORECASE),
            approve=True,
            confidence=0.45,
            reasoning="Network download - needs context verification",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^docker\s+(build|run)\b", re.IGNORECASE),
            approve=True,
            confidence=0.55,
            reasoning="Docker build/run - suggest approval with context check",
        ),
        PermissionPattern(
            tool_name="Write",
            input_pattern=re.compile(r"(package\.json|pyproject\.toml|Cargo\.toml)$", re.IGNORECASE),
            approve=True,
            confidence=0.55,
            reasoning="Config file modification - suggest approval with diff review",
        ),
        PermissionPattern(
            tool_name="Edit",
            input_pattern=re.compile(r"(package\.json|pyproject\.toml|Cargo\.toml)$", re.IGNORECASE),
            approve=True,
            confidence=0.55,
            reasoning="Config file edit - suggest approval with diff review",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^rm\s+", re.IGNORECASE),
            approve=True,
            confidence=0.40,  # Very low - file deletion needs careful review
            reasoning="File removal - needs careful path verification",
        ),
        PermissionPattern(
            tool_name="Bash",
            input_pattern=re.compile(r"^mv\s+", re.IGNORECASE),
            approve=True,
            confidence=0.50,
            reasoning="File move - suggest approval with path verification",
        ),
        PermissionPattern(
            tool_name="Task",
            input_pattern=None,
            approve=True,
            confidence=0.55,
            reasoning="Task delegation - suggest approval but verify task description",
        ),
    )


class RuleBasedBrain(Brain):
    """
    Rule-based brain implementation using pattern matching.

    This brain implementation uses predefined patterns to answer questions
    and evaluate permission requests. It's designed to handle common cases
    without requiring an external AI service.

    Pattern matching strategy:
    - Questions: Match against predefined patterns, select first match
    - Permissions: Match tool name and input against patterns
    - Unknown cases: Return low confidence to trigger human escalation
    """

    def __init__(self, config: RuleBasedConfig | None = None):
        """
        Initialize the rule-based brain.

        Args:
            config: Optional custom configuration. If None, uses defaults.
        """
        if config is None:
            config = RuleBasedConfig(
                question_patterns=default_question_patterns(),
                permission_patterns=default_permission_patterns(),
            )
        self._config = config

    @property
    def config(self) -> RuleBasedConfig:
        """Get the current configuration."""
        return self._config

    async def answer_question(
        self,
        question: str,
        options: list[dict[str, str]],
        context: QuestionContext,
    ) -> BrainAnswer:
        """
        Answer a question from Claude Code using pattern matching.

        Args:
            question: The question text
            options: Available options [{label, description}]
            context: Additional context for answering

        Returns:
            BrainAnswer with selected option, confidence, and reasoning
        """
        # Normalize question for matching
        question_normalized = question.strip()

        # Try to match against patterns
        for pattern in self._config.question_patterns:
            if pattern.pattern.search(question_normalized):
                # If answer is empty, select first option
                if pattern.answer == "" and options:
                    selected = options[0].get("label", options[0].get("description", ""))
                    return BrainAnswer(
                        selected=selected,
                        confidence=pattern.confidence,
                        reasoning=f"{pattern.reasoning} - selected first option",
                    )
                return BrainAnswer(
                    selected=pattern.answer,
                    confidence=pattern.confidence,
                    reasoning=pattern.reasoning,
                )

        # No pattern matched - return low confidence to escalate
        return BrainAnswer(
            selected=None,
            confidence=0.0,
            reasoning="No matching pattern found, requires human input",
        )

    async def evaluate_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """
        Evaluate a medium-risk permission request using pattern matching.

        Args:
            tool_name: Name of the tool (Write, Edit, Bash, etc.)
            tool_input: Input parameters for the tool
            context: Session and project context

        Returns:
            PermissionDecision indicating whether to approve
        """
        # Get the input string to match against
        input_string = self._get_input_string(tool_name, tool_input)

        # Try to match against patterns
        for pattern in self._config.permission_patterns:
            # Check tool name match
            if pattern.tool_name is not None and pattern.tool_name != tool_name:
                continue

            # Check input pattern match
            if pattern.input_pattern is not None:
                if not pattern.input_pattern.search(input_string):
                    continue

            # Pattern matched
            return PermissionDecision(
                approve=pattern.approve,
                confidence=pattern.confidence,
                reason=pattern.reasoning,
                requires_human=not pattern.approve and pattern.confidence < 0.8,
            )

        # No pattern matched - require human approval
        return PermissionDecision(
            approve=False,
            confidence=0.0,
            reason=f"No matching pattern for {tool_name}, requires human approval",
            requires_human=True,
        )

    def _get_input_string(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Extract the primary input string for pattern matching."""
        match tool_name:
            case "Bash":
                return tool_input.get("command", "")
            case "Write" | "Edit" | "Read":
                return tool_input.get("file_path", "")
            case "Glob":
                return tool_input.get("pattern", "")
            case "Grep":
                return tool_input.get("pattern", "")
            case "Task":
                return tool_input.get("prompt", "")
            case _:
                # Fallback: convert entire input to string
                return str(tool_input)

    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold for auto-answering."""
        return self._config.confidence_threshold
