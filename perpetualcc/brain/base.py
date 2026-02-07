"""Base interface for Brain implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BrainAnswer:
    """Answer from the brain for a question."""

    selected: str | None  # Selected option label or custom answer
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Explanation for the answer


@dataclass(frozen=True)
class PermissionContext:
    """Context for permission evaluation."""

    project_path: str
    current_task: str | None = None
    session_id: str | None = None
    recent_tools: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    rag_context: list[dict[str, Any]] = field(default_factory=list)  # RAG results


@dataclass(frozen=True)
class QuestionContext:
    """Context for answering a question."""

    project_path: str
    question: str
    options: list[dict[str, str]]  # [{label, description}]
    current_task: str | None = None
    session_id: str | None = None
    requirements_text: str | None = None
    rag_context: list[dict[str, Any]] = field(default_factory=list)  # RAG results


# Import PermissionDecision from decision engine to avoid circular imports
# We use a local definition here that matches the expected interface
@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission decision (local copy for brain interface)."""

    approve: bool
    confidence: float
    reason: str
    requires_human: bool = False


@dataclass(frozen=True)
class PlanReviewResult:
    """Result of a plan review by the brain.
    
    The brain reviews the output of a planning session and decides:
    - iterate: Plan needs refinement, run another planning session
    - execute: Plan is ready, proceed with execution
    """

    decision: str  # "iterate" | "execute"
    feedback: str  # Explanation or improvement suggestions
    confidence: float  # 0.0 - 1.0


class Brain(ABC):
    """
    Abstract base class for brain implementations.

    A brain is responsible for:
    1. Answering questions from Claude Code
    2. Evaluating permission requests for medium-risk operations

    Implementations include:
    - RuleBasedBrain: Pattern matching, no external AI
    - GeminiBrain: Uses Google Gemini API
    - OllamaBrain: Uses local LLM via Ollama
    """

    @abstractmethod
    async def answer_question(
        self,
        question: str,
        options: list[dict[str, str]],
        context: QuestionContext,
    ) -> BrainAnswer:
        """
        Answer a question from Claude Code.

        Args:
            question: The question text
            options: Available options [{label, description}]
            context: Additional context for answering

        Returns:
            BrainAnswer with selected option, confidence, and reasoning
        """
        ...

    @abstractmethod
    async def evaluate_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """
        Evaluate a medium-risk permission request.

        This is called by the DecisionEngine when a tool use request
        is classified as MEDIUM risk and needs intelligent evaluation.

        Args:
            tool_name: Name of the tool (Write, Edit, Bash, etc.)
            tool_input: Input parameters for the tool
            context: Session and project context

        Returns:
            PermissionDecision indicating whether to approve
        """
        ...

    @abstractmethod
    async def review_plan(
        self,
        plan_content: str,
        original_task: str,
        context: QuestionContext,
    ) -> PlanReviewResult:
        """
        Review the output of a planning session.

        Called after a planning session completes to decide whether to:
        - iterate: Run another planning session to refine the plan
        - execute: Proceed with executing the plan

        Args:
            plan_content: The plan output from Claude Code
            original_task: The original task description
            context: Session and project context

        Returns:
            PlanReviewResult with decision, feedback, and confidence
        """
        ...

    def get_confidence_threshold(self) -> float:
        """
        Get the minimum confidence threshold for auto-answering.

        Returns:
            Confidence threshold (default: 0.7)
        """
        return 0.7
