"""Master Agent - ReAct supervisor loop for orchestrating Claude Code sessions.

The Master Agent follows the ReAct (Reasoning + Acting) pattern:
  THINK -> ACT -> OBSERVE -> (LEARN)

It orchestrates:
- Permission decisions via DecisionEngine
- Question answering via Brain
- Rate limit handling via RateLimitMonitor
- Session state management via SessionManager
- Human intervention via HumanBridge
- Knowledge retrieval and memory systems
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from perpetualcc.brain.base import (
    Brain,
    BrainAnswer,
    PermissionContext,
    QuestionContext,
)
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
from perpetualcc.core.checkpoint import SessionCheckpoint
from perpetualcc.core.decision_engine import DecisionEngine, PermissionDecision
from perpetualcc.core.rate_limit import RateLimitInfo, RateLimitMonitor
from perpetualcc.core.risk_classifier import RiskLevel

if TYPE_CHECKING:
    from perpetualcc.core.session_manager import ManagedSession, SessionManager
    from perpetualcc.human.escalation import HumanBridge

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Type of event analysis result."""

    PERMISSION_REQUEST = "permission_request"
    QUESTION = "question"
    RATE_LIMIT = "rate_limit"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "error"
    SESSION_INIT = "session_init"
    TEXT_OUTPUT = "text_output"
    THINKING = "thinking"
    TOOL_RESULT = "tool_result"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Type of action to take in response to an event."""

    APPROVE_TOOL = "approve_tool"
    DENY_TOOL = "deny_tool"
    ANSWER_QUESTION = "answer"
    ESCALATE_TO_HUMAN = "escalate"
    CHECKPOINT_AND_WAIT = "checkpoint_and_wait"
    NEXT_TASK = "next_task"
    SESSION_COMPLETE = "session_complete"
    SESSION_ERROR = "session_error"
    UPDATE_SESSION = "update_session"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class Analysis:
    """Result of analyzing a Claude Code event.

    The analysis captures:
    - What type of event occurred
    - Relevant context for decision-making
    - Whether this is a novel situation worth learning from

    Attributes:
        type: The type of analysis/event
        event: The original Claude event
        tool_name: For permission requests, the tool being requested
        tool_input: For permission requests, the tool input
        question_text: For questions, the question being asked
        question_options: For questions, available options
        rate_limit_info: For rate limits, the limit details
        result_message: For completion/error, the result message
        is_error: Whether this is an error condition
        is_novel: Whether this situation is worth learning from
        context: Additional context data
    """

    type: AnalysisType
    event: ClaudeEvent
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    question_text: str | None = None
    question_options: list[dict[str, str]] = field(default_factory=list)
    rate_limit_info: RateLimitInfo | None = None
    result_message: str | None = None
    is_error: bool = False
    is_novel: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """Action to take in response to an analyzed event.

    Attributes:
        type: The type of action to take
        value: The action value (e.g., answer text, denial reason)
        reason: Explanation for this action
        confidence: Confidence level (0.0-1.0) for this action
        requires_human: Whether human intervention is needed
        metadata: Additional action metadata
    """

    type: ActionType
    value: str | None = None
    reason: str | None = None
    confidence: float = 1.0
    requires_human: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A recorded episode for learning/memory systems.

    Episodes capture what happened and how we responded, enabling
    future learning and improvement.

    Attributes:
        timestamp: When this episode occurred
        session_id: The session this occurred in
        event_type: Type of event that triggered this episode
        context: What was happening (event summary)
        action_taken: What action we took
        action_reason: Why we took this action
        outcome: Success/failure/pending
        confidence: Confidence level of the action
        metadata: Additional episode data
    """

    timestamp: datetime
    session_id: str
    event_type: str
    context: str
    action_taken: str
    action_reason: str
    outcome: str = "pending"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeEngine(Protocol):
    """Protocol for knowledge engine integration (Phase 7)."""

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant context for a query."""
        ...


class MemoryStore(Protocol):
    """Protocol for memory store integration (Phase 8)."""

    async def record_episode(self, episode: Episode) -> None:
        """Record an episode for future learning."""
        ...

    async def find_similar(self, context: str, top_k: int = 3) -> list[Episode]:
        """Find similar past episodes."""
        ...


@dataclass(frozen=True)
class MasterAgentConfig:
    """Configuration for the Master Agent.

    Attributes:
        confidence_threshold: Minimum confidence for auto-actions (default 0.7)
        max_retries: Maximum retries for transient errors (default 3)
        retry_delay_seconds: Base delay between retries (default 5)
        retry_backoff_multiplier: Exponential backoff multiplier (default 2.0)
        auto_approve_low_risk: Whether to auto-approve low-risk operations
        block_high_risk: Whether to auto-block high-risk operations
        record_episodes: Whether to record episodes for learning
    """

    confidence_threshold: float = 0.7
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0
    auto_approve_low_risk: bool = True
    block_high_risk: bool = True
    record_episodes: bool = True


class MasterAgent:
    """
    The supervisor that orchestrates Claude Code sessions.

    Follows a ReAct (Reasoning + Acting) loop:
      THINK -> ACT -> OBSERVE -> (LEARN)

    The MasterAgent coordinates:
    - Permission decisions for tool use requests
    - Question answering with optional escalation
    - Rate limit detection and recovery
    - Session completion and task queue management
    - Human intervention for escalated decisions
    - Episode recording for future learning

    Integration points:
    - SessionManager: Multi-session coordination
    - DecisionEngine: Permission decisions
    - Brain: Question answering
    - RateLimitMonitor: Rate limit handling
    - HumanBridge: Human intervention
    - KnowledgeEngine: Context retrieval
    - MemoryStore: Episode storage and retrieval
    """

    def __init__(
        self,
        session_manager: SessionManager,
        decision_engine: DecisionEngine,
        brain: Brain | None = None,
        rate_monitor: RateLimitMonitor | None = None,
        human_bridge: HumanBridge | None = None,
        knowledge_engine: KnowledgeEngine | None = None,
        memory: MemoryStore | None = None,
        config: MasterAgentConfig | None = None,
    ):
        """
        Initialize the Master Agent.

        Args:
            session_manager: Manager for session lifecycle
            decision_engine: Engine for permission decisions
            brain: Optional brain for answering questions
            rate_monitor: Optional monitor for rate limit handling
            human_bridge: Optional bridge for human intervention
            knowledge_engine: Optional knowledge engine for context
            memory: Optional memory store for learning
            config: Optional configuration
        """
        self.session_manager = session_manager
        self.decision_engine = decision_engine
        self.brain = brain
        self.rate_monitor = rate_monitor or RateLimitMonitor()
        self.human_bridge = human_bridge
        self.knowledge_engine = knowledge_engine
        self.memory = memory
        self.config = config or MasterAgentConfig()

        # Runtime state
        self._episodes: list[Episode] = []
        self._retry_counts: dict[str, int] = {}
        self._recent_tools: list[str] = []
        self._modified_files: list[str] = []

    async def run_session(self, session: ManagedSession) -> None:
        """
        Main event loop for a session.

        This implements the ReAct loop:
        1. THINK: Analyze incoming event
        2. ACT: Decide and execute appropriate action
        3. OBSERVE: Next event in stream (implicit)
        4. LEARN: Record episode if novel

        Args:
            session: The managed session to run
        """
        logger.info("MasterAgent starting session: %s", session.id)

        # Reset session-specific state
        self._recent_tools = []
        self._modified_files = []
        self._retry_counts.clear()

        try:
            async for event in self.session_manager.stream_events(session):
                # THINK: What's happening?
                analysis = await self._think(event, session)

                # ACT: What should we do?
                action = await self._decide(analysis, session)

                # EXECUTE: Perform the action
                result = await self._execute(action, session)

                # LEARN: Should we remember this?
                if self.config.record_episodes and analysis.is_novel:
                    await self._learn(analysis, action, result, session)

                # Check for session completion
                if action.type in (
                    ActionType.SESSION_COMPLETE,
                    ActionType.SESSION_ERROR,
                ):
                    break

        except asyncio.CancelledError:
            logger.info("MasterAgent session cancelled: %s", session.id)
            raise

        except Exception as e:
            logger.exception("MasterAgent error in session %s: %s", session.id, e)
            # Record error episode
            if self.config.record_episodes:
                await self._record_error_episode(session, str(e))
            raise

        logger.info("MasterAgent finished session: %s", session.id)

    async def _think(self, event: ClaudeEvent, session: ManagedSession) -> Analysis:
        """
        Analyze the current event (THINK phase).

        This method:
        1. Classifies the event type
        2. Extracts relevant information
        3. Gathers additional context if available
        4. Determines if this is a novel situation

        Args:
            event: The Claude event to analyze
            session: Current session context

        Returns:
            Analysis of the event
        """
        # Classify event and extract information
        if isinstance(event, ToolUseEvent):
            return self._analyze_tool_use(event, session)

        elif isinstance(event, AskQuestionEvent):
            return await self._analyze_question(event, session)

        elif isinstance(event, RateLimitEvent):
            return self._analyze_rate_limit(event, session)

        elif isinstance(event, ResultEvent):
            return self._analyze_result(event, session)

        elif isinstance(event, InitEvent):
            return Analysis(
                type=AnalysisType.SESSION_INIT,
                event=event,
                context={"session_id": event.session_id},
            )

        elif isinstance(event, TextEvent):
            return Analysis(
                type=AnalysisType.TEXT_OUTPUT,
                event=event,
                context={"text": event.text[:200]},  # Truncate for context
            )

        elif isinstance(event, ThinkingEvent):
            return Analysis(
                type=AnalysisType.THINKING,
                event=event,
                context={"thinking": event.thinking[:200]},
            )

        elif isinstance(event, ToolResultEvent):
            return Analysis(
                type=AnalysisType.TOOL_RESULT,
                event=event,
                is_error=event.is_error,
                context={
                    "tool_use_id": event.tool_use_id,
                    "is_error": event.is_error,
                },
            )

        else:
            return Analysis(
                type=AnalysisType.UNKNOWN,
                event=event,
                context={"event_type": event.type},
            )

    def _analyze_tool_use(self, event: ToolUseEvent, session: ManagedSession) -> Analysis:
        """Analyze a tool use request."""
        # Track recent tools for context
        self._recent_tools.append(event.tool_name)
        if len(self._recent_tools) > 20:
            self._recent_tools = self._recent_tools[-20:]

        # Track modified files
        if event.tool_name in ("Write", "Edit"):
            file_path = event.tool_input.get("file_path", "")
            if file_path and file_path not in self._modified_files:
                self._modified_files.append(file_path)

        # Determine if novel (first time seeing this pattern)
        is_novel = event.tool_name not in self._recent_tools[:-1]

        return Analysis(
            type=AnalysisType.PERMISSION_REQUEST,
            event=event,
            tool_name=event.tool_name,
            tool_input=event.tool_input,
            is_novel=is_novel,
            context={
                "tool_use_id": event.tool_use_id,
                "recent_tools": self._recent_tools[-5:],
            },
        )

    async def _analyze_question(self, event: AskQuestionEvent, session: ManagedSession) -> Analysis:
        """Analyze a question event."""
        # Extract question details
        questions = event.questions
        if not questions:
            return Analysis(
                type=AnalysisType.QUESTION,
                event=event,
                question_text="",
                question_options=[],
            )

        # Take first question (most common case)
        first_question = questions[0]
        question_text = first_question.question
        options = [
            {"label": opt.label, "description": opt.description} for opt in first_question.options
        ]

        # Questions are generally novel
        is_novel = True

        # Gather additional context from knowledge engine if available
        context: dict[str, Any] = {
            "tool_use_id": event.tool_use_id,
            "multi_select": first_question.multi_select,
        }

        if self.knowledge_engine:
            try:
                rag_results = await self.knowledge_engine.retrieve(question_text, top_k=3)
                context["rag_context"] = rag_results
            except Exception as e:
                logger.warning("Failed to retrieve RAG context: %s", e)

        return Analysis(
            type=AnalysisType.QUESTION,
            event=event,
            question_text=question_text,
            question_options=options,
            is_novel=is_novel,
            context=context,
        )

    def _analyze_rate_limit(self, event: RateLimitEvent, session: ManagedSession) -> Analysis:
        """Analyze a rate limit event."""
        info = self.rate_monitor.detect(event)

        return Analysis(
            type=AnalysisType.RATE_LIMIT,
            event=event,
            rate_limit_info=info,
            is_novel=True,  # Rate limits are always noteworthy
            context={
                "retry_after": event.retry_after,
                "message": event.message,
            },
        )

    def _analyze_result(self, event: ResultEvent, session: ManagedSession) -> Analysis:
        """Analyze a session result event."""
        # Check if this is actually a rate limit in disguise
        if event.is_error and event.result:
            rate_info = self.rate_monitor.detect_from_message(event.result)
            if rate_info:
                return Analysis(
                    type=AnalysisType.RATE_LIMIT,
                    event=event,
                    rate_limit_info=rate_info,
                    is_error=True,
                    is_novel=True,
                    context={"message": event.result},
                )

        analysis_type = AnalysisType.TASK_ERROR if event.is_error else AnalysisType.TASK_COMPLETE

        return Analysis(
            type=analysis_type,
            event=event,
            result_message=event.result,
            is_error=event.is_error,
            is_novel=event.is_error,  # Errors are novel
            context={
                "num_turns": event.num_turns,
                "total_cost_usd": event.total_cost_usd,
                "duration_ms": event.duration_ms,
            },
        )

    async def _decide(self, analysis: Analysis, session: ManagedSession) -> Action:
        """
        Decide what action to take (ACT phase - decision).

        Routes to appropriate handler based on analysis type.

        Args:
            analysis: The event analysis
            session: Current session context

        Returns:
            Action to execute
        """
        match analysis.type:
            case AnalysisType.PERMISSION_REQUEST:
                return await self._decide_permission(analysis, session)

            case AnalysisType.QUESTION:
                return await self._decide_question(analysis, session)

            case AnalysisType.RATE_LIMIT:
                return self._decide_rate_limit(analysis, session)

            case AnalysisType.TASK_COMPLETE:
                return self._decide_task_complete(analysis, session)

            case AnalysisType.TASK_ERROR:
                return self._decide_task_error(analysis, session)

            case AnalysisType.SESSION_INIT:
                return Action(
                    type=ActionType.UPDATE_SESSION,
                    value=analysis.context.get("session_id"),
                    reason="Session initialized",
                )

            case AnalysisType.TEXT_OUTPUT | AnalysisType.THINKING | AnalysisType.TOOL_RESULT:
                return Action(
                    type=ActionType.NO_ACTION,
                    reason=f"Passive event: {analysis.type.value}",
                )

            case _:
                return Action(
                    type=ActionType.NO_ACTION,
                    reason=f"Unknown event type: {analysis.type.value}",
                )

    async def _decide_permission(self, analysis: Analysis, session: ManagedSession) -> Action:
        """Decide on a permission request."""
        tool_name = analysis.tool_name or ""
        tool_input = analysis.tool_input or {}

        # Build permission context
        context = PermissionContext(
            project_path=session.project_path,
            current_task=session.current_task,
            session_id=session.id,
            recent_tools=self._recent_tools[-10:],
            modified_files=self._modified_files[-20:],
        )

        # Use async decision engine with brain support
        decision = await self.decision_engine.decide_permission_async(
            tool_name=tool_name,
            tool_input=tool_input,
            context=context,
            session_id=session.id,
        )

        if decision.approve:
            return Action(
                type=ActionType.APPROVE_TOOL,
                reason=decision.reason,
                confidence=decision.confidence,
                metadata={
                    "tool_name": tool_name,
                    "risk_level": decision.risk_level.value if decision.risk_level else None,
                },
            )
        elif decision.requires_human:
            return Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                reason=decision.reason,
                confidence=decision.confidence,
                requires_human=True,
                metadata={
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "risk_level": decision.risk_level.value if decision.risk_level else None,
                },
            )
        else:
            return Action(
                type=ActionType.DENY_TOOL,
                value=decision.reason,
                reason=decision.reason,
                confidence=decision.confidence,
                metadata={
                    "tool_name": tool_name,
                    "risk_level": decision.risk_level.value if decision.risk_level else None,
                },
            )

    async def _decide_question(self, analysis: Analysis, session: ManagedSession) -> Action:
        """Decide how to answer a question."""
        if not self.brain:
            # No brain available - escalate
            return Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                reason="No brain configured for question answering",
                requires_human=True,
                metadata={
                    "question": analysis.question_text,
                    "options": analysis.question_options,
                },
            )

        # Build question context
        context = QuestionContext(
            project_path=session.project_path,
            question=analysis.question_text or "",
            options=analysis.question_options,
            current_task=session.current_task,
            session_id=session.id,
        )

        # Get answer from brain
        answer = await self.brain.answer_question(
            question=analysis.question_text or "",
            options=analysis.question_options,
            context=context,
        )

        # Check confidence threshold
        if answer.confidence >= self.config.confidence_threshold:
            return Action(
                type=ActionType.ANSWER_QUESTION,
                value=answer.selected,
                reason=answer.reasoning,
                confidence=answer.confidence,
                metadata={
                    "question": analysis.question_text,
                    "selected": answer.selected,
                },
            )
        else:
            # Low confidence - escalate
            return Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                value=answer.selected,  # Include suggestion
                reason=f"Low confidence ({answer.confidence:.2f}): {answer.reasoning}",
                confidence=answer.confidence,
                requires_human=True,
                metadata={
                    "question": analysis.question_text,
                    "options": analysis.question_options,
                    "suggestion": answer.selected,
                    "suggestion_confidence": answer.confidence,
                },
            )

    def _decide_rate_limit(self, analysis: Analysis, session: ManagedSession) -> Action:
        """Decide how to handle a rate limit."""
        return Action(
            type=ActionType.CHECKPOINT_AND_WAIT,
            reason="Rate limit detected, will checkpoint and wait",
            metadata={
                "rate_limit_info": analysis.rate_limit_info,
                "retry_after": (
                    analysis.rate_limit_info.retry_after_seconds if analysis.rate_limit_info else 60
                ),
            },
        )

    def _decide_task_complete(self, analysis: Analysis, session: ManagedSession) -> Action:
        """Decide what to do when a task completes."""
        # Check if there are more tasks in queue
        from perpetualcc.core.task_queue import TaskStatus as QueueTaskStatus

        pending_tasks = self.session_manager.task_queue.list_tasks(
            session_id=session.id, status=QueueTaskStatus.PENDING
        )

        if pending_tasks:
            return Action(
                type=ActionType.NEXT_TASK,
                reason=f"Task completed, {len(pending_tasks)} tasks remaining",
                metadata={
                    "completed_task": session.current_task,
                    "next_task": pending_tasks[0].description if pending_tasks else None,
                    "num_turns": analysis.context.get("num_turns"),
                    "total_cost_usd": analysis.context.get("total_cost_usd"),
                },
            )
        else:
            return Action(
                type=ActionType.SESSION_COMPLETE,
                reason="All tasks completed successfully",
                metadata={
                    "num_turns": analysis.context.get("num_turns"),
                    "total_cost_usd": analysis.context.get("total_cost_usd"),
                },
            )

    def _decide_task_error(self, analysis: Analysis, session: ManagedSession) -> Action:
        """Decide what to do when a task errors."""
        error_key = f"{session.id}:error"
        retry_count = self._retry_counts.get(error_key, 0)

        if retry_count < self.config.max_retries:
            # Retry with backoff
            self._retry_counts[error_key] = retry_count + 1
            return Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                value=analysis.result_message,
                reason=f"Error occurred (retry {retry_count + 1}/{self.config.max_retries})",
                requires_human=True,
                metadata={
                    "error_message": analysis.result_message,
                    "retry_count": retry_count + 1,
                },
            )
        else:
            return Action(
                type=ActionType.SESSION_ERROR,
                value=analysis.result_message,
                reason=f"Max retries ({self.config.max_retries}) exceeded",
                metadata={
                    "error_message": analysis.result_message,
                    "retry_count": retry_count,
                },
            )

    async def _execute(self, action: Action, session: ManagedSession) -> dict[str, Any]:
        """
        Execute the decided action (ACT phase - execution).

        Args:
            action: The action to execute
            session: Current session context

        Returns:
            Result of the action execution
        """
        result: dict[str, Any] = {
            "action_type": action.type.value,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            match action.type:
                case ActionType.APPROVE_TOOL:
                    result["action"] = "tool_approved"
                    logger.info(
                        "Approved tool: %s (confidence=%.2f)",
                        action.metadata.get("tool_name"),
                        action.confidence,
                    )

                case ActionType.DENY_TOOL:
                    result["action"] = "tool_denied"
                    result["reason"] = action.value
                    logger.warning(
                        "Denied tool: %s - %s",
                        action.metadata.get("tool_name"),
                        action.value,
                    )

                case ActionType.ANSWER_QUESTION:
                    result["action"] = "question_answered"
                    result["answer"] = action.value
                    logger.info(
                        "Answered question with '%s' (confidence=%.2f)",
                        action.value,
                        action.confidence,
                    )
                    # Send response through session manager for continuous conversation
                    if action.value:
                        events_processed = await self._send_question_response(
                            session, action.value
                        )
                        result["events_processed"] = events_processed

                case ActionType.ESCALATE_TO_HUMAN:
                    result["action"] = "escalated_to_human"
                    result["reason"] = action.reason
                    logger.info(
                        "Escalated to human: %s",
                        action.reason,
                    )
                    # Trigger human intervention if HumanBridge is configured
                    if self.human_bridge:
                        response = await self._handle_human_escalation(
                            action, session
                        )
                        result["human_response"] = response

                case ActionType.CHECKPOINT_AND_WAIT:
                    result["action"] = "checkpoint_and_wait"
                    await self._handle_checkpoint_and_wait(action, session)

                case ActionType.NEXT_TASK:
                    result["action"] = "starting_next_task"
                    logger.info(
                        "Moving to next task: %s",
                        action.metadata.get("next_task"),
                    )

                case ActionType.SESSION_COMPLETE:
                    result["action"] = "session_completed"
                    logger.info(
                        "Session completed: turns=%s, cost=$%s",
                        action.metadata.get("num_turns"),
                        action.metadata.get("total_cost_usd"),
                    )

                case ActionType.SESSION_ERROR:
                    result["action"] = "session_error"
                    result["error"] = action.value
                    result["success"] = False
                    logger.error(
                        "Session error: %s",
                        action.value,
                    )

                case ActionType.UPDATE_SESSION:
                    result["action"] = "session_updated"
                    # Session state updates are handled by SessionManager

                case ActionType.NO_ACTION:
                    result["action"] = "no_action"

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.exception("Error executing action %s: %s", action.type.value, e)

        return result

    async def _send_question_response(
        self,
        session: ManagedSession,
        response: str,
    ) -> int:
        """Send a response to a question via session manager.

        This enables continuous conversation by sending the answer back
        to Claude and processing the resulting events.

        Args:
            session: The session to respond to
            response: The response text

        Returns:
            Number of events processed
        """
        event_count = 0
        try:
            async for event in self.session_manager.send_response(session.id, response):
                event_count += 1

                # Process resulting events in the ReAct loop
                analysis = await self._think(event, session)
                action = await self._decide(analysis, session)
                result = await self._execute(action, session)

                # Record episode if novel
                if self.config.record_episodes and analysis.is_novel:
                    await self._learn(analysis, action, result, session)

                # Check for session completion
                if action.type in (
                    ActionType.SESSION_COMPLETE,
                    ActionType.SESSION_ERROR,
                ):
                    break

        except Exception as e:
            logger.warning("Error processing question response events: %s", e)

        return event_count

    async def _handle_checkpoint_and_wait(self, action: Action, session: ManagedSession) -> None:
        """Handle checkpoint creation and rate limit waiting."""
        rate_limit_info = action.metadata.get("rate_limit_info")
        retry_after = action.metadata.get("retry_after", 60)

        # Create checkpoint via session manager
        # (The actual checkpoint is created in SessionManager._handle_rate_limit)
        logger.info(
            "Rate limit checkpoint: waiting %d seconds",
            retry_after,
        )

        # Wait for reset with progress logging
        async def progress_callback(remaining: int, total: int) -> None:
            if remaining % 30 == 0 or remaining <= 5:
                logger.info("Rate limit wait: %ds remaining", remaining)

        if rate_limit_info:
            await self.rate_monitor.wait_for_reset(
                rate_limit_info,
                progress_callback=lambda r, t: None,  # Simplified for now
            )

    async def _handle_human_escalation(
        self, action: Action, session: ManagedSession
    ) -> str | None:
        """Handle escalation to human via HumanBridge.

        Args:
            action: The escalation action with metadata
            session: Current session context

        Returns:
            Human's response, or None if no HumanBridge configured
        """
        if not self.human_bridge:
            return None

        metadata = action.metadata
        escalation_type = metadata.get("escalation_type", "question")

        if escalation_type == "permission" or "tool_name" in metadata:
            return await self.human_bridge.escalate_permission(
                session=session,
                tool_name=metadata.get("tool_name", "Unknown"),
                tool_input=metadata.get("tool_input", {}),
                risk_level=metadata.get("risk_level", "UNKNOWN"),
                suggestion=action.value,
                confidence=action.confidence,
            )
        elif "error_message" in metadata:
            return await self.human_bridge.escalate_error(
                session=session,
                error_message=metadata.get("error_message", "Unknown error"),
                options=metadata.get("options"),
            )
        else:
            return await self.human_bridge.escalate_question(
                session=session,
                question=metadata.get("question", action.reason or "?"),
                options=[
                    opt.get("label", str(opt))
                    for opt in metadata.get("options", [])
                ],
                suggestion=metadata.get("suggestion") or action.value,
                confidence=metadata.get("suggestion_confidence", action.confidence),
            )

    async def _learn(
        self,
        analysis: Analysis,
        action: Action,
        result: dict[str, Any],
        session: ManagedSession,
    ) -> None:
        """
        Record an episode for learning (LEARN phase).

        This captures the experience for future improvement:
        - What situation was encountered
        - What action was taken
        - What was the outcome

        Args:
            analysis: The event analysis
            action: The action taken
            result: The execution result
            session: Current session context
        """
        # Create episode
        episode = Episode(
            timestamp=datetime.now(),
            session_id=session.id,
            event_type=analysis.type.value,
            context=self._summarize_context(analysis),
            action_taken=action.type.value,
            action_reason=action.reason or "",
            outcome="success" if result.get("success", True) else "failure",
            confidence=action.confidence,
            metadata={
                "tool_name": analysis.tool_name,
                "question": analysis.question_text,
                "action_value": action.value,
            },
        )

        # Store locally
        self._episodes.append(episode)
        if len(self._episodes) > 100:
            self._episodes = self._episodes[-100:]

        # Store in memory system if available
        if self.memory:
            try:
                await self.memory.record_episode(episode)
            except Exception as e:
                logger.warning("Failed to record episode to memory: %s", e)

        logger.debug(
            "Recorded episode: type=%s, action=%s, outcome=%s",
            episode.event_type,
            episode.action_taken,
            episode.outcome,
        )

    async def _record_error_episode(self, session: ManagedSession, error: str) -> None:
        """Record an error episode."""
        episode = Episode(
            timestamp=datetime.now(),
            session_id=session.id,
            event_type="error",
            context=f"Session error: {error}",
            action_taken="error_handling",
            action_reason="Unhandled exception in MasterAgent",
            outcome="failure",
            confidence=0.0,
            metadata={"error": error},
        )
        self._episodes.append(episode)

        if self.memory:
            try:
                await self.memory.record_episode(episode)
            except Exception as e:
                logger.warning("Failed to record error episode: %s", e)

    def _summarize_context(self, analysis: Analysis) -> str:
        """Create a brief summary of the analysis context."""
        match analysis.type:
            case AnalysisType.PERMISSION_REQUEST:
                return f"Tool use request: {analysis.tool_name}"
            case AnalysisType.QUESTION:
                return f"Question: {(analysis.question_text or '')[:100]}"
            case AnalysisType.RATE_LIMIT:
                return "Rate limit detected"
            case AnalysisType.TASK_COMPLETE:
                return "Task completed"
            case AnalysisType.TASK_ERROR:
                return f"Task error: {(analysis.result_message or '')[:100]}"
            case _:
                return f"Event: {analysis.type.value}"

    def get_recent_episodes(self, limit: int = 20) -> list[Episode]:
        """Get recent episodes for inspection/debugging."""
        return list(self._episodes[-limit:])

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about MasterAgent operation."""
        episodes = self._episodes

        action_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {"success": 0, "failure": 0, "pending": 0}

        for ep in episodes:
            action_counts[ep.action_taken] = action_counts.get(ep.action_taken, 0) + 1
            if ep.outcome in outcome_counts:
                outcome_counts[ep.outcome] += 1

        return {
            "total_episodes": len(episodes),
            "action_counts": action_counts,
            "outcome_counts": outcome_counts,
            "recent_tools": self._recent_tools[-10:],
            "modified_files_count": len(self._modified_files),
        }
