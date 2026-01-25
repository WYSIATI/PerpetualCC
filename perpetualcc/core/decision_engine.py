"""Decision engine for tool use permissions in Claude Code sessions."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from perpetualcc.core.risk_classifier import (
    RiskClassification,
    RiskClassifier,
    RiskConfig,
    RiskLevel,
)

if TYPE_CHECKING:
    from perpetualcc.brain.base import PermissionContext

# Type alias for SDK permission callback
SDKPermissionCallback = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResultAllow | PermissionResultDeny],
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission decision for a tool use request."""

    approve: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    risk_level: RiskLevel | None = None
    requires_human: bool = False


@dataclass
class DecisionRecord:
    """Record of a permission decision for logging/auditing."""

    timestamp: datetime
    tool_name: str
    tool_input: dict[str, Any]
    risk_classification: RiskClassification
    decision: PermissionDecision
    session_id: str | None = None


class PermissionCallback(Protocol):
    """Protocol for permission evaluation callbacks."""

    async def evaluate_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """Evaluate a permission request and return a decision."""
        ...


class DecisionEngine:
    """
    Engine for making permission decisions on tool use requests.

    The decision engine uses a risk classifier to determine the risk level
    of each tool use request, then routes to the appropriate handler:
    - LOW risk: Auto-approve with high confidence
    - MEDIUM risk: Use brain (if available) or escalate
    - HIGH risk: Block or escalate to human
    """

    def __init__(
        self,
        project_path: str | Path,
        brain: PermissionCallback | None = None,
        risk_config: RiskConfig | None = None,
        auto_approve_low_risk: bool = True,
        block_high_risk: bool = True,
    ):
        """
        Initialize the decision engine.

        Args:
            project_path: Path to the project directory
            brain: Optional brain for evaluating medium-risk requests
            risk_config: Optional custom risk configuration
            auto_approve_low_risk: Whether to auto-approve low-risk requests
            block_high_risk: Whether to auto-block high-risk requests
        """
        self.project_path = Path(project_path).resolve()
        self.brain = brain
        self.auto_approve_low_risk = auto_approve_low_risk
        self.block_high_risk = block_high_risk
        self.classifier = RiskClassifier(project_path, risk_config)
        self._decision_history: list[DecisionRecord] = []

    def decide_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        session_id: str | None = None,
    ) -> PermissionDecision:
        """
        Make a synchronous permission decision for a tool use request.

        This is the main entry point for permission decisions. It classifies
        the risk level and routes to the appropriate handler.

        Args:
            tool_name: Name of the tool (Read, Write, Edit, Bash, etc.)
            tool_input: Input parameters for the tool
            session_id: Optional session ID for logging

        Returns:
            PermissionDecision indicating whether to approve and why
        """
        # Classify the risk
        classification = self.classifier.classify(tool_name, tool_input)

        # Route based on risk level
        decision = self._route_decision(classification, tool_name, tool_input)

        # Record the decision
        self._record_decision(
            tool_name=tool_name,
            tool_input=tool_input,
            classification=classification,
            decision=decision,
            session_id=session_id,
        )

        return decision

    async def decide_permission_async(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: PermissionContext | None = None,
        session_id: str | None = None,
    ) -> PermissionDecision:
        """
        Make an async permission decision, potentially using the brain.

        This method can use the brain for medium-risk decisions that require
        more sophisticated evaluation.

        Args:
            tool_name: Name of the tool
            tool_input: Input parameters for the tool
            context: Optional permission context for brain evaluation
            session_id: Optional session ID for logging

        Returns:
            PermissionDecision indicating whether to approve and why
        """
        # Classify the risk
        classification = self.classifier.classify(tool_name, tool_input)

        # For medium risk with brain available, use async brain evaluation
        if classification.level == RiskLevel.MEDIUM and self.brain and context:
            decision = await self.brain.evaluate_permission(tool_name, tool_input, context)
            # Add risk level to decision if not set
            decision = PermissionDecision(
                approve=decision.approve,
                confidence=decision.confidence,
                reason=decision.reason,
                risk_level=RiskLevel.MEDIUM,
                requires_human=decision.requires_human,
            )
        else:
            # Use synchronous routing
            decision = self._route_decision(classification, tool_name, tool_input)

        # Record the decision
        self._record_decision(
            tool_name=tool_name,
            tool_input=tool_input,
            classification=classification,
            decision=decision,
            session_id=session_id,
        )

        return decision

    def _route_decision(
        self,
        classification: RiskClassification,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionDecision:
        """Route to the appropriate decision handler based on risk level."""
        match classification.level:
            case RiskLevel.LOW:
                return self._handle_low_risk(classification)
            case RiskLevel.MEDIUM:
                return self._handle_medium_risk(classification)
            case RiskLevel.HIGH:
                return self._handle_high_risk(classification)

    def _handle_low_risk(self, classification: RiskClassification) -> PermissionDecision:
        """Handle low-risk requests - auto-approve if configured."""
        if self.auto_approve_low_risk:
            return PermissionDecision(
                approve=True,
                confidence=0.95,
                reason=f"Auto-approved: {classification.reason}",
                risk_level=RiskLevel.LOW,
            )
        else:
            return PermissionDecision(
                approve=False,
                confidence=0.0,
                reason="Auto-approve disabled, requires human approval",
                risk_level=RiskLevel.LOW,
                requires_human=True,
            )

    def _handle_medium_risk(self, classification: RiskClassification) -> PermissionDecision:
        """Handle medium-risk requests - escalate to human if no brain."""
        # Without a brain or async context, we need human approval
        return PermissionDecision(
            approve=False,
            confidence=0.0,
            reason=f"Medium risk - requires evaluation: {classification.reason}",
            risk_level=RiskLevel.MEDIUM,
            requires_human=True,
        )

    def _handle_high_risk(self, classification: RiskClassification) -> PermissionDecision:
        """Handle high-risk requests - block if configured."""
        if self.block_high_risk:
            return PermissionDecision(
                approve=False,
                confidence=0.99,
                reason=f"Blocked: {classification.reason}",
                risk_level=RiskLevel.HIGH,
            )
        else:
            return PermissionDecision(
                approve=False,
                confidence=0.0,
                reason=f"High risk - requires human approval: {classification.reason}",
                risk_level=RiskLevel.HIGH,
                requires_human=True,
            )

    def _record_decision(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        classification: RiskClassification,
        decision: PermissionDecision,
        session_id: str | None,
    ) -> None:
        """Record a decision for logging and auditing."""
        record = DecisionRecord(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_input=tool_input,
            risk_classification=classification,
            decision=decision,
            session_id=session_id,
        )
        self._decision_history.append(record)

        # Log the decision
        level = logging.INFO if decision.approve else logging.WARNING
        logger.log(
            level,
            "Permission decision: %s %s -> %s (risk=%s, confidence=%.2f)",
            tool_name,
            self._summarize_input(tool_name, tool_input),
            "APPROVED" if decision.approve else "DENIED",
            classification.level.value,
            decision.confidence,
        )

    def _summarize_input(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Create a brief summary of tool input for logging."""
        match tool_name:
            case "Read" | "Write" | "Edit":
                return tool_input.get("file_path", "?")
            case "Bash":
                cmd = tool_input.get("command", "?")
                return cmd[:50] + "..." if len(cmd) > 50 else cmd
            case "Glob":
                return tool_input.get("pattern", "?")
            case "Grep":
                return f"/{tool_input.get('pattern', '?')}/"
            case _:
                return str(tool_input)[:50]

    def get_decision_history(
        self, session_id: str | None = None, limit: int | None = None
    ) -> list[DecisionRecord]:
        """
        Get decision history, optionally filtered by session.

        Args:
            session_id: Optional filter by session ID
            limit: Optional limit on number of records

        Returns:
            List of DecisionRecord objects
        """
        history = self._decision_history
        if session_id:
            history = [r for r in history if r.session_id == session_id]
        if limit:
            history = history[-limit:]
        return list(history)

    def get_statistics(self, session_id: str | None = None) -> dict[str, Any]:
        """
        Get statistics about permission decisions.

        Args:
            session_id: Optional filter by session ID

        Returns:
            Dictionary with decision statistics
        """
        history = self.get_decision_history(session_id)
        total = len(history)
        if total == 0:
            return {
                "total": 0,
                "approved": 0,
                "denied": 0,
                "by_risk_level": {},
            }

        approved = sum(1 for r in history if r.decision.approve)
        by_risk = {}
        for level in RiskLevel:
            level_records = [r for r in history if r.risk_classification.level == level]
            by_risk[level.value] = {
                "total": len(level_records),
                "approved": sum(1 for r in level_records if r.decision.approve),
            }

        return {
            "total": total,
            "approved": approved,
            "denied": total - approved,
            "approval_rate": approved / total if total > 0 else 0,
            "by_risk_level": by_risk,
        }


def create_permission_callback(
    decision_engine: DecisionEngine,
    context: PermissionContext | None = None,
) -> SDKPermissionCallback:
    """
    Create an async callback function compatible with ClaudeSDKClient's can_use_tool.

    This creates an async callback that returns PermissionResultAllow or
    PermissionResultDeny, which is the proper return type for ClaudeSDKClient.

    Args:
        decision_engine: The decision engine to use
        context: Optional permission context for brain evaluation

    Returns:
        An async callback function for permission decisions
    """

    async def permission_callback(
        tool_name: str,
        input_data: dict[str, Any],
        sdk_context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Async callback for ClaudeSDKClient permission handling."""
        # Use async decision method to support brain evaluation
        decision = await decision_engine.decide_permission_async(
            tool_name=tool_name,
            tool_input=input_data,
            context=context,
        )

        if decision.approve:
            return PermissionResultAllow(updated_input=input_data)

        return PermissionResultDeny(
            message=decision.reason,
            interrupt=decision.requires_human,
        )

    return permission_callback
