"""Human bridge - escalation, notifications, and interactive prompts.

This module provides the human intervention infrastructure for PerpetualCC:
- Escalation queue for requesting human decisions
- macOS notifications for alerting humans
- Interactive CLI prompts for responding to escalations

Usage:
    from perpetualcc.human import HumanBridge, EscalationRequest

    bridge = HumanBridge()
    await bridge.initialize()

    # Escalate a question
    response = await bridge.escalate_question(
        session=session,
        question="Which database?",
        options=["PostgreSQL", "SQLite"],
    )
"""

from perpetualcc.human.cli_prompt import (
    InteractiveConfig,
    InteractiveContext,
    InteractivePrompt,
    SignalHandler,
    UserAction,
)
from perpetualcc.human.escalation import (
    EscalationQueue,
    EscalationRequest,
    EscalationStatus,
    EscalationType,
    HumanBridge,
)
from perpetualcc.human.notifications import (
    NotificationConfig,
    NotificationType,
    Notifier,
)

__all__ = [
    # Escalation
    "EscalationQueue",
    "EscalationRequest",
    "EscalationStatus",
    "EscalationType",
    "HumanBridge",
    # Notifications
    "NotificationConfig",
    "NotificationType",
    "Notifier",
    # CLI Prompts
    "InteractiveConfig",
    "InteractiveContext",
    "InteractivePrompt",
    "SignalHandler",
    "UserAction",
]
