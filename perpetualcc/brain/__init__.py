"""Brain layer - pluggable LLM backends for intelligent decision making.

Available brain implementations:
    - RuleBasedBrain: Pattern matching, no external AI (default)
    - GeminiBrain: Google Gemini API (requires google-genai package)
    - OllamaBrain: Local LLM via Ollama (requires ollama package)

Use the factory to create brains from configuration:
    from perpetualcc.brain.factory import create_brain
    from perpetualcc.config import get_config

    config = get_config()
    brain = create_brain(config.brain)
"""

from perpetualcc.brain.base import (
    Brain,
    BrainAnswer,
    PermissionContext,
    PermissionDecision,
    QuestionContext,
)
from perpetualcc.brain.factory import create_brain, get_brain_status
from perpetualcc.brain.rule_based import (
    PermissionPattern,
    QuestionPattern,
    RuleBasedBrain,
    RuleBasedConfig,
    default_permission_patterns,
    default_question_patterns,
)

# Optional imports for Gemini and Ollama brains
# These are only available if the respective packages are installed
try:
    from perpetualcc.brain.gemini import GeminiBrain
except ImportError:
    GeminiBrain = None  # type: ignore

try:
    from perpetualcc.brain.ollama import OllamaBrain, check_ollama_available, list_ollama_models
except ImportError:
    OllamaBrain = None  # type: ignore
    check_ollama_available = None  # type: ignore
    list_ollama_models = None  # type: ignore

__all__ = [
    # Base
    "Brain",
    "BrainAnswer",
    "PermissionContext",
    "PermissionDecision",
    "QuestionContext",
    # Factory
    "create_brain",
    "get_brain_status",
    # Rule-based
    "RuleBasedBrain",
    "RuleBasedConfig",
    "QuestionPattern",
    "PermissionPattern",
    "default_question_patterns",
    "default_permission_patterns",
    # Optional (may be None if not installed)
    "GeminiBrain",
    "OllamaBrain",
    "check_ollama_available",
    "list_ollama_models",
]
