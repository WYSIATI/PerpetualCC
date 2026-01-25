"""Brain factory for creating brain instances from configuration.

This module provides a factory function to create the appropriate
brain implementation based on PerpetualCC configuration.

Usage:
    from perpetualcc.brain.factory import create_brain
    from perpetualcc.config import get_config

    config = get_config()
    brain = create_brain(config.brain)
"""

from __future__ import annotations

import logging

from perpetualcc.brain.base import Brain
from perpetualcc.brain.rule_based import RuleBasedBrain, RuleBasedConfig

logger = logging.getLogger(__name__)


def create_brain(brain_config: "BrainConfig") -> Brain:
    """Create a brain instance based on configuration.

    Args:
        brain_config: Brain configuration from PerpetualCC config

    Returns:
        Configured Brain instance

    Raises:
        ValueError: If brain type is invalid or required dependencies missing
    """
    from perpetualcc.config import BrainType

    brain_type = brain_config.type

    if brain_type == BrainType.RULE_BASED:
        return _create_rule_based_brain(brain_config)
    elif brain_type == BrainType.GEMINI:
        return _create_gemini_brain(brain_config)
    elif brain_type == BrainType.OLLAMA:
        return _create_ollama_brain(brain_config)
    else:
        logger.warning(f"Unknown brain type: {brain_type}, falling back to rule-based")
        return _create_rule_based_brain(brain_config)


def _create_rule_based_brain(brain_config: "BrainConfig") -> RuleBasedBrain:
    """Create a rule-based brain."""
    from perpetualcc.brain.rule_based import (
        default_permission_patterns,
        default_question_patterns,
    )

    config = RuleBasedConfig(
        question_patterns=default_question_patterns(),
        permission_patterns=default_permission_patterns(),
        confidence_threshold=brain_config.confidence_threshold,
    )
    return RuleBasedBrain(config=config)


def _create_gemini_brain(brain_config: "BrainConfig") -> Brain:
    """Create a Gemini brain."""
    try:
        from perpetualcc.brain.gemini import GeminiBrain
    except ImportError as e:
        logger.error(
            f"Failed to import GeminiBrain: {e}. "
            "Install with: pip install perpetualcc[gemini]"
        )
        logger.warning("Falling back to rule-based brain")
        return _create_rule_based_brain(brain_config)

    api_key = brain_config.gemini.get_api_key()
    if not api_key:
        logger.warning("No Gemini API key configured, falling back to rule-based brain")
        return _create_rule_based_brain(brain_config)

    return GeminiBrain(
        api_key=api_key,
        model=brain_config.gemini.model,
        confidence_threshold=brain_config.confidence_threshold,
    )


def _create_ollama_brain(brain_config: "BrainConfig") -> Brain:
    """Create an Ollama brain."""
    try:
        from perpetualcc.brain.ollama import OllamaBrain, check_ollama_available
    except ImportError as e:
        logger.error(
            f"Failed to import OllamaBrain: {e}. "
            "Install with: pip install perpetualcc[ollama]"
        )
        logger.warning("Falling back to rule-based brain")
        return _create_rule_based_brain(brain_config)

    # Check if Ollama is available
    available, message = check_ollama_available(brain_config.ollama.host)
    if not available:
        logger.warning(f"Ollama not available: {message}. Falling back to rule-based brain")
        return _create_rule_based_brain(brain_config)

    return OllamaBrain(
        host=brain_config.ollama.host,
        model=brain_config.ollama.model,
        confidence_threshold=brain_config.confidence_threshold,
        timeout=brain_config.ollama.timeout,
    )


def get_brain_status(brain_config: "BrainConfig") -> dict:
    """Get the status of the configured brain.

    Returns information about whether the brain is available
    and any issues with the configuration.

    Args:
        brain_config: Brain configuration

    Returns:
        Dict with status information:
            - type: Brain type name
            - available: Whether the brain is available
            - message: Status message
            - details: Additional details
    """
    from perpetualcc.config import BrainType

    brain_type = brain_config.type
    result = {
        "type": brain_type.value,
        "available": False,
        "message": "",
        "details": {},
    }

    if brain_type == BrainType.RULE_BASED:
        result["available"] = True
        result["message"] = "Rule-based brain is always available"
        result["details"] = {
            "confidence_threshold": brain_config.confidence_threshold,
        }

    elif brain_type == BrainType.GEMINI:
        try:
            from perpetualcc.brain.gemini import GeminiBrain  # noqa: F401

            api_key = brain_config.gemini.get_api_key()
            if api_key:
                result["available"] = True
                result["message"] = "Gemini brain is configured"
                result["details"] = {
                    "model": brain_config.gemini.model,
                    "api_key_set": True,
                }
            else:
                result["message"] = "Gemini API key not set"
                result["details"] = {
                    "model": brain_config.gemini.model,
                    "api_key_set": False,
                    "hint": "Set GEMINI_API_KEY environment variable or configure in config.toml",
                }
        except ImportError:
            result["message"] = "google-genai package not installed"
            result["details"] = {
                "hint": "Install with: pip install perpetualcc[gemini]",
            }

    elif brain_type == BrainType.OLLAMA:
        try:
            from perpetualcc.brain.ollama import check_ollama_available, list_ollama_models

            available, message = check_ollama_available(brain_config.ollama.host)
            result["available"] = available
            result["message"] = message

            if available:
                models = list_ollama_models(brain_config.ollama.host)
                result["details"] = {
                    "host": brain_config.ollama.host,
                    "model": brain_config.ollama.model,
                    "available_models": models,
                    "model_ready": brain_config.ollama.model in models,
                }
                if brain_config.ollama.model not in models:
                    result["message"] = f"Model {brain_config.ollama.model} not pulled"
                    result["details"]["hint"] = f"Run: ollama pull {brain_config.ollama.model}"
            else:
                result["details"] = {
                    "host": brain_config.ollama.host,
                    "hint": "Start Ollama with: ollama serve",
                }
        except ImportError:
            result["message"] = "ollama package not installed"
            result["details"] = {
                "hint": "Install with: pip install perpetualcc[ollama]",
            }

    return result


# Type annotation import (deferred to avoid circular imports)
if False:  # TYPE_CHECKING
    from perpetualcc.config import BrainConfig
