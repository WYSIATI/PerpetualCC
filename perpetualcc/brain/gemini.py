"""Gemini brain implementation using Google Generative AI API.

This brain uses Google's Gemini models to answer questions
and evaluate permission requests with full context awareness.

Requires the google-genai package:
    pip install google-genai
    # or
    pip install perpetualcc[gemini]
"""

from __future__ import annotations

import logging
from typing import Any

from perpetualcc.brain.base import (
    Brain,
    BrainAnswer,
    PermissionContext,
    PermissionDecision,
    QuestionContext,
)

logger = logging.getLogger(__name__)


# System prompts for the Gemini brain
QUESTION_SYSTEM_PROMPT = """You are an intelligent assistant helping to answer questions during a Claude Code session.

Your role is to:
1. Answer routine questions that a senior developer would typically answer
2. Make sensible default choices when asked about options
3. Escalate to human when uncertain or when the question involves significant decisions

Guidelines:
- For "proceed/continue?" questions, answer "Yes" if the context seems normal
- For technical choices, prefer widely-used, well-supported options
- For questions about project conventions, infer from the context if possible
- If the question involves sensitive operations (credentials, production, payments), escalate
- If you're uncertain about the correct answer, escalate to human

Your response MUST be in this exact format:
ANSWER: <your answer or selected option>
CONFIDENCE: <0.0 to 1.0>
REASONING: <brief explanation>

If you think a human should decide, respond with:
ANSWER: ESCALATE
CONFIDENCE: 0.0
REASONING: <why this needs human input>
"""

PERMISSION_SYSTEM_PROMPT = """You are evaluating whether to approve a tool use request during a Claude Code session.

Context: This is a medium-risk operation that needs intelligent evaluation.

Your role is to:
1. Approve operations that are safe and align with normal development workflows
2. Deny operations that could be harmful or risky
3. Escalate to human when uncertain

Consider:
- Is this a common development operation?
- Could this cause data loss or security issues?
- Does the file/command target sensitive areas?
- Is this consistent with the current task?

Your response MUST be in this exact format:
DECISION: APPROVE | DENY | ESCALATE
CONFIDENCE: <0.0 to 1.0>
REASONING: <brief explanation>
"""


class GeminiBrain(Brain):
    """Brain implementation using Google Gemini API.

    Uses Gemini models for context-aware question answering
    and permission evaluation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        confidence_threshold: float = 0.7,
    ):
        """Initialize the Gemini brain.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Gemini model to use.
            confidence_threshold: Minimum confidence for auto-answering.
        """
        self._api_key = api_key
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package is required for Gemini brain. "
                "Install with: pip install google-genai"
            ) from e

        # Get API key
        api_key = self._api_key
        if not api_key:
            import os

            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        self._client = genai.Client(api_key=api_key)
        return self._client

    async def answer_question(
        self,
        question: str,
        options: list[dict[str, str]],
        context: QuestionContext,
    ) -> BrainAnswer:
        """Answer a question from Claude Code using Gemini.

        Args:
            question: The question text
            options: Available options [{label, description}]
            context: Additional context for answering

        Returns:
            BrainAnswer with selected option, confidence, and reasoning
        """
        try:
            client = self._get_client()

            # Build the prompt
            options_text = "\n".join(
                f"  - {opt.get('label', '')}: {opt.get('description', '')}"
                for opt in options
            )

            user_prompt = f"""Project: {context.project_path}
Current Task: {context.current_task or 'Not specified'}

Question: {question}

Available options:
{options_text if options else 'No specific options provided - answer freely'}

{f'Requirements context: {context.requirements_text[:500]}...' if context.requirements_text else ''}

Please analyze this question and provide your response."""

            # Make the API call
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    {"role": "user", "parts": [{"text": QUESTION_SYSTEM_PROMPT}]},
                    {"role": "model", "parts": [{"text": "I understand. I'll evaluate questions and respond in the specified format."}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
            )

            # Parse the response
            return self._parse_question_response(response.text, options)

        except ImportError:
            logger.warning("google-genai not installed, falling back to escalation")
            return BrainAnswer(
                selected=None,
                confidence=0.0,
                reasoning="Gemini brain not available - google-genai package not installed",
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return BrainAnswer(
                selected=None,
                confidence=0.0,
                reasoning=f"Gemini API error: {e}",
            )

    async def evaluate_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """Evaluate a medium-risk permission request using Gemini.

        Args:
            tool_name: Name of the tool (Write, Edit, Bash, etc.)
            tool_input: Input parameters for the tool
            context: Session and project context

        Returns:
            PermissionDecision indicating whether to approve
        """
        try:
            client = self._get_client()

            # Build the prompt
            input_summary = self._format_tool_input(tool_name, tool_input)

            user_prompt = f"""Project: {context.project_path}
Current Task: {context.current_task or 'Not specified'}

Tool: {tool_name}
Input: {input_summary}

Recent tools used: {', '.join(context.recent_tools[-5:]) if context.recent_tools else 'None'}
Recently modified files: {', '.join(context.modified_files[-5:]) if context.modified_files else 'None'}

Please evaluate whether to approve this tool use."""

            # Make the API call
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    {"role": "user", "parts": [{"text": PERMISSION_SYSTEM_PROMPT}]},
                    {"role": "model", "parts": [{"text": "I understand. I'll evaluate permissions and respond in the specified format."}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
            )

            # Parse the response
            return self._parse_permission_response(response.text)

        except ImportError:
            logger.warning("google-genai not installed, falling back to escalation")
            return PermissionDecision(
                approve=False,
                confidence=0.0,
                reason="Gemini brain not available - google-genai package not installed",
                requires_human=True,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return PermissionDecision(
                approve=False,
                confidence=0.0,
                reason=f"Gemini API error: {e}",
                requires_human=True,
            )

    def _parse_question_response(
        self, response_text: str, options: list[dict[str, str]]
    ) -> BrainAnswer:
        """Parse Gemini's response to a question."""
        lines = response_text.strip().split("\n")

        answer = None
        confidence = 0.5
        reasoning = "Unable to parse response"

        for line in lines:
            line = line.strip()
            if line.startswith("ANSWER:"):
                answer = line[7:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

        # Handle escalation
        if answer and answer.upper() == "ESCALATE":
            return BrainAnswer(
                selected=None,
                confidence=0.0,
                reasoning=reasoning,
            )

        # Try to match answer to options
        if answer and options:
            # Check for exact match
            for opt in options:
                if opt.get("label", "").lower() == answer.lower():
                    return BrainAnswer(
                        selected=opt.get("label"),
                        confidence=confidence,
                        reasoning=reasoning,
                    )

            # Check for partial match
            for opt in options:
                if answer.lower() in opt.get("label", "").lower():
                    return BrainAnswer(
                        selected=opt.get("label"),
                        confidence=confidence * 0.9,  # Slightly lower for partial match
                        reasoning=reasoning,
                    )

        return BrainAnswer(
            selected=answer,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _parse_permission_response(self, response_text: str) -> PermissionDecision:
        """Parse Gemini's response to a permission request."""
        lines = response_text.strip().split("\n")

        decision = "ESCALATE"
        confidence = 0.5
        reasoning = "Unable to parse response"

        for line in lines:
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line[9:].strip().upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

        if decision == "APPROVE":
            return PermissionDecision(
                approve=True,
                confidence=confidence,
                reason=reasoning,
                requires_human=False,
            )
        elif decision == "DENY":
            return PermissionDecision(
                approve=False,
                confidence=confidence,
                reason=reasoning,
                requires_human=False,
            )
        else:  # ESCALATE or unknown
            return PermissionDecision(
                approve=False,
                confidence=0.0,
                reason=reasoning,
                requires_human=True,
            )

    def _format_tool_input(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Format tool input for the prompt."""
        match tool_name:
            case "Bash":
                return f"command: {tool_input.get('command', '')}"
            case "Write":
                content = tool_input.get("content", "")
                lines = len(content.splitlines())
                return f"file: {tool_input.get('file_path', '')} ({lines} lines)"
            case "Edit":
                return (
                    f"file: {tool_input.get('file_path', '')}, "
                    f"old: '{tool_input.get('old_string', '')[:50]}...', "
                    f"new: '{tool_input.get('new_string', '')[:50]}...'"
                )
            case "Read":
                return f"file: {tool_input.get('file_path', '')}"
            case "Task":
                return f"prompt: {tool_input.get('prompt', '')[:100]}..."
            case _:
                # Generic formatting
                return str(tool_input)[:200]

    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold for auto-answering."""
        return self._confidence_threshold
