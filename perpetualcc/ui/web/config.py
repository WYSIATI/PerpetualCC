"""Session configuration editor for Web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["configuration"])

templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_path)

# Valid configuration values
VALID_BRAIN_TYPES = frozenset({"rule_based", "llm", "hybrid", "custom"})
VALID_RISK_THRESHOLDS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
MAX_ITERATIONS_MIN = 1
MAX_ITERATIONS_MAX = 1000


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def validate_brain(brain: str) -> str:
    """Validate brain type.

    Args:
        brain: The brain type to validate.

    Returns:
        Validated brain type.

    Raises:
        ConfigValidationError: If brain type is invalid.
    """
    brain = brain.strip().lower()
    if brain not in VALID_BRAIN_TYPES:
        raise ConfigValidationError(
            "brain",
            f"Invalid brain type. Must be one of: {', '.join(sorted(VALID_BRAIN_TYPES))}"
        )
    return brain


def validate_risk_threshold(threshold: str) -> str:
    """Validate risk threshold.

    Args:
        threshold: The risk threshold to validate.

    Returns:
        Validated risk threshold.

    Raises:
        ConfigValidationError: If risk threshold is invalid.
    """
    threshold = threshold.strip().upper()
    if threshold not in VALID_RISK_THRESHOLDS:
        raise ConfigValidationError(
            "risk_threshold",
            f"Invalid risk threshold. Must be one of: {', '.join(sorted(VALID_RISK_THRESHOLDS))}"
        )
    return threshold


def validate_max_iterations(max_iterations: int) -> int:
    """Validate max iterations.

    Args:
        max_iterations: The max iterations value to validate.

    Returns:
        Validated max iterations.

    Raises:
        ConfigValidationError: If max iterations is invalid.
    """
    if max_iterations < MAX_ITERATIONS_MIN:
        raise ConfigValidationError(
            "max_iterations",
            f"Max iterations must be at least {MAX_ITERATIONS_MIN}"
        )
    if max_iterations > MAX_ITERATIONS_MAX:
        raise ConfigValidationError(
            "max_iterations",
            f"Max iterations must be at most {MAX_ITERATIONS_MAX}"
        )
    return max_iterations


@router.get("/", response_class=HTMLResponse)
async def config_editor(request: Request) -> HTMLResponse:
    """Session configuration editor page."""
    logger.debug("Rendering configuration editor")
    return templates.TemplateResponse(
        request,
        "config.html",
        {
            "config": {
                "brain": "rule_based",
                "auto_approve": False,
                "risk_threshold": "MEDIUM",
                "max_iterations": 50,
            },
            "valid_brain_types": sorted(VALID_BRAIN_TYPES),
            "valid_risk_thresholds": sorted(VALID_RISK_THRESHOLDS),
            "max_iterations_range": {
                "min": MAX_ITERATIONS_MIN,
                "max": MAX_ITERATIONS_MAX,
            },
        },
    )


@router.post("/save", response_class=HTMLResponse)
async def save_config(
    request: Request,
    brain: str = Form("rule_based"),
    auto_approve: bool = Form(False),
    risk_threshold: str = Form("MEDIUM"),
    max_iterations: int = Form(50),
) -> HTMLResponse:
    """Save session configuration.

    Returns HTML response for HTMX integration.
    """
    errors: list[str] = []
    validated_config: dict[str, Any] = {}

    # Validate brain type
    try:
        validated_config["brain"] = validate_brain(brain)
    except ConfigValidationError as e:
        errors.append(e.message)
        logger.warning(f"Config validation failed: {e}")

    # Validate auto_approve (boolean, no special validation needed)
    validated_config["auto_approve"] = bool(auto_approve)

    # Validate risk threshold
    try:
        validated_config["risk_threshold"] = validate_risk_threshold(risk_threshold)
    except ConfigValidationError as e:
        errors.append(e.message)
        logger.warning(f"Config validation failed: {e}")

    # Validate max iterations
    try:
        validated_config["max_iterations"] = validate_max_iterations(max_iterations)
    except ConfigValidationError as e:
        errors.append(e.message)
        logger.warning(f"Config validation failed: {e}")

    if errors:
        # Return error response for HTMX
        logger.warning(f"Configuration save failed with {len(errors)} validation errors")
        error_html = f"""
        <div class="alert alert-error" role="alert">
            <strong>Validation Error</strong>
            <ul>
                {"".join(f"<li>{error}</li>" for error in errors)}
            </ul>
        </div>
        """
        return HTMLResponse(
            content=error_html,
            status_code=status.HTTP_400_BAD_REQUEST
        )

    # TODO: Actually save to database/file
    logger.info(f"Configuration saved: {validated_config}")

    # Return success response for HTMX
    success_html = """
    <div class="alert alert-success" role="alert">
        <strong>Success!</strong> Configuration saved successfully.
    </div>
    """
    return HTMLResponse(content=success_html, status_code=status.HTTP_200_OK)


@router.post("/save/json")
async def save_config_json(
    brain: str = Form("rule_based"),
    auto_approve: bool = Form(False),
    risk_threshold: str = Form("MEDIUM"),
    max_iterations: int = Form(50),
) -> dict[str, Any]:
    """Save session configuration (JSON response).

    Returns JSON response for API clients.
    """
    errors: dict[str, str] = {}
    validated_config: dict[str, Any] = {}

    # Validate brain type
    try:
        validated_config["brain"] = validate_brain(brain)
    except ConfigValidationError as e:
        errors[e.field] = e.message

    # Validate auto_approve (boolean, no special validation needed)
    validated_config["auto_approve"] = bool(auto_approve)

    # Validate risk threshold
    try:
        validated_config["risk_threshold"] = validate_risk_threshold(risk_threshold)
    except ConfigValidationError as e:
        errors[e.field] = e.message

    # Validate max iterations
    try:
        validated_config["max_iterations"] = validate_max_iterations(max_iterations)
    except ConfigValidationError as e:
        errors[e.field] = e.message

    if errors:
        logger.warning(f"Configuration save failed with validation errors: {errors}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Validation failed", "errors": errors}
        )

    # TODO: Actually save to database/file
    logger.info(f"Configuration saved (JSON): {validated_config}")

    return {"status": "saved", "config": validated_config}


@router.get("/validate")
async def validate_config_values(
    brain: str | None = None,
    risk_threshold: str | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Validate configuration values without saving.

    Useful for real-time validation in the UI.
    """
    result: dict[str, Any] = {"valid": True, "errors": {}}

    if brain is not None:
        try:
            validate_brain(brain)
        except ConfigValidationError as e:
            result["valid"] = False
            result["errors"]["brain"] = e.message

    if risk_threshold is not None:
        try:
            validate_risk_threshold(risk_threshold)
        except ConfigValidationError as e:
            result["valid"] = False
            result["errors"]["risk_threshold"] = e.message

    if max_iterations is not None:
        try:
            validate_max_iterations(max_iterations)
        except ConfigValidationError as e:
            result["valid"] = False
            result["errors"]["max_iterations"] = e.message

    return result


@router.get("/defaults")
async def get_default_config() -> dict[str, Any]:
    """Get default configuration values."""
    return {
        "config": {
            "brain": "rule_based",
            "auto_approve": False,
            "risk_threshold": "MEDIUM",
            "max_iterations": 50,
        },
        "valid_brain_types": sorted(VALID_BRAIN_TYPES),
        "valid_risk_thresholds": sorted(VALID_RISK_THRESHOLDS),
        "max_iterations_range": {
            "min": MAX_ITERATIONS_MIN,
            "max": MAX_ITERATIONS_MAX,
        },
    }
