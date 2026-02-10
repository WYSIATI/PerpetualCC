"""Session configuration editor for Web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/config", tags=["configuration"])

templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=templates_path)


@router.get("/", response_class=HTMLResponse)
async def config_editor(request: Request) -> HTMLResponse:
    """Session configuration editor page."""
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "config": {
                "brain": "rule_based",
                "auto_approve": False,
                "risk_threshold": "MEDIUM",
                "max_iterations": 50,
            }
        }
    )


@router.post("/save")
async def save_config(
    brain: str = Form("rule_based"),
    auto_approve: bool = Form(False),
    risk_threshold: str = Form("MEDIUM"),
    max_iterations: int = Form(50),
) -> dict[str, Any]:
    """Save session configuration."""
    config = {
        "brain": brain,
        "auto_approve": auto_approve,
        "risk_threshold": risk_threshold,
        "max_iterations": max_iterations,
    }
    # TODO: Save to database
    return {"status": "saved", "config": config}
