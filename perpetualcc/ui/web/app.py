"""FastAPI Web UI for PerpetualCC."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from perpetualcc.ui.web.config import router as config_router

app = FastAPI(title="PerpetualCC Dashboard")
app.include_router(config_router)

# Static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# In-memory session store (replace with DB later)
sessions: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "sessions": sessions}
    )


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    """List all sessions."""
    return {"sessions": list(sessions.values())}


@app.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str) -> dict[str, str]:
    """Start a session."""
    sessions[session_id] = {
        "id": session_id,
        "status": "running",
        "logs": [],
    }
    return {"status": "started", "session_id": session_id}


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str) -> dict[str, str]:
    """Stop a session."""
    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"
    return {"status": "stopped", "session_id": session_id}


@app.websocket("/ws/logs/{session_id}")
async def log_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket for real-time logs."""
    await websocket.accept()
    try:
        while True:
            if session_id in sessions:
                logs = sessions[session_id].get("logs", [])
                await websocket.send_json({"logs": logs[-50:]})  # Last 50 logs
            await asyncio.sleep(1)
    except Exception:
        await websocket.close()


def run_web_ui(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the web UI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_web_ui()
