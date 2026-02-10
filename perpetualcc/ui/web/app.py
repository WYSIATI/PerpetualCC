"""FastAPI Web UI for PerpetualCC."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from perpetualcc.ui.web.config import router as config_router
from perpetualcc.ui.web.analytics import analytics
from perpetualcc.ui.web.export_manager import exporter

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
async def start_session(session_id: str, config: dict = None) -> dict[str, str]:
    """Start a session."""
    sessions[session_id] = {
        "id": session_id,
        "status": "running",
        "logs": [],
        "config": config or {},
    }
    # Track analytics
    analytics.record_session_start(session_id, config or {})
    return {"status": "started", "session_id": session_id}


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str) -> dict[str, str]:
    """Stop a session."""
    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"
        # Track analytics
        analytics.record_session_end(session_id, "stopped")
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


# ===== PHASE 3: Analytics & Export Endpoints =====

@app.get("/api/analytics/sessions/{session_id}")
async def get_session_analytics(session_id: str) -> dict[str, Any]:
    """Get analytics for a specific session."""
    return analytics.get_session_stats(session_id)


@app.get("/api/analytics/overall")
async def get_overall_analytics() -> dict[str, Any]:
    """Get overall analytics."""
    return analytics.get_overall_stats()


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request) -> HTMLResponse:
    """Analytics dashboard page."""
    return templates.TemplateResponse(
        "analytics.html",
        {"request": request, "stats": analytics.get_overall_stats()}
    )


@app.post("/api/sessions/{session_id}/export/json")
async def export_session_json(session_id: str) -> dict[str, str]:
    """Export session as JSON."""
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    filepath = exporter.export_session_json(session_id, sessions[session_id])
    return {"status": "exported", "filepath": filepath}


@app.post("/api/sessions/{session_id}/export/csv")
async def export_session_csv(session_id: str) -> dict[str, str]:
    """Export session logs as CSV."""
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    logs = sessions[session_id].get("logs", [])
    filepath = exporter.export_session_csv(session_id, logs)
    return {"status": "exported", "filepath": filepath}


@app.get("/api/exports")
async def list_exports() -> dict[str, Any]:
    """List all exported files."""
    return {"exports": exporter.list_exports()}


@app.get("/api/theme/{theme_name}")
async def set_theme(theme_name: str) -> dict[str, str]:
    """Set UI theme (light/dark)."""
    # Theme is handled client-side, this endpoint is for tracking
    return {"theme": theme_name, "status": "set"}


def run_web_ui(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the web UI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_web_ui()
