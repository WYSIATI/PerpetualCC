"""FastAPI Web UI for PerpetualCC."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from perpetualcc.ui.web.config import router as config_router
from perpetualcc.ui.web.analytics import analytics
from perpetualcc.ui.web.export_manager import exporter

# Configure logging
logger = logging.getLogger(__name__)

# Session ID validation pattern: alphanumeric, hyphens, underscores only
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
SESSION_ID_MAX_LENGTH = 128

# Session cleanup configuration
SESSION_MAX_AGE_SECONDS = 3600 * 24  # 24 hours
SESSION_CLEANUP_INTERVAL = 3600  # 1 hour

# In-memory session store (replace with DB later)
sessions: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("PerpetualCC Web UI starting up")
    yield
    # Shutdown
    logger.info("PerpetualCC Web UI shutting down")
    sessions.clear()


app = FastAPI(title="PerpetualCC Dashboard", lifespan=lifespan)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config_router)

# Static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# Track last cleanup time
_last_cleanup_time: float = time.time()


def validate_session_id(session_id: str) -> None:
    """Validate session ID format.

    Raises:
        HTTPException: If session_id is invalid.
    """
    if not session_id:
        logger.warning("Empty session_id provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required"
        )

    if len(session_id) > SESSION_ID_MAX_LENGTH:
        logger.warning(f"Session ID too long: {len(session_id)} characters")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session ID must be {SESSION_ID_MAX_LENGTH} characters or less"
        )

    if not SESSION_ID_PATTERN.match(session_id):
        logger.warning(f"Invalid session_id format: {session_id[:50]}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID must contain only alphanumeric characters, hyphens, and underscores"
        )


def cleanup_old_sessions() -> int:
    """Remove sessions older than SESSION_MAX_AGE_SECONDS.

    Returns:
        Number of sessions cleaned up.
    """
    global _last_cleanup_time
    current_time = time.time()

    # Only run cleanup periodically
    if current_time - _last_cleanup_time < SESSION_CLEANUP_INTERVAL:
        return 0

    _last_cleanup_time = current_time
    cleaned = 0

    sessions_to_remove = []
    for session_id, session_data in sessions.items():
        created_at = session_data.get("created_at", 0)
        if current_time - created_at > SESSION_MAX_AGE_SECONDS:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        del sessions[session_id]
        cleaned += 1
        logger.info(f"Cleaned up old session: {session_id}")

    if cleaned > 0:
        logger.info(f"Session cleanup complete: removed {cleaned} sessions")

    return cleaned


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page."""
    logger.debug("Rendering dashboard")
    cleanup_old_sessions()
    return templates.TemplateResponse(
        request, "index.html", {"sessions": sessions}
    )


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    """List all sessions."""
    logger.debug(f"Listing {len(sessions)} sessions")
    cleanup_old_sessions()
    return {"sessions": list(sessions.values())}


@app.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str, config: dict | None = None) -> dict[str, str]:
    """Start a session."""
    validate_session_id(session_id)

    if session_id in sessions:
        logger.warning(f"Attempted to start existing session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session '{session_id}' already exists"
        )

    logger.info(f"Starting session: {session_id}")
    sessions[session_id] = {
        "id": session_id,
        "status": "running",
        "logs": [],
        "config": config or {},
        "created_at": time.time(),
    }

    # Track analytics
    analytics.record_session_start(session_id, config or {})

    cleanup_old_sessions()
    return {"status": "started", "session_id": session_id}


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str) -> dict[str, str]:
    """Stop a session."""
    validate_session_id(session_id)

    if session_id not in sessions:
        logger.warning(f"Attempted to stop non-existent session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    logger.info(f"Stopping session: {session_id}")
    sessions[session_id]["status"] = "stopped"

    # Track analytics
    analytics.record_session_end(session_id, "stopped")

    return {"status": "stopped", "session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session."""
    validate_session_id(session_id)

    if session_id not in sessions:
        logger.warning(f"Attempted to delete non-existent session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    logger.info(f"Deleting session: {session_id}")
    del sessions[session_id]

    return {"status": "deleted", "session_id": session_id}


@app.websocket("/ws/logs/{session_id}")
async def log_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket for real-time logs."""
    # Validate session_id format (can't use HTTPException in WebSocket)
    if not session_id or not SESSION_ID_PATTERN.match(session_id):
        logger.warning(f"Invalid session_id in WebSocket connection: {session_id[:50] if session_id else 'empty'}")
        await websocket.close(code=4000, reason="Invalid session ID format")
        return

    if len(session_id) > SESSION_ID_MAX_LENGTH:
        logger.warning(f"Session ID too long in WebSocket: {len(session_id)}")
        await websocket.close(code=4000, reason="Session ID too long")
        return

    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")

    try:
        while True:
            if session_id in sessions:
                logs = sessions[session_id].get("logs", [])
                await websocket.send_json({"logs": logs[-50:]})  # Last 50 logs
            else:
                # Session doesn't exist, send empty logs but keep connection
                await websocket.send_json({"logs": [], "warning": "Session not found"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except asyncio.CancelledError:
        logger.info(f"WebSocket cancelled for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {type(e).__name__}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # Connection may already be closed


# ===== PHASE 3: Analytics & Export Endpoints =====

@app.get("/api/analytics/sessions/{session_id}")
async def get_session_analytics(session_id: str) -> dict[str, Any]:
    """Get analytics for a specific session."""
    validate_session_id(session_id)

    stats = analytics.get_session_stats(session_id)
    if not stats:
        logger.warning(f"Analytics not found for session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analytics for session '{session_id}' not found"
        )

    logger.debug(f"Returning analytics for session: {session_id}")
    return stats


@app.get("/api/analytics/overall")
async def get_overall_analytics() -> dict[str, Any]:
    """Get overall analytics."""
    logger.debug("Returning overall analytics")
    return analytics.get_overall_stats()


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request) -> HTMLResponse:
    """Analytics dashboard page."""
    logger.debug("Rendering analytics dashboard")
    return templates.TemplateResponse(
        request, "analytics.html", {"stats": analytics.get_overall_stats()}
    )


@app.post("/api/sessions/{session_id}/export/json")
async def export_session_json(session_id: str) -> dict[str, str]:
    """Export session as JSON."""
    validate_session_id(session_id)

    if session_id not in sessions:
        logger.warning(f"Export failed - session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    try:
        logger.info(f"Exporting session as JSON: {session_id}")
        filepath = exporter.export_session_json(session_id, sessions[session_id])
        return {"status": "exported", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to export session {session_id} as JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export session"
        )


@app.post("/api/sessions/{session_id}/export/csv")
async def export_session_csv(session_id: str) -> dict[str, str]:
    """Export session logs as CSV."""
    validate_session_id(session_id)

    if session_id not in sessions:
        logger.warning(f"Export failed - session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    try:
        logger.info(f"Exporting session logs as CSV: {session_id}")
        logs = sessions[session_id].get("logs", [])
        filepath = exporter.export_session_csv(session_id, logs)
        return {"status": "exported", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to export session {session_id} as CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export session logs"
        )


@app.get("/api/exports")
async def list_exports() -> dict[str, Any]:
    """List all exported files."""
    try:
        logger.debug("Listing exports")
        return {"exports": exporter.list_exports()}
    except Exception as e:
        logger.error(f"Failed to list exports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list exports"
        )


@app.delete("/api/exports/cleanup")
async def cleanup_exports(max_age_hours: int = 24) -> dict[str, Any]:
    """Cleanup old export files."""
    if max_age_hours < 1 or max_age_hours > 8760:  # 1 hour to 1 year
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_age_hours must be between 1 and 8760"
        )

    try:
        logger.info(f"Cleaning up exports older than {max_age_hours} hours")
        cleaned = exporter.cleanup_old_exports(max_age_hours=max_age_hours)
        return {"status": "cleaned", "files_removed": cleaned}
    except Exception as e:
        logger.error(f"Failed to cleanup exports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup exports"
        )


@app.get("/api/theme/{theme_name}")
async def set_theme(theme_name: str) -> dict[str, str]:
    """Set UI theme (light/dark)."""
    valid_themes = {"light", "dark", "auto"}
    if theme_name not in valid_themes:
        logger.warning(f"Invalid theme requested: {theme_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid theme. Must be one of: {', '.join(valid_themes)}"
        )

    logger.debug(f"Theme set to: {theme_name}")
    return {"theme": theme_name, "status": "set"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def run_web_ui(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the web UI server."""
    import uvicorn

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info(f"Starting PerpetualCC Web UI on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_web_ui()
