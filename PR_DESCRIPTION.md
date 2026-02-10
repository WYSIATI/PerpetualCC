# 🚀 PerpetualCC Web UI - Pull Request

## Summary
Adds a modern web-based dashboard for managing PerpetualCC sessions, making it easier to monitor and control agent operations visually.

## Features

### Phase 1 - Core Dashboard ✅
- **Session Management**: Visual cards showing all active sessions
- **Real-time Controls**: Start/stop buttons with HTMX
- **Live Logs**: WebSocket streaming for real-time output
- **Dark Theme**: GitHub-inspired UI design

### Phase 2 - Configuration ✅  
- **Session Config Editor**: Brain selection, risk thresholds
- **Settings UI**: Form-based configuration
- **Validation**: Input validation ready

### Phase 3 - Coming Soon ⏳
- Permission rule editor
- Analytics dashboard
- Session analytics

## Usage

```bash
# Install with web support
pip install perpetualcc[web]

# Launch dashboard
pcc web              # Default port 8080
pcc web --port 3000  # Custom port
```

## Screenshots

*Dashboard showing session overview with controls*

## Technical Details

### Architecture
- **Backend**: FastAPI with async support
- **Frontend**: HTMX + Jinja2 (no React needed)
- **Real-time**: WebSocket for log streaming
- **Styling**: Custom CSS (dark theme)

### Files Added
```
perpetualcc/ui/web/
├── app.py              # FastAPI application
├── config.py           # Configuration routes
├── static/
│   └── style.css      # Styling
└── templates/
    ├── index.html     # Dashboard
    └── config.html    # Settings page
```

## Testing

```bash
# Run locally
cd perpetualcc/ui/web
python -m uvicorn app:app --reload

# Or use CLI
pcc web
```

## Dependencies

Added to `pyproject.toml`:
- fastapi>=0.110.0
- uvicorn>=0.27.0
- jinja2>=3.1.0
- python-multipart>=0.0.9

## Checklist

- [x] Code follows project style
- [x] Type hints included
- [x] Documentation added
- [x] CLI integration complete
- [x] Tested locally
- [ ] Review feedback addressed
- [ ] Tests added (if needed)

## Future Work

- Mobile responsiveness improvements
- User authentication
- Session persistence
- Plugin system for UI extensions

---

Ready for review! 🦞
