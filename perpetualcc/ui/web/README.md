# Web UI for PerpetualCC

A browser-based dashboard for managing PerpetualCC sessions.

## Features

- 📊 **Visual Session Management** - See all sessions at a glance
- 🎮 **One-Click Controls** - Start/stop sessions with buttons
- 📜 **Real-Time Logs** - Live log streaming via WebSocket
- 🎨 **Dark Theme** - Easy on the eyes for long coding sessions

## Installation

```bash
# Install with web UI support
pip install perpetualcc[web]

# Or install all features including web
pip install perpetualcc[all]
```

## Usage

```bash
# Start web UI on default port 8080
pcc web

# Custom port
pcc web --port 3000

# Custom host
pcc web --host 127.0.0.1 --port 5000
```

Then open http://localhost:8080 in your browser.

## Screenshots

*Dashboard showing multiple sessions with status and logs*

## Architecture

```
perpetualcc/ui/web/
├── app.py           # FastAPI application
├── static/          # CSS, JS assets
│   └── style.css
└── templates/       # HTML templates
    └── index.html
```

## Development

```bash
# Run in development mode with auto-reload
uvicorn perpetualcc.ui.web.app:app --reload --port 8080
```

## Future Enhancements

- [ ] Session configuration editor
- [ ] Brain selection UI
- [ ] Permission rule editor
- [ ] Session analytics dashboard
- [ ] Mobile-responsive design

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTMX + Jinja2 Templates
- **Styling:** Custom CSS (dark theme)
- **Real-time:** WebSocket

---

Part of [PerpetualCC](https://github.com/WYSIATI/PerpetualCC)
