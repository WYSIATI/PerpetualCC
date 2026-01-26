# PerpetualCC

**An intelligent master agent that orchestrates Claude Code sessions 24/7**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PerpetualCC acts like an experienced human developer - understanding project context, making informed decisions about tool permissions, answering Claude Code's questions automatically, and knowing when to escalate to a real human.

---

## Why PerpetualCC?

### 🔄 Never Miss a Token Reset

When Claude Code hits its token limit, PerpetualCC **automatically detects the reset time and resumes your session** the moment tokens are available again. No more forgetting to come back, no more lost momentum.

### 🧠 Intelligent Question Answering

Claude Code often asks questions like "Should I proceed?" or "Which approach do you prefer?" PerpetualCC **automatically answers these questions** using:
- **Rule-based patterns** - No AI needed, handles common questions instantly
- **Gemini API** - Context-aware responses for complex decisions
- **Local LLM (Ollama)** - Privacy-focused, works offline with DeepSeek or other models

You stay focused on high-level decisions while PerpetualCC handles the routine interactions.

---

## Features

- **Autonomous Operation** - Runs Claude Code sessions unattended, handling permissions and questions automatically
- **Auto-Resume on Token Reset** - Detects rate limits, saves session state, and automatically resumes when tokens are available
- **Smart Question Answering** - Multiple brain options (rule-based, Gemini, Ollama) to answer Claude Code's questions
- **Risk-Based Permissions** - Auto-approves safe operations, asks brain for medium-risk, escalates dangerous ones to human
- **Multi-Session Management** - Run and monitor multiple concurrent sessions
- **Human Bridge** - macOS notifications when human input is needed
- **Project Understanding** - RAG-powered codebase knowledge (optional)
- **Learning System** - Improves decisions over time (optional)

---

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/perpetualcc.git
cd perpetualcc

# Create virtual environment with uv (recommended)
uv venv .venv
source .venv/bin/activate

# Install with development dependencies
uv pip install -e ".[dev]"

# Or install with all optional features
uv pip install -e ".[all,dev]"
```

### Optional Dependencies

| Install Command | Features Added |
|-----------------|----------------|
| `uv pip install -e ".[gemini]"` | Gemini API brain |
| `uv pip install -e ".[ollama]"` | Local LLM brain via Ollama |
| `uv pip install -e ".[knowledge]"` | RAG + code understanding |
| `uv pip install -e ".[memory]"` | SQLite learning persistence |
| `uv pip install -e ".[all]"` | All optional features |

---

## Quick Start

### 1. Start a Session

```bash
# Simple task
pcc start ./your-project --task "Build a REST API with authentication"

# With requirements file
pcc start ./your-project --requirements requirements.md

# Run in background
pcc start ./your-project --task "Refactor the auth module" --detach
```

### 2. Monitor Sessions

```bash
# List all sessions
pcc list

# Check detailed status
pcc status <session-id>

# Attach to live session
pcc attach <session-id>
```

### 3. Handle Escalations

```bash
# View pending decisions
pcc pending

# Respond to a question
pcc respond 1 "Use PostgreSQL"

# Interactive response
pcc respond 1 --interactive
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         PerpetualCC                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Claude Code Session ──▶ Event Stream ──▶ Master Agent        │
│                                                 │                │
│                              ┌──────────────────┼──────────────┐│
│                              │                  │              ││
│                              ▼                  ▼              ▼│
│                        ┌──────────┐      ┌──────────┐    ┌─────┴│
│                        │ Decision │      │  Brain   │    │Human ││
│                        │ Engine   │      │ (answer) │    │Bridge││
│                        └────┬─────┘      └────┬─────┘    └──────┘│
│                             │                 │                  │
│                    LOW risk: Auto-approve     │                  │
│                    MED risk: Use brain ───────┘                  │
│                    HIGH risk: Escalate to human ─────────────────│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Engine

| Risk Level | Examples | Action |
|------------|----------|--------|
| **LOW** | Read any file, Glob/Grep, Write to src/tests/ | Auto-approve |
| **MEDIUM** | Git commands, config files, network requests | Ask brain |
| **HIGH** | rm -rf, sudo, .env files, --force flags | Escalate to human |

### Brain Options

| Type | Setup | Best For |
|------|-------|----------|
| **Rule-Based** | Default, no setup | Simple projects, predictable workflows |
| **Gemini** | Set `GEMINI_API_KEY` | Context-aware decisions, complex questions |
| **Ollama** | Run `ollama serve` locally | Privacy-focused, offline operation |

---

## Configuration

Configuration is stored at `~/.perpetualcc/config.toml`. Key settings:

```bash
# Configure brain
pcc config brain gemini    # Use Gemini API
pcc config brain ollama    # Use local Ollama
pcc config brain none      # Rule-based only

# View configuration
pcc config show
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for Claude Code sessions |
| `GEMINI_API_KEY` | Optional, for Gemini brain |

---

## Requirements File Format

Create a `requirements.md` to give PerpetualCC context:

```markdown
# Project: My API

## Overview
REST API for user management.

## Preferences
- Database: PostgreSQL
- ORM: Prisma
- Auth: JWT with refresh tokens
- Test framework: Vitest

## Tasks
- [ ] Set up project structure
- [ ] Create user registration
- [ ] Implement authentication
- [ ] Write tests
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pcc start <path>` | Start a new session |
| `pcc list` | List all sessions |
| `pcc status <id>` | Show session details |
| `pcc attach <id>` | Attach to live session |
| `pcc pause <id>` | Pause a session |
| `pcc resume <id>` | Resume a session |
| `pcc add <id> "task"` | Add task to queue |
| `pcc pending` | List pending human decisions |
| `pcc respond <n>` | Respond to escalation |
| `pcc config` | Configure settings |
| `pcc logs <id>` | View session logs |

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=perpetualcc

# Run specific test file
pytest tests/unit/test_risk_classifier.py
```

### Code Quality

```bash
# Lint
ruff check perpetualcc/

# Format
ruff format perpetualcc/
```

### Project Structure

```
perpetualcc/
├── perpetualcc/
│   ├── cli.py              # CLI commands (typer)
│   ├── claude/             # Claude Agent SDK integration
│   ├── core/               # Decision engine, session management
│   ├── brain/              # Question answering (rule-based, Gemini, Ollama)
│   ├── knowledge/          # RAG + code understanding (optional)
│   ├── memory/             # Learning systems (optional)
│   └── human/              # Notifications + escalation
├── tests/
│   ├── unit/
│   └── integration/
├── ARCHITECTURE.md         # Detailed design document
└── CLAUDE.md               # Context for Claude Code sessions
```

---

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | CLI + Claude Adapter | Done |
| 2 | Decision Engine | Done |
| 3 | Brain (Rule-Based) | Done |
| 4 | Rate Limit + Checkpoint | Done |
| 5 | Session Manager | Done |
| 6 | Master Agent | Done |
| 7 | Knowledge Engine | Planned |
| 8 | Memory Systems | Planned |
| 9 | Human Bridge | Planned |

---

## Contributing

Contributions are welcome! Please read the [Architecture Document](ARCHITECTURE.md) to understand the system design.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'feat: add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
