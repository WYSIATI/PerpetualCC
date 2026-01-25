# PerpetualCC Development Guide

This file provides context for Claude Code sessions working on PerpetualCC.

## Project Overview

PerpetualCC is an intelligent master agent that orchestrates Claude Code sessions 24/7. It acts like an experienced human developer - understanding project context, making informed decisions about tool permissions, answering Claude Code's questions, and knowing when to escalate to a real human.

## Quick Start (Development)

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
uv pip install -e ".[dev,all]"

# Run CLI
pcc --help
pcc start . --task "Your task here"

# Run tests
pytest

# Lint
ruff check perpetualcc/
ruff format perpetualcc/
```

## Dependency Management with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Managing Dependencies

```bash
# Always activate venv first
source .venv/bin/activate

# Install base package (editable mode)
uv pip install -e .

# Install with development tools (pytest, ruff, etc.)
uv pip install -e ".[dev]"

# Install with specific brain backends
uv pip install -e ".[gemini]"      # Google Gemini API
uv pip install -e ".[ollama]"      # Local LLM via Ollama
uv pip install -e ".[knowledge]"   # RAG/code understanding
uv pip install -e ".[memory]"      # SQLite persistence

# Install everything
uv pip install -e ".[all,dev]"

# Add a new dependency
uv pip install <package>

# Check installed packages
uv pip list

# Sync dependencies from pyproject.toml
uv pip install -e ".[dev,all]" --reinstall
```

### Optional Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `gemini` | google-genai | Gemini API brain backend |
| `ollama` | ollama | Local LLM brain backend |
| `knowledge` | chromadb, sentence-transformers, networkx, tree-sitter | RAG + code graph |
| `memory` | aiosqlite | Async SQLite for memory systems |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff | Development tools |
| `all` | gemini + ollama + knowledge + memory | All optional features |

### Verify Installation

```bash
source .venv/bin/activate
python3 -c "
import perpetualcc
print(f'perpetualcc: {perpetualcc.__version__}')

# Check optional dependencies
try:
    import google.genai
    print('✓ google-genai installed')
except ImportError:
    print('✗ google-genai not installed')

try:
    import ollama
    print('✓ ollama installed')
except ImportError:
    print('✗ ollama not installed')
"
```

## Project Structure

```
perpetualcc/
├── pyproject.toml              # Project config (hatchling build)
├── ARCHITECTURE.md             # Comprehensive design document
├── CLAUDE.md                   # This file - context for Claude
├── README.md                   # User-facing documentation
│
├── perpetualcc/
│   ├── __init__.py             # Version: "0.1.0"
│   ├── __main__.py             # Entry: python -m perpetualcc
│   ├── cli.py                  # Typer CLI (pcc command)
│   │
│   ├── claude/                 # Claude Agent SDK integration
│   │   ├── __init__.py
│   │   ├── types.py            # Event types (ClaudeEvent hierarchy)
│   │   └── adapter.py          # SDK wrapper (ClaudeCodeAdapter)
│   │
│   ├── core/                   # Core orchestration (Phases 2, 4-6)
│   │   ├── __init__.py
│   │   ├── decision_engine.py  # Permission decisions (Phase 2)
│   │   ├── risk_classifier.py  # Risk level classification (Phase 2)
│   │   ├── rate_limit.py       # Rate limit detection (Phase 4)
│   │   ├── checkpoint.py       # Session state save/restore (Phase 4)
│   │   ├── task_queue.py       # Task ordering (Phase 5)
│   │   ├── session_manager.py  # Multi-session management (Phase 5)
│   │   └── master_agent.py     # ReAct supervisor loop (Phase 6)
│   │
│   ├── brain/                  # Question answering (Phase 3)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract Brain interface
│   │   ├── rule_based.py       # Default no-AI brain
│   │   ├── gemini.py           # Gemini API brain
│   │   └── ollama.py           # Local LLM brain
│   │
│   ├── knowledge/              # RAG + Code understanding (Phase 7)
│   │   ├── __init__.py
│   │   ├── indexer.py          # Codebase scanner
│   │   ├── embeddings.py       # Hybrid embedding provider
│   │   ├── rag.py              # Retrieval pipeline
│   │   └── code_graph.py       # Code relationship graph
│   │
│   ├── memory/                 # Learning systems (Phase 8)
│   │   ├── __init__.py
│   │   ├── store.py            # SQLite persistence
│   │   ├── episodic.py         # Event history
│   │   ├── procedural.py       # Learned rules
│   │   └── semantic.py         # Project knowledge
│   │
│   └── human/                  # Human intervention (Phase 9)
│       ├── __init__.py
│       ├── escalation.py       # Escalation queue
│       ├── notifications.py    # macOS notifications
│       └── cli_prompt.py       # Interactive prompts
│
└── tests/
    ├── unit/
    └── integration/
```

## Implementation Status

| Phase | Component | Status | Files |
|-------|-----------|--------|-------|
| 1 | CLI + Claude Adapter | **DONE** | `cli.py`, `claude/adapter.py`, `claude/types.py` |
| 2 | Decision Engine | **DONE** | `core/decision_engine.py`, `core/risk_classifier.py`, `brain/base.py` |
| 3 | Brain (Rule-Based) | **DONE** | `brain/rule_based.py` |
| 4 | Rate Limit + Checkpoint | **DONE** | `core/rate_limit.py`, `core/checkpoint.py` |
| 5 | Session Manager | **DONE** | `core/session_manager.py`, `core/task_queue.py` |
| 6 | Master Agent | **DONE** | `core/master_agent.py` |
| 7 | Knowledge Engine | TODO | `knowledge/*.py` |
| 8 | Memory Systems | TODO | `memory/*.py` |
| 9 | Human Bridge | TODO | `human/*.py` |

## Key Dependencies

- **claude-agent-sdk** (>=0.1.0): The official Claude Agent SDK for Python
  - Package: `claude-agent-sdk` (NOT `claude-code-sdk` which is deprecated)
  - Docs: https://platform.claude.com/docs/en/agent-sdk/python
- **typer** (>=0.12.0): CLI framework
- **rich** (>=13.0.0): Terminal formatting
- **pydantic** (>=2.0.0): Data validation
- **anyio** (>=4.0.0): Async utilities

Optional dependencies (install with `uv pip install -e ".[all]"`):
- **google-genai**: Gemini API for brain
- **ollama**: Local LLM for brain
- **chromadb**: Vector store for RAG
- **sentence-transformers**: Embeddings
- **networkx**: Code graph
- **tree-sitter**: AST parsing

## Coding Standards

### Python Style
- Python 3.11+ required
- Use `from __future__ import annotations` for forward references
- Use dataclasses for data structures
- Use type hints everywhere
- Line length: 100 characters
- Format with `ruff format`
- Lint with `ruff check`

### Async Patterns
- Use `async/await` for I/O operations
- Use `AsyncIterator` for streaming
- Use `anyio` for async primitives (not raw asyncio)

### Import Order (ruff handles this)
1. Standard library
2. Third-party packages
3. Local imports

### Error Handling
- Use specific exception types
- Log errors before re-raising
- Graceful degradation where possible

## Claude Agent SDK Reference

### Starting a Session

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Your task here",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",  # or "default", "acceptEdits"
        cwd="/path/to/project",
        max_turns=10,  # optional
    )
):
    # Handle message
    pass
```

### Message Types

```python
from claude_agent_sdk.types import (
    AssistantMessage,    # Claude's response with content blocks
    ResultMessage,       # Final result with cost/usage
    SystemMessage,       # System events (init, etc.)
    TextBlock,           # Text content
    ThinkingBlock,       # Reasoning content
    ToolUseBlock,        # Tool invocation
    ToolResultBlock,     # Tool result
)
```

### Session Resumption

```python
# Resume existing session
options = ClaudeAgentOptions(resume="session-uuid-here")
async for message in query(prompt="Continue", options=options):
    ...
```

### Custom Permission Handling

```python
from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

async def can_use_tool(
    tool_name: str,
    input_data: dict,
    context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    if is_dangerous(tool_name, input_data):
        return PermissionResultDeny(message="Blocked")
    return PermissionResultAllow(updated_input=input_data)

options = ClaudeAgentOptions(can_use_tool=can_use_tool)
```

## Phase Implementation Guidelines

### Phase 2: Decision Engine (DONE)

**Goal**: Auto-approve safe operations, block dangerous ones.

**Implemented files**:
1. `perpetualcc/core/risk_classifier.py`
   - `RiskLevel` enum: LOW, MEDIUM, HIGH
   - `RiskClassification` dataclass with level, reason, matched_pattern
   - `RiskConfig` dataclass for customizable patterns
   - `RiskClassifier` class with `classify()` method
   - `classify_risk()` convenience function

2. `perpetualcc/core/decision_engine.py`
   - `PermissionDecision` dataclass: approve, confidence, reason, risk_level, requires_human
   - `DecisionRecord` for auditing/logging
   - `DecisionEngine` class with sync and async `decide_permission()` methods
   - `create_permission_callback()` for SDK integration

3. `perpetualcc/brain/base.py`
   - `Brain` ABC with `answer_question()` and `evaluate_permission()`
   - `QuestionContext`, `PermissionContext`, `BrainAnswer` dataclasses
   - Protocol for brain implementations

4. `perpetualcc/claude/adapter.py` updated to accept `permission_callback`

**Risk Classification Rules**:
- LOW: Read (any), Glob/Grep (any), Write/Edit (in src/tests/lib/app)
- LOW: Bash with safe commands (npm, yarn, pip, pytest, cargo)
- MEDIUM: Write/Edit to config files or outside safe dirs
- MEDIUM: Bash with git commands or network commands
- HIGH: Bash with rm/sudo/chmod, pipes, curl|sh, --force flags
- HIGH: Write to .env, credentials, system files

**Tests**: 77 unit tests covering all risk levels and decision scenarios.

### Phase 3: Brain (Rule-Based) (DONE)

**Goal**: Answer Claude Code's questions automatically.

**Implemented files**:
1. `perpetualcc/brain/rule_based.py`
   - `QuestionPattern` dataclass for matching questions
   - `PermissionPattern` dataclass for matching permission requests
   - `RuleBasedConfig` dataclass for configurable patterns
   - `RuleBasedBrain` class implementing `Brain` ABC
   - `default_question_patterns()` - 12 patterns for common questions
   - `default_permission_patterns()` - 17 patterns for permission requests

**Question Matching Rules**:
- "proceed/continue?" -> Yes (0.85 confidence)
- "Should I proceed/continue?" -> Yes (0.85 confidence)
- "Is this ok/good/correct?" -> Yes (0.75 confidence)
- "Ready to start/begin?" -> Yes (0.85 confidence)
- "Run tests/build/lint?" -> Yes (0.80 confidence)
- "Install dependencies?" -> Yes (0.75 confidence)
- "Create/add file/directory?" -> Yes (0.75 confidence)
- "Update/modify code/file?" -> Yes (0.75 confidence)
- "Commit changes?" -> Yes (0.70 confidence)
- "Push to remote?" -> Yes (0.65 confidence)
- "Which type/format/option?" -> Select first option (0.60 confidence)
- Unknown -> escalate (0.0 confidence)

**Permission Matching Rules** (for MEDIUM risk):
- Git read-only (status/log/diff/branch) -> approve (0.85)
- Git add/commit -> approve (0.75)
- Git fetch/pull -> approve (0.75)
- Git push (no force) -> approve (0.70)
- Git push --force -> deny (0.90)
- curl/wget -> approve cautiously (0.60-0.65)
- Docker build/run/ps/images/logs -> approve (0.70)
- Docker rm/rmi/prune/system -> deny (0.80)
- Config files (package.json, tsconfig, pyproject.toml) -> approve (0.70)
- Simple rm (not -rf) -> approve (0.65)
- mv -> approve (0.65)
- Task tool -> approve (0.75)
- Unknown -> escalate (0.0)

**Integration**: Brain is called by DecisionEngine for MEDIUM risk decisions.

**Tests**: 65 unit tests covering all patterns and edge cases.

### Phase 4: Rate Limit + Checkpoint

**Goal**: Detect rate limits, save state, auto-resume.

**Files to create**:
1. `perpetualcc/core/rate_limit.py`
   - `RateLimitInfo` dataclass
   - `RateLimitMonitor.detect(event) -> RateLimitInfo | None`
   - `RateLimitMonitor.wait_for_reset(info)` - async countdown

2. `perpetualcc/core/checkpoint.py`
   - `SessionCheckpoint` dataclass (all session state)
   - `CheckpointManager.save(session) -> checkpoint`
   - `CheckpointManager.restore(checkpoint) -> resume_prompt`
   - Storage in `~/.perpetualcc/data/checkpoints/`

### Phase 5: Session Manager

**Goal**: Manage multiple concurrent sessions.

**Files to create**:
1. `perpetualcc/core/task_queue.py`
   - Priority-ordered task queue
   - SQLite persistence

2. `perpetualcc/core/session_manager.py`
   - `ManagedSession` dataclass
   - `SessionManager` class managing multiple sessions
   - State persistence to SQLite

**CLI extensions**: Add `pcc list`, `pcc status`, `pcc pause`, `pcc resume`, `pcc attach`

### Phase 6: Master Agent (ReAct) (DONE)

**Goal**: Full orchestration with Think -> Act -> Observe -> Learn loop.

**Implemented files**:
1. `perpetualcc/core/master_agent.py`
   - `AnalysisType` enum: PERMISSION_REQUEST, QUESTION, RATE_LIMIT, TASK_COMPLETE, etc.
   - `ActionType` enum: APPROVE_TOOL, DENY_TOOL, ANSWER_QUESTION, ESCALATE_TO_HUMAN, etc.
   - `Analysis` dataclass with event analysis results
   - `Action` dataclass with action decisions
   - `Episode` dataclass for learning/memory integration
   - `MasterAgentConfig` with confidence thresholds and retry settings
   - `MasterAgent` class orchestrating all components

**MasterAgent Methods**:
- `run_session(session)` - main ReAct event loop
- `_think(event, session) -> Analysis` - analyze events, classify type
- `_decide(analysis, session) -> Action` - route to appropriate handler
- `_execute(action, session) -> dict` - perform actions, log results
- `_learn(analysis, action, result, session)` - record episodes for future memory

**ReAct Loop Flow**:
1. **THINK**: Analyze event type (permission, question, rate limit, result)
2. **ACT**: Decide appropriate action based on analysis
3. **EXECUTE**: Perform action (approve/deny/answer/escalate)
4. **LEARN**: Record episode if novel (for Phase 8 Memory)

**Integration Points**:
- Uses `DecisionEngine` for permission decisions (Phase 2)
- Uses `Brain` for question answering (Phase 3)
- Uses `RateLimitMonitor` for rate limit handling (Phase 4)
- Uses `SessionManager.stream_events()` for event streaming (Phase 5)
- Prepares for `KnowledgeEngine` integration (Phase 7)
- Prepares for `MemoryStore` integration (Phase 8)

**Tests**: 62 unit tests covering all scenarios

### Phase 7: Knowledge Engine

**Goal**: RAG-powered project understanding.

**Dependencies**: Install with `uv pip install -e ".[knowledge]"`

**Files**:
1. `perpetualcc/knowledge/indexer.py` - Scan codebase, chunk code
2. `perpetualcc/knowledge/embeddings.py` - Ollama/Gemini embeddings
3. `perpetualcc/knowledge/rag.py` - ChromaDB vector store
4. `perpetualcc/knowledge/code_graph.py` - tree-sitter + NetworkX

**CLI**: Add `pcc init` command to index a project.

### Phase 8: Memory Systems

**Goal**: Learn from experience.

**Database**: SQLite at `~/.perpetualcc/data/perpetualcc.db`

**Files**:
1. `perpetualcc/memory/store.py` - SQLite with aiosqlite
2. `perpetualcc/memory/episodic.py` - Event history with embeddings
3. `perpetualcc/memory/procedural.py` - Learned rules with confidence
4. `perpetualcc/memory/semantic.py` - Project facts

### Phase 9: Human Bridge

**Goal**: Seamless human intervention.

**Files**:
1. `perpetualcc/human/notifications.py` - macOS osascript notifications
2. `perpetualcc/human/escalation.py` - Queue and routing
3. `perpetualcc/human/cli_prompt.py` - Rich interactive prompts

**CLI**: Add `pcc pending` and `pcc respond` commands.

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_risk_classifier.py

# Run with coverage
pytest --cov=perpetualcc

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Common Tasks

### Adding a New CLI Command

1. Add command function to `perpetualcc/cli.py`
2. Decorate with `@app.command()`
3. Test manually: `pcc <command> --help`

### Adding a New Event Type

1. Add dataclass to `perpetualcc/claude/types.py`
2. Handle in `ClaudeCodeAdapter._convert_message()`
3. Handle in `cli.py:_print_event()` for display

### Adding a New Brain Implementation

1. Create file in `perpetualcc/brain/`
2. Implement `Brain` ABC from `base.py`
3. Register in brain factory (TODO: implement factory pattern)

## Environment Variables

- `ANTHROPIC_API_KEY`: Required for Claude Code sessions
- `GEMINI_API_KEY`: Optional, for Gemini brain
- No env vars needed for Ollama (uses local server)

## Data Storage Locations

- `~/.perpetualcc/config.toml`: User configuration
- `~/.perpetualcc/data/`: Runtime data directory
  - `perpetualcc.db`: SQLite database
  - `checkpoints/`: Session checkpoints
  - `chromadb/`: Vector store (if using RAG)
- `.perpetualcc/`: Per-project data (created by `pcc init`)

## Useful Links

- [Claude Agent SDK Docs](https://platform.claude.com/docs/en/agent-sdk/python)
- [Claude Agent SDK GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Typer Docs](https://typer.tiangolo.com/)
- [Rich Docs](https://rich.readthedocs.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)
