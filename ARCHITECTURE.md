# PerpetualCC - Architecture & Implementation Plan

## Language: Python
## Interface: CLI first, TUI/Web later

---

## Overview

PerpetualCC is an intelligent master agent that orchestrates Claude Code sessions 24/7.
It acts like an experienced human developer - understanding project context, making informed
decisions about tool permissions, answering Claude Code's questions, and knowing when to
escalate to a real human.

---

## Session Management: How It Works

### How Claude Code Sessions Work

Sessions are stored at `~/.claude/`:
- `~/.claude/session-env/{uuid}/` - Individual session environments
- `~/.claude/projects/{path-encoded}/sessions-index.json` - Per-project session index

Session index contains: version, session entries with sessionId, fullPath, messageCount, timestamps, gitBranch, and projectPath.

### Getting Session IDs

| Method | Description |
|--------|-------------|
| **SDK (primary)** | Capture `session_id` from init message in `query()` stream |
| **Resume** | Pass `resume="session-uuid"` to `ClaudeAgentOptions` |
| **Fork** | Pass `resume="session-id"` + `fork_session=True` |
| **CLI** | `claude --resume` (interactive) or `claude --resume SESSION_ID` |
| **VSCode** | Past Conversations dropdown (internal management) |
| **Filesystem** | Parse `sessions-index.json` directly |

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                    PerpetualCC                                            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           KNOWLEDGE ENGINE (Phase 7-8)                            │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐  │   │
│  │  │   RAG Pipeline     │  │    Code Graph      │  │     Memory Systems         │  │   │
│  │  │  ┌──────────────┐  │  │  ┌──────────────┐  │  │  ┌────────────────────┐   │  │   │
│  │  │  │ ChromaDB     │  │  │  │ tree-sitter  │  │  │  │ Episodic Memory    │   │  │   │
│  │  │  │ Vector Store │  │  │  │ AST Parser   │  │  │  │ (event history,    │   │  │   │
│  │  │  └──────┬───────┘  │  │  └──────┬───────┘  │  │  │  similarity search)│   │  │   │
│  │  │         │          │  │         │          │  │  └────────────────────┘   │  │   │
│  │  │  ┌──────▼───────┐  │  │  ┌──────▼───────┐  │  │  ┌────────────────────┐   │  │   │
│  │  │  │ Embeddings   │  │  │  │  NetworkX    │  │  │  │ Semantic Memory    │   │  │   │
│  │  │  │ (Ollama/     │  │  │  │  Directed    │  │  │  │ (project facts,    │   │  │   │
│  │  │  │  Gemini)     │  │  │  │  Graph       │  │  │  │  conventions)      │   │  │   │
│  │  │  └──────────────┘  │  │  └──────────────┘  │  │  └────────────────────┘   │  │   │
│  │  │                    │  │                    │  │  ┌────────────────────┐   │  │   │
│  │  │  Retrieves relevant│  │  Maps imports,     │  │  │ Procedural Memory  │   │  │   │
│  │  │  code chunks for   │  │  functions, classes│  │  │ (learned rules,    │   │  │   │
│  │  │  context-aware     │  │  and dependencies  │  │  │  confidence scores)│   │  │   │
│  │  │  answering         │  │                    │  │  └────────────────────┘   │  │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       MASTER AGENT - ReAct Supervisor (Phase 6)                   │   │
│  │                                                                                    │   │
│  │    ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│  │    │                         ReAct Loop (continuous)                          │   │   │
│  │    │  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │   │
│  │    │  │  THINK   │────▶│   ACT    │────▶│ OBSERVE  │────▶│  LEARN   │───┐   │   │   │
│  │    │  │          │     │          │     │          │     │          │   │   │   │   │
│  │    │  │ Analyze  │     │ Execute  │     │ Check    │     │ Record   │   │   │   │   │
│  │    │  │ event,   │     │ decision,│     │ results, │     │ episode, │   │   │   │   │
│  │    │  │ classify │     │ respond  │     │ update   │     │ update   │   │   │   │   │
│  │    │  │ type     │     │ to SDK   │     │ state    │     │ rules    │   │   │   │   │
│  │    │  └──────────┘     └──────────┘     └──────────┘     └────┬─────┘   │   │   │   │
│  │    │       ▲                                                  │         │   │   │   │
│  │    │       └──────────────────────────────────────────────────┴─────────┘   │   │   │
│  │    └─────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                    │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │  Decision Engine  │  │   Brain Layer     │  │   Checkpoint Manager        │   │   │
│  │  │  (Phase 2)        │  │   (Phase 3)       │  │   (Phase 4)                 │   │   │
│  │  │                   │  │                   │  │                             │   │   │
│  │  │  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌───────────────────────┐  │   │   │
│  │  │  │    Risk     │  │  │  │ Rule-Based  │  │  │  │ Save session state:   │  │   │   │
│  │  │  │ Classifier  │  │  │  │ (default)   │  │  │  │ - Current task        │  │   │   │
│  │  │  │             │  │  │  ├─────────────┤  │  │  │ - Completed subtasks  │  │   │   │
│  │  │  │ LOW→approve │  │  │  │ Gemini API  │  │  │  │ - Key decisions       │  │   │   │
│  │  │  │ MED→brain   │  │  │  ├─────────────┤  │  │  │ - Modified files      │  │   │   │
│  │  │  │ HIGH→human  │  │  │  │ Ollama LLM  │  │  │  │ - Conversation summary│  │   │   │
│  │  │  └─────────────┘  │  │  └─────────────┘  │  │  └───────────────────────┘  │   │   │
│  │  └───────────────────┘  └───────────────────┘  └─────────────────────────────┘   │   │
│  │                                                                                    │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Rate Limit Monitor (Phase 4)                             │   │   │
│  │  │  Detects: 429 errors • "rate/usage/token limit" text • Low remaining tokens │   │   │
│  │  │  Actions: Save checkpoint → Wait with countdown → Auto-resume with context  │   │   │
│  │  └────────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          │ Session Manager (Phase 5)                     │
│                                          │ Manages multiple concurrent sessions          │
│         ┌────────────────────────────────┼────────────────────────────────┐             │
│         │                                │                                │             │
│         ▼                                ▼                                ▼             │
│  ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐     │
│  │  Claude Code      │         │  Claude Code      │         │  Claude Code      │     │
│  │  Session 1        │         │  Session 2        │         │  Session N        │     │
│  │                   │         │                   │         │                   │     │
│  │  ┌─────────────┐  │         │  ┌─────────────┐  │         │  ┌─────────────┐  │     │
│  │  │ SDK Stream  │  │         │  │ SDK Stream  │  │         │  │ SDK Stream  │  │     │
│  │  │ (events)    │  │         │  │ (events)    │  │         │  │ (events)    │  │     │
│  │  └─────────────┘  │         │  └─────────────┘  │         │  └─────────────┘  │     │
│  │                   │         │                   │         │                   │     │
│  │  Status: RUNNING  │         │  Status: PAUSED   │         │  Status: LIMITED  │     │
│  │  Task: "Build auth│         │  Task: "Add tests"│         │  Task: "Refactor" │     │
│  │  system"          │         │  Queue: 2 pending │         │  Resumes: 17:30   │     │
│  └───────────────────┘         └───────────────────┘         └───────────────────┘     │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         HUMAN BRIDGE (Phase 9)                                    │   │
│  │                                                                                    │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────────┐   │   │
│  │  │   CLI Interface    │  │  macOS Notifier    │  │   Escalation Queue       │   │   │
│  │  │                    │  │                    │  │                          │   │   │
│  │  │  • pcc attach      │  │  • osascript       │  │  • Pending decisions     │   │   │
│  │  │  • pcc pending     │  │    notifications   │  │  • Priority ordering     │   │   │
│  │  │  • pcc respond     │  │  • Sound alerts    │  │  • Timeout handling      │   │   │
│  │  │  • Interactive     │  │  • Badge updates   │  │  • Response routing      │   │   │
│  │  │    rich prompts    │  │                    │  │                          │   │   │
│  │  └────────────────────┘  └────────────────────┘  └──────────────────────────┘   │   │
│  │                                                                                    │   │
│  │  Escalation Triggers: Low confidence (<0.7) • HIGH risk operations • Errors       │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components (Implementation Order)

Each component is designed to be implementable independently with minimal cross-dependencies.

---

### Component 1: CLI Scaffold + Claude Code Adapter

**Purpose**: Basic CLI that can start a Claude Code session and stream output.

**Files:**
- `perpetualcc/__init__.py`
- `perpetualcc/__main__.py`
- `perpetualcc/cli.py` - CLI commands (typer)
- `perpetualcc/claude/adapter.py` - Claude Code SDK wrapper
- `perpetualcc/claude/types.py` - Message types, session state
- `pyproject.toml` - Project config

**Key Interfaces:**

| Class | Method | Description |
|-------|--------|-------------|
| `ClaudeCodeAdapter` | `start_session(project_path, task, allowed_tools)` | Start new session, return session_id |
| | `resume_session(session_id, prompt)` | Resume existing session |
| | `stream_events()` | Yield typed events from active session |
| | `respond(response)` | Send response to Claude Code |

**CLI Commands:**

| Command | Description |
|---------|-------------|
| `pcc start <path> --task "..."` | Start session with inline task |
| `pcc start <path> --requirements file.md` | Start with requirements file |

**Deliverable**: Can start a Claude Code session and see streaming output.

---

### Component 2: Decision Engine (Permission Handler)

**Purpose**: Decide whether to approve/deny tool use requests from Claude Code.

**Files:**
- `perpetualcc/core/decision_engine.py`
- `perpetualcc/core/risk_classifier.py`

**Key Types:**

| Type | Fields | Description |
|------|--------|-------------|
| `RiskLevel` | `LOW`, `MEDIUM`, `HIGH` | Enum for risk classification |
| `PermissionDecision` | `approve`, `confidence`, `reason` | Decision result with explanation |

**DecisionEngine Flow:**

```
Tool Request → classify_risk()
       │
       ├─ LOW risk    → Auto-approve (confidence: 0.95)
       │
       ├─ MEDIUM risk → Brain.evaluate_permission() → Decision based on context
       │
       └─ HIGH risk   → Deny, escalate to human (confidence: 0.0)
```

**Risk classification rules:**

| Risk | Tool | Condition |
|------|------|-----------|
| LOW | Read | Any file |
| LOW | Glob, Grep | Any pattern |
| LOW | Write, Edit | File within project `src/`, `tests/`, `lib/` |
| LOW | Bash | `npm`, `yarn`, `pip`, `pytest`, `cargo` commands |
| MEDIUM | Write, Edit | Config files, outside src/ |
| MEDIUM | Bash | `git` commands, network commands |
| HIGH | Bash | `rm`, `sudo`, `chmod`, pipes to files, `curl \| sh` |
| HIGH | Write | `.env`, credentials, system files |
| HIGH | Bash | Any command with `--force`, `-f` on destructive ops |

**Deliverable**: Claude Code auto-approves safe operations, blocks dangerous ones.

---

### Component 3: Brain Layer (Question Answering)

**Purpose**: Answer Claude Code's questions with project context awareness.

**Files:**
- `perpetualcc/brain/base.py` - Abstract brain interface
- `perpetualcc/brain/rule_based.py` - Default, no AI needed
- `perpetualcc/brain/gemini.py` - Gemini API brain
- `perpetualcc/brain/ollama.py` - Local LLM brain

**Key Types:**

| Type | Fields | Description |
|------|--------|-------------|
| `BrainAnswer` | `selected`, `confidence`, `reasoning` | Answer with explanation |
| `QuestionContext` | `project_path`, `current_task`, `requirements_text` | Context for answering |

**Brain Interface (ABC):**

| Method | Parameters | Returns |
|--------|------------|---------|
| `answer_question()` | question, options, context | `BrainAnswer` |
| `evaluate_permission()` | tool_name, tool_input, context | `PermissionDecision` |

**Brain implementations:**

1. **Rule-based** (default):
   - Pattern matching on common question types
   - "Proceed/continue?" → Yes
   - Questions about project conventions → check requirements file
   - Everything else → escalate to human

2. **Gemini** (`google-genai`):
   - Full context-aware answering
   - Uses project requirements + RAG context
   - Confidence-based escalation

3. **Ollama** (local LLM):
   - Same as Gemini but runs locally
   - DeepSeek, CodeLlama, or other models
   - Requires `ollama` running locally

**Deliverable**: Claude Code's questions get intelligent answers.

---

### Component 4: Rate Limit Monitor + Checkpoint

**Purpose**: Detect rate limits, save state, auto-resume when limits reset.

**Files:**
- `perpetualcc/core/rate_limit.py`
- `perpetualcc/core/checkpoint.py`

**RateLimitMonitor:**

| Detection | Description |
|-----------|-------------|
| HTTP 429 | Parse `retry-after` header |
| Text patterns | "rate limit", "usage limit", "token limit" |
| Token headers | Remaining tokens < 1000 |

| Method | Description |
|--------|-------------|
| `detect(event)` | Returns `RateLimitInfo` or None |
| `wait_for_reset(info)` | Async wait with countdown display |

**SessionCheckpoint Fields:**

| Field | Description |
|-------|-------------|
| `session_id`, `timestamp` | Session identification |
| `project_path`, `current_task` | Task context |
| `completed_subtasks`, `pending_subtasks` | Progress tracking |
| `modified_files` | List of {path, hash} for validation |
| `conversation_summary` | Brain-generated summary |
| `key_decisions` | List of {question, answer, rationale} |
| `next_action` | What to continue from |

**CheckpointManager:**

| Method | Description |
|--------|-------------|
| `save(session)` | Serialize state to `data/checkpoints/{session_id}/{timestamp}.json` |
| `restore(checkpoint)` | Generate context restoration prompt |
| `list_checkpoints(session_id)` | List all checkpoints for session |

**Resume Prompt Structure:**

```
CONTEXT RESTORATION:
├── Project & Original Task
├── Completed subtasks
├── Remaining subtasks
├── Key decisions made (with rationale)
├── Progress summary
└── Continue from: {next_action}
```

**Deliverable**: Sessions auto-resume after rate limits with full context.

---

### Component 5: Session Manager (Multi-Session)

**Purpose**: Manage multiple concurrent Claude Code sessions.

**Files:**
- `perpetualcc/core/session_manager.py`
- `perpetualcc/core/task_queue.py`

**SessionStatus Enum:**

| Status | Description |
|--------|-------------|
| `IDLE` | Session created, not yet started |
| `PROCESSING` | Actively processing events |
| `WAITING_INPUT` | Waiting for human response |
| `RATE_LIMITED` | Hit rate limit, waiting to resume |
| `PAUSED` | Manually paused by user |
| `COMPLETED` | Task finished successfully |
| `ERROR` | Unrecoverable error occurred |

**ManagedSession Fields:**

| Field | Description |
|-------|-------------|
| `session_id`, `project_path` | Session identification |
| `status` | Current SessionStatus |
| `current_task`, `task_queue` | Task tracking |
| `created_at`, `checkpoint` | Timing and state recovery |

**SessionManager Methods:**

| Method | Description |
|--------|-------------|
| `create_session(project_path, task)` | Create and start new session |
| `resume_session(session_id)` | Resume paused/rate-limited session |
| `add_task(session_id, task)` | Add task to session queue |
| `get_session(session_id)` | Get session by ID |
| `list_sessions()` | List all managed sessions |

**CLI Commands:**

| Command | Description |
|---------|-------------|
| `pcc list` | List all sessions with status |
| `pcc status <id>` | Detailed session info |
| `pcc add <id> "task"` | Add task to queue |
| `pcc pause <id>` | Pause session |
| `pcc resume <id>` | Resume session |

**Deliverable**: Run and manage multiple sessions from one CLI.

---

### Component 6: Master Agent (ReAct Supervisor)

**Purpose**: The main orchestration loop that ties everything together.

**Files:**
- `perpetualcc/core/master_agent.py`

**MasterAgent Architecture:**

The supervisor orchestrates Claude Code sessions using a **ReAct (Reasoning + Acting) loop**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ReAct Event Loop                                 │
│                                                                          │
│  Event Stream ──▶ THINK ──▶ ACT ──▶ OBSERVE ──▶ LEARN ──┐              │
│       ▲                                                   │              │
│       └───────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Dependencies Injected:**

| Dependency | Purpose |
|------------|---------|
| `SessionManager` | Event streaming, session state |
| `DecisionEngine` | Permission decisions |
| `Brain` | Question answering |
| `KnowledgeEngine` (optional) | RAG context retrieval |
| `MemoryStore` (optional) | Learning from experience |

**Event Analysis (_think):**

| Event Type | Analysis Type |
|------------|---------------|
| `tool_use` | `permission_request` |
| `ask_user_question` | `question` |
| `rate_limit` | `rate_limit` |
| `result` | `task_complete` |
| `error` | `error` |

**Decision Logic (_decide):**

| Analysis | Condition | Action |
|----------|-----------|--------|
| `permission_request` | Approved | `approve_tool` |
| | Confidence = 0 | `escalate_to_human` |
| | Denied | `deny_tool` |
| `question` | Confidence ≥ 0.7 | `answer` |
| | Confidence < 0.7 | `escalate_to_human` |
| `rate_limit` | — | `checkpoint_and_wait` |
| `task_complete` | Queue not empty | `next_task` |
| | Queue empty | `session_complete` |

**Deliverable**: Fully autonomous session orchestration.

---

### Component 7: Knowledge Engine (RAG + Code Graph)

**Purpose**: Give the master agent deep project understanding.

**Files:**
- `perpetualcc/knowledge/__init__.py`
- `perpetualcc/knowledge/indexer.py` - Scan and parse codebase
- `perpetualcc/knowledge/embeddings.py` - Hybrid embedding provider
- `perpetualcc/knowledge/rag.py` - Retrieval pipeline
- `perpetualcc/knowledge/code_graph.py` - Code relationship graph

**EmbeddingProvider Hierarchy:**

```
EmbeddingProvider (ABC)
├── OllamaEmbeddings  → Local via nomic-embed-text (~270MB)
├── GeminiEmbeddings  → API via genai.embed_content()
└── HybridEmbeddings  → Try local first, fall back to API
```

| Method | Description |
|--------|-------------|
| `embed(text)` | Single text → vector |
| `embed_batch(texts)` | Batch embedding for efficiency |

**RAGPipeline:**

| Method | Description |
|--------|-------------|
| `index_project(path)` | Scan → chunk → embed → store in ChromaDB |
| `retrieve(query, top_k=5)` | Query → embed → search → ranked results |

**CodeGraph (tree-sitter + NetworkX):**

| Method | Description |
|--------|-------------|
| `build_from_project(path)` | Parse AST, extract entities (files, functions, classes, imports), build directed graph |
| `get_related(entity, depth=2)` | Find related entities within N hops |
| `get_file_context(path)` | Get imports, exports, and callers for a file |

**Embedding strategy (hybrid):**
- Default: `ollama pull nomic-embed-text` (local, free, ~270MB)
- Fallback: Gemini embeddings API (requires key)
- Error if neither available with helpful message

**Deliverable**: Master agent understands the codebase structure and can retrieve relevant context.

---

### Component 8: Memory Systems

**Purpose**: Learn from past interactions to improve over time.

**Files:**
- `perpetualcc/memory/__init__.py`
- `perpetualcc/memory/store.py` - SQLite persistence
- `perpetualcc/memory/episodic.py` - Event history
- `perpetualcc/memory/procedural.py` - Learned rules
- `perpetualcc/memory/semantic.py` - Project knowledge

**Database Schema (SQLite):**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           episodes                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ id (PK) │ session_id │ timestamp │ event_type │ context │ action │     │
│         │            │           │            │         │        │     │
│ outcome ('success'/'failure') │ rationale │ embedding (BLOB for search)│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           procedures                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ id (PK) │ trigger_pattern (regex/glob) │ action │ confidence (0.0-1.0) │
│         │ success_count │ failure_count │ created_at │ updated_at       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                             facts                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ id (PK) │ project_path │ category │ fact │ source │ created_at         │
│         │ Categories: 'convention', 'architecture', 'decision'          │
└─────────────────────────────────────────────────────────────────────────┘
```

**MemoryStore Methods:**

| Method | Description |
|--------|-------------|
| `record_episode(episode)` | Store event with embedding for similarity search |
| `find_similar(context, top_k=3)` | Embed query, find similar past episodes |
| `update_procedure(trigger, outcome)` | Success: confidence +0.05 (cap 0.99), Failure: -0.10 (floor 0.1) |
| `get_procedures(event_type)` | Get applicable procedures for event |

**Deliverable**: Master agent learns from experience and improves over time.

---

### Component 9: Human Bridge (Notifications + Escalation)

**Purpose**: When confidence is low, involve the human seamlessly.

**Files:**
- `perpetualcc/human/__init__.py`
- `perpetualcc/human/escalation.py` - Queue and routing
- `perpetualcc/human/notifications.py` - macOS notifications
- `perpetualcc/human/cli_prompt.py` - Interactive CLI prompts

**EscalationRequest Fields:**

| Field | Description |
|-------|-------------|
| `session_id` | Which session needs input |
| `type` | `permission`, `question`, `error`, or `review` |
| `context` | What's happening (for human understanding) |
| `options` | Available choices |
| `brain_suggestion`, `brain_confidence` | AI recommendation |
| `timestamp` | When escalated |

**EscalationQueue Methods:**

| Method | Description |
|--------|-------------|
| `escalate(request)` | Add to queue → notify → wait for response |
| `get_pending()` | List pending escalations |
| `respond(request_id, answer)` | Submit human response |

**CLI Interaction Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│ [14:30:22] ⚠️  Human input needed for session abc123            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude asks: "Should I use PostgreSQL or SQLite?"              │
│  Brain suggestion: "PostgreSQL" (confidence: 62%)                │
│                                                                  │
│  Options:                                                        │
│    [1] PostgreSQL - Full-featured, scalable                     │
│    [2] SQLite - Lightweight, no server needed                   │
│    [3] Type custom response                                     │
│                                                                  │
│  [A]ccept brain  [1-2] Select  [T]ype custom  > _               │
└─────────────────────────────────────────────────────────────────┘
```

**Deliverable**: Human can review and respond to escalated decisions.

---

## Project Structure

```
perpetualcc/
├── pyproject.toml                # Project config, dependencies, CLI entry
├── README.md
│
├── perpetualcc/
│   ├── __init__.py
│   ├── __main__.py              # python -m perpetualcc
│   ├── cli.py                   # Typer CLI commands
│   │
│   ├── claude/
│   │   ├── __init__.py
│   │   ├── adapter.py           # Claude Code SDK wrapper
│   │   └── types.py             # Event types, message types
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── master_agent.py      # ReAct supervisor loop
│   │   ├── session_manager.py   # Multi-session management
│   │   ├── decision_engine.py   # Permission decisions
│   │   ├── risk_classifier.py   # Risk level classification
│   │   ├── checkpoint.py        # Save/restore state
│   │   ├── rate_limit.py        # Rate limit detection
│   │   └── task_queue.py        # Task ordering
│   │
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract brain interface
│   │   ├── rule_based.py        # No-AI brain (default)
│   │   ├── gemini.py            # Gemini API brain
│   │   └── ollama.py            # Local LLM brain
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── indexer.py           # Codebase scanner
│   │   ├── embeddings.py        # Hybrid embedding provider
│   │   ├── rag.py               # Retrieval pipeline
│   │   └── code_graph.py        # Code relationship graph
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py             # SQLite persistence
│   │   ├── episodic.py          # Event history
│   │   ├── procedural.py        # Learned rules
│   │   └── semantic.py          # Project knowledge
│   │
│   └── human/
│       ├── __init__.py
│       ├── escalation.py        # Escalation queue
│       ├── notifications.py     # macOS notifications
│       └── cli_prompt.py        # Interactive prompts
│
├── tests/
│   ├── unit/
│   │   ├── test_decision_engine.py
│   │   ├── test_risk_classifier.py
│   │   ├── test_rate_limit.py
│   │   └── test_checkpoint.py
│   └── integration/
│       ├── test_claude_adapter.py
│       └── test_brain.py
│
└── data/                        # Runtime data (gitignored)
    ├── perpetualcc.db           # SQLite database
    └── chromadb/                # Vector store
```

---

## Dependencies

**Project:** perpetualcc v0.1.0, requires Python >=3.11

**Core Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| claude-agent-sdk | >=0.1.0 | Claude Code SDK integration |
| typer | >=0.12.0 | CLI framework |
| rich | >=13.0.0 | Terminal formatting |
| anyio | >=4.0.0 | Async utilities |
| pydantic | >=2.0.0 | Data validation |

**Brain Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| google-genai | >=1.0.0 | Gemini API brain |
| ollama | >=0.4.0 | Local LLM brain |

**Knowledge Engine Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| chromadb | >=0.5.0 | Vector store |
| sentence-transformers | >=3.0.0 | Embeddings |
| networkx | >=3.0 | Code graph |
| tree-sitter | >=0.22.0 | AST parsing |
| tree-sitter-language-pack | >=0.3.0 | Language parsers |

**Optional:**

| Package | Version | Purpose |
|---------|---------|---------|
| faiss-cpu | >=1.8.0 | Alternative vector search |

**CLI Entry:** `pcc = "perpetualcc.cli:app"`

---

## Component Interactions

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   USER LAYER                                             │
│                                                                                          │
│    pcc start          pcc list         pcc attach        pcc respond                    │
│        │                  │                │                  │                          │
│        └──────────────────┴────────────────┴──────────────────┘                          │
│                                   │                                                      │
│                                   ▼                                                      │
│                         ┌─────────────────┐                                             │
│                         │   CLI (typer)   │                                             │
│                         │   + rich UI     │                                             │
│                         └────────┬────────┘                                             │
└──────────────────────────────────┼───────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER                                         │
│                                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           SessionManager                                           │  │
│  │  • Creates/destroys sessions    • Manages task queues    • Persists to SQLite     │  │
│  └───────────────────────────────────┬───────────────────────────────────────────────┘  │
│                                      │                                                   │
│                                      │ spawns per session                               │
│                                      ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            MasterAgent (ReAct)                                     │  │
│  │                                                                                    │  │
│  │   Event ──▶ _think() ──▶ _decide() ──▶ _execute() ──▶ _learn()                   │  │
│  │              │               │              │              │                       │  │
│  │              │               │              │              │                       │  │
│  │         ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐                 │  │
│  │         │ Classify│    │ Route to│    │ Call SDK│    │ Record  │                 │  │
│  │         │ event   │    │ handler │    │ or Human│    │ episode │                 │  │
│  │         └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                                   │
└──────────────────────────────────────┼───────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   DECISION LAYER    │  │    BRAIN LAYER      │  │   KNOWLEDGE LAYER   │
│                     │  │                     │  │                     │
│  DecisionEngine     │  │  Brain (ABC)        │  │  KnowledgeEngine    │
│       │             │  │       │             │  │       │             │
│       ▼             │  │       ├─ RuleBased  │  │       ├─ RAGPipeline│
│  RiskClassifier     │  │       ├─ Gemini     │  │       ├─ CodeGraph  │
│                     │  │       └─ Ollama     │  │       └─ Embeddings │
│  LOW → approve      │  │                     │  │                     │
│  MED → brain        │  │  Answers questions  │  │  Retrieves relevant │
│  HIGH → human       │  │  with confidence    │  │  code context       │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               PERSISTENCE LAYER                                          │
│                                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────────────────┐   │
│  │  CheckpointManager │  │   MemoryStore      │  │        ChromaDB                 │   │
│  │                    │  │                    │  │                                 │   │
│  │  • Session state   │  │  • Episodes table  │  │  • Code chunk embeddings       │   │
│  │  • Task progress   │  │  • Procedures      │  │  • Semantic search             │   │
│  │  • Resume prompts  │  │  • Facts           │  │  • Similarity retrieval        │   │
│  └─────────┬──────────┘  └─────────┬──────────┘  └──────────────┬──────────────────┘   │
│            │                       │                             │                      │
│            └───────────────────────┼─────────────────────────────┘                      │
│                                    ▼                                                     │
│                          ┌─────────────────────┐                                        │
│                          │      SQLite DB      │                                        │
│                          │  ~/.perpetualcc/    │                                        │
│                          │    data/            │                                        │
│                          │      perpetualcc.db │                                        │
│                          └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                EXTERNAL LAYER                                            │
│                                                                                          │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │         ClaudeCodeAdapter           │  │              HumanBridge                 │  │
│  │                                     │  │                                          │  │
│  │  ┌─────────────────────────────┐   │  │  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │    Claude Agent SDK         │   │  │  │  macOS       │  │  EscalationQueue│  │  │
│  │  │    (claude_agent_sdk)       │   │  │  │  Notifier    │  │                 │  │  │
│  │  └──────────────┬──────────────┘   │  │  │  (osascript) │  │  Pending items  │  │  │
│  │                 │                   │  │  └──────────────┘  └─────────────────┘  │  │
│  │                 ▼                   │  │                                          │  │
│  │  Events: ToolUse, Question,         │  │  Escalation Triggers:                   │  │
│  │          RateLimit, Result, Error   │  │  • Confidence < 0.7                     │  │
│  │                                     │  │  • HIGH risk permission                 │  │
│  │  Methods: start_session(),          │  │  • Unrecoverable error                  │  │
│  │           stream_events(),          │  │                                          │  │
│  │           respond_permission(),     │  │  Response Methods:                       │  │
│  │           respond_answer()          │  │  • CLI interactive prompt               │  │
│  └─────────────────────────────────────┘  │  • pcc pending / pcc respond            │  │
│                                            └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Interaction Contracts

#### CLI → SessionManager

**Flow:**
1. CLI calls `SessionManager.create_session(project_path, task)`
2. SessionManager creates ClaudeCodeAdapter instance
3. SessionManager calls `adapter.start_session()`
4. SessionManager wraps in ManagedSession with status tracking
5. SessionManager starts `MasterAgent.run_session()` as async task
6. SessionManager returns ManagedSession for CLI to display

#### SessionManager → MasterAgent

**Flow:**
1. SessionManager creates MasterAgent per session with all dependencies injected
2. MasterAgent runs the event loop via `run_session(session)`
3. Dependencies: session_manager, decision_engine, brain, knowledge_engine (optional), memory (optional)

#### MasterAgent → ClaudeCodeAdapter

**Event Types:**

| Event | Fields | MasterAgent Response |
|-------|--------|---------------------|
| `ToolUseEvent` | tool_name, tool_input | `respond_permission(decision)` |
| `AskQuestionEvent` | questions | `respond_answer(answer)` |
| `ResultEvent` | is_error, result | Update session state |
| `RateLimitEvent` | retry_after, reset_time | Save checkpoint, wait, resume |

#### MasterAgent → DecisionEngine → Brain

**Permission Flow:**

```
ToolUseEvent
    │
    ▼
DecisionEngine.decide_permission()
    │
    ▼
RiskClassifier.classify()
    │
    ├─ LOW risk   ──▶ Approve immediately
    │
    ├─ MEDIUM risk ──▶ Brain.evaluate_permission() ──▶ Decision
    │
    └─ HIGH risk  ──▶ Escalate to human
```

**Question Flow:**

```
AskQuestionEvent
    │
    ├──▶ KnowledgeEngine.retrieve() ──▶ Relevant code context
    │
    ├──▶ MemoryStore.find_similar() ──▶ Similar past episodes
    │
    └──▶ Brain.answer_question(question, context, memory)
              │
              ├─ Confidence ≥ 0.7 ──▶ Send answer to adapter
              │
              └─ Confidence < 0.7 ──▶ Escalate to HumanBridge
```

#### Brain → KnowledgeEngine → MemoryStore

**Context-Aware Answering (GeminiBrain example):**

```
Question received
    │
    ├──▶ RAG: retrieve(question, top_k=5) ──▶ Relevant code chunks
    │
    ├──▶ Memory: find_similar(question, top_k=3) ──▶ Past episodes
    │
    ├──▶ Memory: get_facts(project_path) ──▶ Project conventions
    │
    └──▶ Build prompt with all context ──▶ Call Gemini ──▶ BrainAnswer
```

#### CheckpointManager → RateLimitMonitor → SessionManager

**Rate Limit Recovery:**

```
RateLimitEvent detected
    │
    ├──▶ CheckpointManager.save(session) ──▶ Serialize to SQLite
    │
    ├──▶ SessionManager: status = RATE_LIMITED
    │
    ├──▶ RateLimitMonitor.wait_for_reset() ──▶ Sleep with countdown
    │
    ├──▶ SessionManager.resume_session() ──▶ adapter.resume_session(prompt)
    │
    └──▶ MasterAgent continues event loop
```

#### HumanBridge ← MasterAgent (Escalation)

**Escalation Flow:**

```
Low confidence / HIGH risk detected
    │
    ├──▶ Create EscalationRequest
    │
    ├──▶ HumanBridge.escalate()
    │        │
    │        ├──▶ Send macOS notification (osascript)
    │        ├──▶ Add to pending queue
    │        │
    │        └──▶ If attached: show interactive prompt
    │             If detached: wait for `pcc respond`
    │
    ├──▶ Human provides response
    │
    ├──▶ MasterAgent sends response to adapter
    │
    └──▶ Memory records decision for future learning
```

---

## Detailed Implementation Phases

### Phase 1: CLI + Claude Code Adapter

**Goal**: `pcc start ./project --task "task"` runs a Claude Code session and streams output.

**Step-by-step implementation:**

1. **Create project structure**
   ```bash
   mkdir -p perpetualcc/{claude,core,brain,knowledge,memory,human}
   touch perpetualcc/__init__.py perpetualcc/__main__.py
   ```

2. **Create `pyproject.toml`**
   - Project metadata, Python >=3.11
   - Dependencies: `claude-agent-sdk`, `typer`, `rich`, `anyio`, `pydantic`
   - Script entry: `pcc = "perpetualcc.cli:app"`

3. **Implement `perpetualcc/claude/types.py`**
   - Define `ClaudeEvent` base class
   - Define event subtypes: `ToolUseEvent`, `AskQuestionEvent`, `TextEvent`, `ResultEvent`, `RateLimitEvent`
   - Define `SessionState` enum

4. **Implement `perpetualcc/claude/adapter.py`**
   - `ClaudeCodeAdapter.__init__(project_path, anthropic_api_key?)`
   - `start_session(task, allowed_tools?) → session_id`
     - Calls `claude_agent_sdk.query()` with options
     - Captures session_id from init message
   - `stream_events() → AsyncIterator[ClaudeEvent]`
     - Converts SDK messages to our typed events
     - Handles text blocks, tool use blocks, result messages
   - `respond_permission(decision: PermissionDecision)`
   - `respond_answer(answer: str)`
   - `resume_session(session_id, prompt)`

5. **Implement `perpetualcc/cli.py`**
   - Typer app with `start` command
   - `pcc start <project_path> --task <task>`
   - Streams output to terminal using `rich.console`
   - Shows: timestamps, Claude's text, tool uses, results
   - Ctrl+C gracefully stops

6. **Implement `perpetualcc/__main__.py`**
   - `python -m perpetualcc` entry point
   - Calls `app()` from cli.py

**Test**: Run `pcc start ./test-project --task "Create a hello.py file"` and see Claude Code create the file.

---

### Phase 2: Decision Engine

**Goal**: Auto-approve safe operations, block dangerous ones.

**Step-by-step implementation:**

1. **Implement `perpetualcc/core/risk_classifier.py`**
   - `classify_risk(tool_name, tool_input, project_path) → RiskLevel`
   - Rules based on tool type and input patterns:
     - Parse `tool_input["command"]` for Bash to detect dangerous patterns
     - Parse `tool_input["file_path"]` for Write/Edit to check location
     - Configurable safe directories (default: `src/`, `tests/`, `lib/`, `app/`)
   - Special patterns:
     - `rm -rf` → always HIGH
     - `git push --force` → HIGH
     - Write to `.env` → HIGH
     - `pip install` → LOW
     - Read anything → LOW

2. **Implement `perpetualcc/core/decision_engine.py`**
   - `DecisionEngine.__init__(brain, project_path)`
   - `decide_permission(tool_name, tool_input, context) → PermissionDecision`
   - Logging: log every decision with timestamp, tool, risk level, result

3. **Integrate with adapter**
   - Modify `ClaudeCodeAdapter` to accept a `permission_callback`
   - Wire `DecisionEngine.decide_permission` as the callback
   - Map SDK's `canUseTool` hook to our callback

**Test**: Unit tests with mock tool requests covering all risk levels. Integration: start session, verify Read auto-approves, `rm -rf /` gets blocked.

---

### Phase 3: Brain (Rule-Based)

**Goal**: Answer simple Claude Code questions automatically.

**Step-by-step implementation:**

1. **Implement `perpetualcc/brain/base.py`**
   - `Brain` ABC with `answer_question()` and `evaluate_permission()`
   - `QuestionContext` dataclass (project_path, current_task, requirements_text)
   - `BrainAnswer` dataclass (selected, confidence, reasoning)

2. **Implement `perpetualcc/brain/rule_based.py`**
   - Pattern matching rules:
     - "proceed/continue/go ahead?" → "Yes" (confidence: 0.85)
     - "which file/where to save?" → infer from project structure (confidence: 0.7)
     - Yes/No questions about standard operations → "Yes" (confidence: 0.75)
     - Questions mentioning options from requirements file → match keyword
   - Load requirements from `--requirements` file path
   - Parse requirements for keywords (tech stack, preferences)
   - Everything unmatched → confidence: 0.0 (escalate)

3. **Integrate with MasterAgent loop**
   - When `AskQuestionEvent` is received:
     - Call `brain.answer_question()`
     - If confidence >= threshold → auto-answer
     - If below threshold → print to CLI, wait for user input

**Test**: Mock questions like "Should I proceed?", "Which database?", verify correct answers.

---

### Phase 4: Rate Limit + Checkpoint

**Goal**: Detect rate limits, save state, auto-wait, resume.

**Step-by-step implementation:**

1. **Implement `perpetualcc/core/rate_limit.py`**
   - `RateLimitMonitor.detect(event) → RateLimitInfo | None`
   - Detection patterns:
     - Error status 429 → parse `retry-after` header
     - Text containing "rate limit", "usage limit", "token limit"
     - Remaining tokens header < 1000
   - `wait_for_reset(info) → async wait`
     - Uses `asyncio.sleep()` with countdown display
     - Shows "Resuming in X:XX..."

2. **Implement `perpetualcc/core/checkpoint.py`**
   - `SessionCheckpoint` dataclass (all session state)
   - `CheckpointManager.__init__(data_dir: Path)`
   - `save(session) → SessionCheckpoint`
     - Serialize session state to JSON
     - Store in `data/checkpoints/{session_id}/{timestamp}.json`
     - Generate conversation summary (use brain if available)
   - `restore(checkpoint) → resume_prompt: str`
     - Build context restoration prompt from checkpoint
     - Include task state, decisions, progress summary
   - `list_checkpoints(session_id) → list[SessionCheckpoint]`

3. **Wire into event loop**
   - On rate limit: save checkpoint → wait → resume with context
   - On approaching limit (85%): warn user via CLI output
   - On resume: validate file hashes, build prompt, call adapter.resume_session()

**Test**: Mock rate limit event, verify checkpoint saved to disk, verify resume prompt is coherent.

---

### Phase 5: Session Manager

**Goal**: Manage multiple concurrent sessions.

**Step-by-step implementation:**

1. **Implement `perpetualcc/core/task_queue.py`**
   - `TaskQueue.add(task: str, priority: int = 0)`
   - `TaskQueue.next() → str | None`
   - `TaskQueue.complete(task: str)`
   - Priority ordering, persistence to SQLite

2. **Implement `perpetualcc/core/session_manager.py`**
   - `SessionManager.__init__(config: Config)`
   - `create_session(project_path, task) → ManagedSession`
     - Creates adapter, decision engine, brain
     - Starts async event loop task
   - `list_sessions() → list[ManagedSession]`
   - `get_session(session_id) → ManagedSession | None`
   - `add_task(session_id, task)`
   - `pause_session(session_id)` / `resume_session(session_id)`
   - State persistence: save session list to SQLite on changes

3. **Extend CLI**
   - `pcc list` - table of sessions with status, task, tokens
   - `pcc status <id>` - detailed session info
   - `pcc add <id> "task"` - add task to queue
   - `pcc pause <id>` / `pcc resume <id>`
   - `pcc attach <id>` - live stream session output

4. **Concurrency model**
   - Each session runs in its own `asyncio.Task`
   - SessionManager coordinates via shared state (thread-safe dict)
   - Event emission for cross-session awareness

**Test**: Start two sessions, verify both run concurrently, `pcc list` shows both.

---

### Phase 6: Master Agent (ReAct)

**Goal**: Full orchestration with Think → Act → Observe → Learn loop.

**Step-by-step implementation:**

1. **Define Analysis and Action types**
   - `Analysis`: type (permission_request | question | rate_limit | task_complete | error), event, context, is_novel
   - `Action`: type (approve_tool | deny_tool | answer | escalate | checkpoint_and_wait | next_task | session_complete), value, reason

2. **Implement `perpetualcc/core/master_agent.py`**
   - `MasterAgent.__init__(...)` - takes all components
   - `run_session(session)` - main async event loop
   - `_think(event, session) → Analysis`
     - Classify event type
     - Gather context from knowledge engine
     - Check memory for similar past events
   - `_decide(analysis) → Action`
     - Route to appropriate handler
     - Apply confidence thresholds
   - `_execute(action, session)`
     - Send responses to adapter
     - Update session state
     - Trigger notifications
   - `_learn(event, action, result)`
     - Record episode in memory
     - Update procedural rules

3. **Error handling**
   - Max retries for transient errors (3 attempts)
   - Exponential backoff between retries
   - After max retries → pause session, notify human
   - Unrecoverable errors → log, notify, mark session ERROR

4. **Session completion**
   - When task completes: check task queue
   - If more tasks: start next task
   - If no more: mark COMPLETED, send notification, summarize results

**Test**: Full integration test with mock Claude Code responses covering all event types.

---

### Phase 7: Knowledge Engine

**Goal**: RAG-powered project understanding.

**Step-by-step implementation:**

1. **Implement `perpetualcc/knowledge/indexer.py`**
   - `CodebaseIndexer.__init__(project_path)`
   - `scan() → list[CodeFile]`
     - Walk directory, respect .gitignore
     - Parse each file: extract functions, classes, imports
     - Generate text chunks (by function/class, max ~500 tokens each)
   - `CodeFile` dataclass: path, language, chunks, metadata

2. **Implement `perpetualcc/knowledge/embeddings.py`**
   - `OllamaEmbeddings(model="nomic-embed-text")`
     - Calls `ollama.embed(model, input)` locally
   - `GeminiEmbeddings(api_key)`
     - Calls `genai.embed_content()`
   - `HybridEmbeddings(local, api?)`
     - Try local → fallback to API → raise if neither works

3. **Implement `perpetualcc/knowledge/rag.py`**
   - `RAGPipeline.__init__(embeddings, store_path)`
   - `index_project(project_path)`
     - Scan files → chunk → embed → store in ChromaDB
   - `retrieve(query, top_k=5) → list[RetrievalResult]`
     - Embed query → search ChromaDB → return ranked results
   - `RetrievalResult`: score, file_path, chunk_text, metadata

4. **Implement `perpetualcc/knowledge/code_graph.py`**
   - `CodeGraph.__init__()`
   - `build_from_project(project_path)`
     - Use tree-sitter to parse each file's AST
     - Extract: imports, function definitions, class definitions
     - Build NetworkX directed graph of relationships
   - `get_related(entity, depth=2) → list[str]`
   - `get_file_context(path) → dict` (imports, exports, callers)

5. **Integrate with Brain**
   - Before answering a question, retrieve relevant context
   - Include top-5 RAG results in brain prompt
   - Use code graph to understand related files

6. **CLI command: `pcc init`**
   - Scans project, builds index, stores in `.perpetualcc/`
   - Shows progress: "Indexing 47 files... Building code graph..."
   - Creates `.perpetualcc/` directory with ChromaDB + graph data

**Test**: Index a sample project, query "where is authentication handled?", verify relevant results.

---

### Phase 8: Memory Systems

**Goal**: Learn from interactions, improve over time.

**Step-by-step implementation:**

1. **Implement `perpetualcc/memory/store.py`**
   - SQLite database at `data/perpetualcc.db`
   - Schema migration on first run (create tables)
   - `MemoryStore.__init__(db_path)`
   - Connection pool with `aiosqlite`

2. **Implement `perpetualcc/memory/episodic.py`**
   - `EpisodicMemory.record(episode: Episode)`
     - Store event with embedded vector for similarity search
   - `EpisodicMemory.find_similar(context: str, top_k=3) → list[Episode]`
     - Embed query, compare against stored episode embeddings
     - Return most similar past events
   - `EpisodicMemory.get_recent(session_id, limit=10) → list[Episode]`

3. **Implement `perpetualcc/memory/procedural.py`**
   - `ProceduralMemory.match_procedure(event_type, input) → Procedure | None`
     - Match against stored trigger patterns
     - Return highest-confidence match
   - `ProceduralMemory.update(trigger, outcome: "success"|"failure")`
     - Success: confidence += 0.05 (capped at 0.99)
     - Failure: confidence -= 0.1 (floor at 0.1)
   - `ProceduralMemory.add_procedure(trigger, action, initial_confidence=0.5)`

4. **Implement `perpetualcc/memory/semantic.py`**
   - `SemanticMemory.add_fact(project, category, fact, source)`
   - `SemanticMemory.get_facts(project, category?) → list[Fact]`
   - Categories: "convention", "architecture", "decision", "preference"

5. **Integrate with MasterAgent**
   - After each decision: record episode
   - On success/failure: update procedural confidence
   - Before answering: check for similar past episodes

**Test**: Record 10 episodes, query for similar, verify retrieval relevance.

---

### Phase 9: Human Bridge

**Goal**: Seamless human intervention.

**Step-by-step implementation:**

1. **Implement `perpetualcc/human/notifications.py`**
   - macOS notification via `osascript` (no dependencies needed)
   - Command: `display notification "{message}" with title "PerpetualCC" subtitle "{session}" sound name "Ping"`
   - Notification types: question_pending, task_complete, rate_limited, error

2. **Implement `perpetualcc/human/escalation.py`**
   - `EscalationQueue.__init__()`
   - `escalate(request) → str` (blocking until human responds)
   - `get_pending() → list[EscalationRequest]`
   - `respond(request_id, answer: str)`
   - Storage: SQLite table for pending escalations

3. **Implement `perpetualcc/human/cli_prompt.py`**
   - Interactive prompt using `rich` + `typer`:
     - Show question context
     - Show brain suggestion with confidence
     - Numbered options
     - Free-text input option
   - Used in `pcc attach` mode (foreground)

4. **CLI commands for async escalation**
   - `pcc pending` - list pending human decisions
   - `pcc respond <request-id> <answer>` - answer from CLI
   - `pcc respond <request-id> --interactive` - opens rich prompt

**Test**: Trigger low-confidence question, verify notification appears, verify `pcc pending` shows it, verify `pcc respond` clears it.

---

## Detailed CLI Manual

### Installation

```bash
# From PyPI (once published)
pipx install perpetualcc

# From source (development)
git clone <repo>
cd perpetualcc
pip install -e ".[dev]"
```

### Global Options

```
pcc [OPTIONS] COMMAND [ARGS]

Options:
  --config PATH          Config file path (default: ~/.perpetualcc/config.toml)
  --data-dir PATH        Data directory (default: ~/.perpetualcc/data/)
  --verbose / --quiet    Output verbosity
  --version              Show version
  --help                 Show help
```

### Commands

#### `pcc init`

Initialize a project for PerpetualCC management.

```
pcc init <project-path> [OPTIONS]

Arguments:
  project-path           Path to the project directory

Options:
  --requirements PATH    Path to requirements file (markdown or YAML)
  --skip-index           Skip codebase indexing (faster init)

Example:
  pcc init ./my-project --requirements ./requirements.md
  pcc init .

Output:
  ✓ Scanning project structure... (47 files found)
  ✓ Building knowledge graph... (12 modules, 156 functions)
  ✓ Indexing codebase for RAG... (312 chunks embedded)
  ✓ Created .perpetualcc/ directory

  Project initialized. Start a session with:
    pcc start ./my-project --task "Your task here"
```

#### `pcc start`

Start a new Claude Code session.

```
pcc start <project-path> [OPTIONS]

Arguments:
  project-path           Path to the project directory

Options:
  --task TEXT            Task description (inline)
  --requirements PATH    Requirements file (markdown, YAML, or text)
  --brain TYPE           Brain type: none|gemini|ollama (default: none)
  --model TEXT           Model for brain (e.g., deepseek-coder:33b)
  --auto-approve         Auto-approve all LOW risk operations (default: true)
  --confidence FLOAT     Min confidence for auto-answer (default: 0.7)
  --max-turns INT        Max turns before pausing (default: unlimited)
  --detach               Start in background, don't stream output
  --session-name TEXT    Human-readable name for this session

Example:
  pcc start ./api --task "Build user authentication with JWT"
  pcc start . --requirements tasks.md --brain gemini --detach
  pcc start ./frontend --task "Add dark mode" --brain ollama --model deepseek-coder:33b

Output (foreground):
  ╭─ PerpetualCC Session: auth-feature ──────────────────────────╮
  │ Project: ./api                                                │
  │ Brain: rule-based                                             │
  │ Status: Running                                               │
  ╰───────────────────────────────────────────────────────────────╯

  [12:00:01] Starting Claude Code session...
  [12:00:05] Claude: Reading project structure...
  [12:00:08] [TOOL] Read → src/index.ts ✓ (auto-approved)
  [12:00:12] Claude: Creating authentication module...
  ...

  Ctrl+C to detach (session continues in background)
  Ctrl+D to stop session

Output (detached):
  Session started: abc123 (auth-feature)
  Attach with: pcc attach abc123
```

#### `pcc list`

List all managed sessions.

```
pcc list [OPTIONS]

Options:
  --status STATUS        Filter by status (running|paused|rate_limited|completed|error)
  --project PATH         Filter by project path
  --json                 Output as JSON

Example:
  pcc list
  pcc list --status running

Output:
  ┌──────────┬──────────────────┬────────────┬─────────────────────────────┬───────────┐
  │ ID       │ Name             │ Status     │ Current Task                │ Tokens    │
  ├──────────┼──────────────────┼────────────┼─────────────────────────────┼───────────┤
  │ abc123   │ auth-feature     │ ● Running  │ Implementing JWT middleware │ 45K/200K  │
  │ def456   │ dark-mode        │ ⏸ Paused   │ Adding theme toggle         │ 12K/200K  │
  │ ghi789   │ api-tests        │ ⏳ Limited │ Writing integration tests   │ 200K/200K │
  │          │                  │            │ Resumes: 17:30              │           │
  └──────────┴──────────────────┴────────────┴─────────────────────────────┴───────────┘
```

#### `pcc status`

Show detailed session status.

```
pcc status <session-id> [OPTIONS]

Options:
  --show-history         Show recent event history
  --show-decisions       Show brain decisions made

Example:
  pcc status abc123

Output:
  Session: abc123 (auth-feature)
  ──────────────────────────────────────
  Project:     ./api
  Status:      ● Running
  Brain:       gemini (confidence threshold: 0.7)
  Started:     2026-01-13 12:00:01
  Duration:    2h 15m

  Current Task: Implementing JWT middleware
  Progress:
    ✓ Created src/auth/jwt.ts
    ✓ Installed jsonwebtoken dependency
    → Implementing refresh token logic
    ○ Writing tests
    ○ Updating API documentation

  Token Usage: 45,231 / 200,000 (22%)
  Questions Answered: 3 (2 auto, 1 human)
  Tools Approved: 47 (45 auto, 2 human)

  Task Queue:
    1. Add password reset endpoint
    2. Implement rate limiting
```

#### `pcc attach`

Attach to a running session for live output and interaction.

```
pcc attach <session-id>

Example:
  pcc attach abc123

Output:
  Attached to session abc123 (auth-feature)
  ─────────────────────────────────────────

  [14:15:30] Claude: Implementing refresh token rotation...
  [14:15:35] [TOOL] Write → src/auth/refresh.ts ✓ (auto-approved)
  [14:15:40] [TOOL] Bash → npm test ✓ (auto-approved)

  [14:16:01] ⚠️  QUESTION from Claude:
             "Should refresh tokens expire after 7 days or 30 days?"
             Options: [1] 7 days  [2] 30 days
             Brain: "7 days" (confidence: 72%)

             [A]ccept brain  [1-2] Select  [T]ype custom  > _

  Ctrl+C to detach (session continues)
  Ctrl+D to stop session
```

#### `pcc add`

Add a task to a session's queue.

```
pcc add <session-id> <task-description> [OPTIONS]

Options:
  --priority INT         Priority (higher = sooner, default: 0)
  --after TEXT           Run after specific task completes

Example:
  pcc add abc123 "Add rate limiting to all endpoints"
  pcc add abc123 "Deploy to staging" --priority 10
```

#### `pcc pause` / `pcc resume`

Pause or resume a session.

```
pcc pause <session-id>     # Gracefully pauses after current operation
pcc resume <session-id>    # Resumes from checkpoint
```

#### `pcc pending`

List pending human decisions.

```
pcc pending [OPTIONS]

Options:
  --session TEXT          Filter by session ID

Output:
  Pending Human Decisions:
  ┌──────┬──────────┬────────┬─────────────────────────────────────────┬───────────┐
  │ #    │ Session  │ Type   │ Question                                │ Waiting   │
  ├──────┼──────────┼────────┼─────────────────────────────────────────┼───────────┤
  │ 1    │ abc123   │ Q&A    │ Should I use PostgreSQL or SQLite?      │ 5m ago    │
  │ 2    │ def456   │ Perm.  │ Approve: rm -rf node_modules/           │ 2m ago    │
  └──────┴──────────┴────────┴─────────────────────────────────────────┴───────────┘

  Respond with: pcc respond 1 "PostgreSQL"
  Or interactive: pcc respond 1 --interactive
```

#### `pcc respond`

Respond to a pending escalation.

```
pcc respond <request-number> [ANSWER] [OPTIONS]

Options:
  --interactive          Open rich interactive prompt

Example:
  pcc respond 1 "Use PostgreSQL, it's our standard"
  pcc respond 2 --approve    # For permission requests
  pcc respond 2 --deny       # For permission requests
  pcc respond 1 --interactive
```

#### `pcc config`

Configure PerpetualCC settings.

```
pcc config <subcommand>

Subcommands:
  brain       Configure the AI brain
  show        Show current configuration
  reset       Reset to defaults

Examples:
  # Configure Gemini brain
  pcc config brain gemini
  # Prompts for API key, stores in config

  # Configure Ollama brain
  pcc config brain ollama --model deepseek-coder:33b
  # Checks if Ollama is running, model is available

  # Configure rule-based (no AI)
  pcc config brain none

  # Show all config
  pcc config show

  # Set specific values
  pcc config set confidence_threshold 0.8
  pcc config set auto_approve true
  pcc config set notification_sound true
```

#### `pcc logs`

View session logs.

```
pcc logs <session-id> [OPTIONS]

Options:
  --tail INT             Show last N lines (default: 50)
  --follow / -f          Follow log output in real-time
  --level LEVEL          Filter: debug|info|warn|error

Example:
  pcc logs abc123 --tail 20
  pcc logs abc123 -f
```

### Configuration File

Located at `~/.perpetualcc/config.toml`

**[brain] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `type` | `"none"` | Brain type: `none`, `gemini`, or `ollama` |
| `confidence_threshold` | `0.7` | Min confidence for auto-answer |
| `auto_answer` | `true` | Enable auto-answering |

**[brain.gemini] Section:**

| Key | Description |
|-----|-------------|
| `api_key` | API key or `"env:GEMINI_API_KEY"` |
| `model` | Model name (default: `gemini-2.0-flash`) |

**[brain.ollama] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `http://localhost:11434` | Ollama server URL |
| `model` | — | Model name (e.g., `deepseek-coder:33b`) |

**[sessions] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `max_concurrent` | `5` | Max parallel sessions |
| `auto_resume` | `true` | Auto-resume after rate limits |
| `persist_history` | `true` | Save session history |

**[permissions] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `auto_approve_low_risk` | `true` | Auto-approve LOW risk operations |
| `safe_directories` | `["src/", "tests/", "lib/", "app/", "packages/"]` | Dirs for LOW risk writes |
| `blocked_commands` | `["rm -rf /", "sudo", "curl \| sh"]` | Always block these |

**[notifications] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable macOS notifications |
| `sound` | `true` | Play sound with notification |
| `on_question` | `true` | Notify on escalated questions |
| `on_complete` | `true` | Notify on task completion |
| `on_rate_limit` | `true` | Notify on rate limit |
| `on_error` | `true` | Notify on errors |

**[knowledge] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `embedding_provider` | `"local"` | Provider: `local`, `gemini`, or `hybrid` |
| `local_model` | `"nomic-embed-text"` | Ollama embedding model |
| `auto_index` | `true` | Re-index on session start |

**[data] Section:**

| Key | Default | Description |
|-----|---------|-------------|
| `directory` | `~/.perpetualcc/data` | Data storage location |
| `checkpoint_retention_days` | `30` | Keep checkpoints for N days |
| `episode_retention_days` | `90` | Keep episodes for N days |

### Requirements File Format

**Supported formats:** Markdown (recommended), YAML, or plain text

**Markdown Structure:**

| Section | Purpose |
|---------|---------|
| `# Project: Name` | Project title |
| `## Overview` | Brief description |
| `## Tasks` | Checkbox list of tasks (`- [ ] Task`) |
| `## Preferences` | Key-value preferences (Database, ORM, Auth, etc.) |
| `## Constraints` | Requirements and limitations |

**YAML Structure:**

| Field | Description |
|-------|-------------|
| `project` | Project name |
| `description` | Brief description |
| `tasks` | List with `id`, `description`, `priority`, `depends_on` |
| `preferences` | Map of preference keys to values |
| `constraints` | List of constraint strings |

**Brain uses these to:**
- Answer questions about project preferences
- Understand task dependencies
- Make context-aware decisions

---

## Implementation Order

| Phase | Component | What You Get |
|-------|-----------|--------------|
| **1** | CLI + Claude Adapter | `pcc start` runs a session |
| **2** | Decision Engine | Auto-approves safe operations |
| **3** | Brain (rule-based) | Answers simple questions |
| **4** | Rate Limit + Checkpoint | Detects limits, saves state |
| **5** | Session Manager | Multiple sessions |
| **6** | Master Agent | Full ReAct orchestration |
| **7** | Knowledge Engine | RAG + code understanding |
| **8** | Memory Systems | Learns from experience |
| **9** | Human Bridge | Notifications + escalation |

**Each phase is independently testable and useful.**

---

## Verification Plan

### Per-Component Testing

| Component | How to Verify |
|-----------|---------------|
| CLI + Adapter | `pcc start ./test-project --task "Create hello.py"` → see output |
| Decision Engine | Unit tests with mock tool requests |
| Brain | Unit tests with sample questions |
| Rate Limit | Mock rate limit events, verify detection |
| Checkpoint | Save/restore roundtrip test |
| Session Manager | `pcc list`, `pcc status`, multi-session |
| Master Agent | Integration test: full session with auto-answers |
| Knowledge Engine | Index a project, query for relevant code |
| Memory | Record episodes, verify retrieval |
| Human Bridge | Trigger escalation, verify notification |

### End-to-End Test

1. `pcc start ./sample-project --task "Build a REST API"`
2. Verify auto-approval of file reads/writes
3. Verify brain answers a question
4. Simulate rate limit → verify checkpoint saved
5. Wait/simulate reset → verify auto-resume with context
6. Verify task completion notification

---

## Requirements Satisfaction Checklist

| PRD Requirement | How Addressed |
|-----------------|---------------|
| Read requirements from input/file | CLI `--task` and `--requirements` flags |
| Detect token limits, auto-resume | RateLimitMonitor + CheckpointManager + auto-wait |
| Know which session to prompt | SessionManager tracks all sessions by ID |
| Answer Claude Code's questions | Brain layer (rule-based / Gemini / Ollama) |
| Manage multiple sessions | SessionManager + CLI `list`/`attach`/`status` |
| Human intervention during daytime | EscalationQueue + macOS notifications + CLI prompts |
| No AI brain option | RuleBasedBrain (default) |
| Gemini API brain | GeminiBrain with google-genai |
| Local LLM brain | OllamaBrain with ollama client |
| User-friendly setup | `pip install perpetualcc` + `pcc config brain` wizard |
| **NEW: Understand project context** | KnowledgeEngine (RAG + CodeGraph) |
| **NEW: Learn from experience** | MemoryStore (episodic + procedural) |
| **NEW: Act like human developer** | DecisionEngine + risk classification |
