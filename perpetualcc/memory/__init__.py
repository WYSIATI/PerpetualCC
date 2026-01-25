"""Memory systems - episodic, semantic, and procedural memory.

This module provides learning and memory capabilities for PerpetualCC:

- **Episodic Memory**: Records events/episodes with context, actions, and outcomes.
  Enables learning from past decisions through similarity search.

- **Procedural Memory**: Stores learned rules with confidence levels.
  Patterns adjust based on success/failure feedback.

- **Semantic Memory**: Stores project-specific facts and knowledge.
  Categories include conventions, architecture, decisions, and preferences.

All memory systems share a common SQLite persistence layer.

Usage:
    from perpetualcc.memory import (
        MemoryStore,
        EpisodicMemory,
        ProceduralMemory,
        SemanticMemory,
        Episode,
    )

    # Initialize store
    store = MemoryStore()
    await store.initialize()

    # Create memory systems
    episodic = EpisodicMemory(store, embedding_provider)
    procedural = ProceduralMemory(store)
    semantic = SemanticMemory(store)

    # Record an episode
    episode = Episode(
        timestamp=datetime.now(),
        session_id="session-123",
        event_type="permission_request",
        context="Tool use request: Write",
        action_taken="approve_tool",
        action_reason="Safe operation",
        outcome="success",
        confidence=0.95,
    )
    await episodic.record(episode)

    # Find similar past episodes
    similar = await episodic.find_similar("Tool use request: Read")

    # Match a procedure
    match = await procedural.match("tool_use", "Read:src/main.py")
    if match and match.procedure.confidence >= 0.7:
        action = match.procedure.action

    # Get project facts
    facts = await semantic.get_facts("/path/to/project", category="convention")
"""

from perpetualcc.memory.episodic import (
    Episode,
    EpisodicMemory,
    EpisodicMemoryConfig,
    SimilarEpisode,
)
from perpetualcc.memory.procedural import (
    ActionType,
    ProcedureMatch,
    ProceduralMemory,
    ProceduralMemoryConfig,
    TriggerType,
)
from perpetualcc.memory.semantic import (
    Fact,
    FactCategory,
    SemanticMemory,
    SemanticMemoryConfig,
)
from perpetualcc.memory.store import (
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    MemoryStore,
    MemoryStoreConfig,
    StoredEpisode,
    StoredFact,
    StoredProcedure,
)

__all__ = [
    # Store
    "MemoryStore",
    "MemoryStoreConfig",
    "StoredEpisode",
    "StoredProcedure",
    "StoredFact",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DB_PATH",
    # Episodic
    "Episode",
    "EpisodicMemory",
    "EpisodicMemoryConfig",
    "SimilarEpisode",
    # Procedural
    "ProceduralMemory",
    "ProceduralMemoryConfig",
    "ProcedureMatch",
    "TriggerType",
    "ActionType",
    # Semantic
    "Fact",
    "FactCategory",
    "SemanticMemory",
    "SemanticMemoryConfig",
]
