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


class MasterAgentMemoryAdapter:
    """Adapter to connect memory systems to MasterAgent.

    This adapter implements the MemoryStore protocol expected by MasterAgent,
    bridging to the actual EpisodicMemory implementation.

    Usage:
        store = MemoryStore()
        await store.initialize()
        episodic = EpisodicMemory(store)
        adapter = MasterAgentMemoryAdapter(episodic)
        # Pass adapter to MasterAgent as memory parameter
    """

    def __init__(self, episodic_memory: EpisodicMemory):
        """Initialize the adapter.

        Args:
            episodic_memory: The EpisodicMemory instance to use
        """
        self.episodic = episodic_memory

    async def record_episode(self, episode) -> None:
        """Record an episode for future learning.

        Converts the MasterAgent Episode to memory Episode format.

        Args:
            episode: Episode from master_agent.Episode
        """
        # The episode from MasterAgent has the same structure
        memory_episode = Episode(
            timestamp=episode.timestamp,
            session_id=episode.session_id,
            event_type=episode.event_type,
            context=episode.context,
            action_taken=episode.action_taken,
            action_reason=episode.action_reason,
            outcome=episode.outcome,
            confidence=episode.confidence,
            metadata=episode.metadata,
        )
        await self.episodic.record(memory_episode)

    async def find_similar(self, context: str, top_k: int = 3):
        """Find similar past episodes.

        Args:
            context: The context to search for
            top_k: Number of results to return

        Returns:
            List of similar episodes
        """
        similar = await self.episodic.find_similar(context, top_k=top_k)
        # Convert to format expected by MasterAgent
        return [
            Episode(
                timestamp=s.episode.timestamp,
                session_id=s.episode.session_id,
                event_type=s.episode.event_type,
                context=s.episode.context,
                action_taken=s.episode.action_taken,
                action_reason=s.episode.action_reason,
                outcome=s.episode.outcome,
                confidence=s.episode.confidence,
                metadata=s.episode.metadata,
            )
            for s in similar
        ]


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
    # Adapter
    "MasterAgentMemoryAdapter",
]
