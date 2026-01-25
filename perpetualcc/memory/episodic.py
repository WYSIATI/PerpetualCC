"""Episodic memory - records what happened and how we responded.

Episodic memory stores individual events/episodes with their context, actions taken,
and outcomes. It supports similarity search via embeddings to find relevant past
experiences when making decisions.

This enables the system to:
- Learn from past decisions (what worked, what didn't)
- Find similar past situations to inform current decisions
- Build confidence based on past success patterns
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from perpetualcc.memory.store import MemoryStore, StoredEpisode

if TYPE_CHECKING:
    from perpetualcc.knowledge.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """An episode representing an event and response.

    This is the primary data structure for recording experiences.
    It matches the Episode dataclass defined in master_agent.py.

    Attributes:
        timestamp: When this episode occurred
        session_id: Session that generated this episode
        event_type: Type of event (permission_request, question, error, etc.)
        context: Summary of what was happening
        action_taken: What action was taken (approve_tool, deny_tool, answer, escalate)
        action_reason: Why this action was taken
        outcome: Result (success/failure/pending)
        confidence: Confidence level of the action (0.0-1.0)
        metadata: Additional structured data (tool_name, question, etc.)
        embedding: Vector embedding for similarity search (optional)
    """

    timestamp: datetime
    session_id: str
    event_type: str
    context: str
    action_taken: str
    action_reason: str
    outcome: str = "pending"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SimilarEpisode:
    """An episode with similarity score.

    Attributes:
        episode: The stored episode
        similarity: Cosine similarity score (0.0-1.0, higher is more similar)
    """

    episode: StoredEpisode
    similarity: float


@dataclass(frozen=True)
class EpisodicMemoryConfig:
    """Configuration for episodic memory.

    Attributes:
        max_episodes: Maximum episodes to keep (oldest are pruned)
        similarity_threshold: Minimum similarity for search results
        default_top_k: Default number of results for similarity search
        auto_embed: Whether to automatically embed episodes on record
    """

    max_episodes: int = 10000
    similarity_threshold: float = 0.5
    default_top_k: int = 5
    auto_embed: bool = True


class EpisodicMemory:
    """Episodic memory system for storing and retrieving experiences.

    This class provides high-level operations on the episode storage,
    including similarity search using embeddings.

    Usage:
        memory = EpisodicMemory(store, embedding_provider)

        # Record an episode
        await memory.record(episode)

        # Find similar past episodes
        similar = await memory.find_similar("Tool use request: Write", top_k=5)

        # Get recent episodes
        recent = await memory.get_recent(session_id="abc123", limit=10)
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        config: EpisodicMemoryConfig | None = None,
    ):
        """Initialize episodic memory.

        Args:
            store: The underlying memory store
            embedding_provider: Optional provider for generating embeddings
            config: Optional configuration
        """
        self.store = store
        self.embedding_provider = embedding_provider
        self.config = config or EpisodicMemoryConfig()

    async def record(self, episode: Episode) -> int:
        """Record a new episode.

        This stores the episode in the database and optionally generates
        an embedding for similarity search.

        Args:
            episode: The episode to record

        Returns:
            The ID of the stored episode
        """
        # Generate embedding if provider available and auto_embed enabled
        embedding = episode.embedding
        if embedding is None and self.embedding_provider and self.config.auto_embed:
            try:
                # Create text representation for embedding
                text = self._episode_to_text(episode)
                embedding = await self.embedding_provider.embed(text)
            except Exception as e:
                logger.warning("Failed to generate episode embedding: %s", e)

        # Store in database
        episode_id = await self.store.insert_episode(
            session_id=episode.session_id,
            timestamp=episode.timestamp,
            event_type=episode.event_type,
            context=episode.context,
            action_taken=episode.action_taken,
            action_reason=episode.action_reason,
            outcome=episode.outcome,
            confidence=episode.confidence,
            metadata=episode.metadata,
            embedding=embedding,
        )

        logger.debug(
            "Recorded episode %d: type=%s, action=%s",
            episode_id,
            episode.event_type,
            episode.action_taken,
        )

        return episode_id

    async def record_from_master_agent(
        self,
        timestamp: datetime,
        session_id: str,
        event_type: str,
        context: str,
        action_taken: str,
        action_reason: str,
        outcome: str = "pending",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Record an episode from MasterAgent data.

        This is a convenience method that matches the Episode interface
        used by MasterAgent._learn().

        Args:
            timestamp: When the episode occurred
            session_id: Session ID
            event_type: Type of event
            context: Context summary
            action_taken: Action that was taken
            action_reason: Reason for the action
            outcome: Result of the action
            confidence: Confidence level
            metadata: Additional data

        Returns:
            The ID of the stored episode
        """
        episode = Episode(
            timestamp=timestamp,
            session_id=session_id,
            event_type=event_type,
            context=context,
            action_taken=action_taken,
            action_reason=action_reason,
            outcome=outcome,
            confidence=confidence,
            metadata=metadata or {},
        )
        return await self.record(episode)

    async def find_similar(
        self,
        context: str,
        top_k: int | None = None,
        event_type: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SimilarEpisode]:
        """Find episodes similar to a given context.

        This performs semantic similarity search using embeddings.
        Falls back to keyword matching if no embedding provider.

        Args:
            context: The context to search for
            top_k: Number of results to return
            event_type: Optional filter by event type
            min_similarity: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar episodes with similarity scores
        """
        top_k = top_k or self.config.default_top_k
        min_similarity = min_similarity or self.config.similarity_threshold

        if self.embedding_provider:
            return await self._find_similar_by_embedding(context, top_k, event_type, min_similarity)
        else:
            return await self._find_similar_by_keywords(context, top_k, event_type)

    async def _find_similar_by_embedding(
        self,
        context: str,
        top_k: int,
        event_type: str | None,
        min_similarity: float,
    ) -> list[SimilarEpisode]:
        """Find similar episodes using embedding similarity."""
        # Generate query embedding
        try:
            query_embedding = await self.embedding_provider.embed(context)
        except Exception as e:
            logger.warning("Failed to generate query embedding: %s", e)
            return await self._find_similar_by_keywords(context, top_k, event_type)

        # Get all episodes with embeddings (could be optimized with vector index)
        episodes = await self.store.query_episodes(
            event_type=event_type,
            limit=1000,  # Reasonable limit for in-memory search
        )

        # Filter to episodes with embeddings and compute similarity
        results = []
        for episode in episodes:
            if episode.embedding:
                similarity = self._cosine_similarity(query_embedding, episode.embedding)
                if similarity >= min_similarity:
                    results.append(SimilarEpisode(episode=episode, similarity=similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    async def _find_similar_by_keywords(
        self,
        context: str,
        top_k: int,
        event_type: str | None,
    ) -> list[SimilarEpisode]:
        """Find similar episodes using simple keyword matching."""
        # Get episodes
        episodes = await self.store.query_episodes(
            event_type=event_type,
            limit=500,
        )

        # Simple keyword matching
        keywords = set(context.lower().split())
        results = []

        for episode in episodes:
            episode_keywords = set(episode.context.lower().split())
            overlap = len(keywords & episode_keywords)
            if overlap > 0:
                # Simple Jaccard-like similarity
                similarity = overlap / max(len(keywords | episode_keywords), 1)
                results.append(SimilarEpisode(episode=episode, similarity=similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get_recent(
        self,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[StoredEpisode]:
        """Get recent episodes.

        Args:
            limit: Maximum number of episodes to return
            session_id: Optional filter by session

        Returns:
            List of recent episodes
        """
        return await self.store.get_recent_episodes(limit=limit, session_id=session_id)

    async def get_by_event_type(self, event_type: str, limit: int = 50) -> list[StoredEpisode]:
        """Get episodes by event type.

        Args:
            event_type: Type of event to filter by
            limit: Maximum number of results

        Returns:
            List of matching episodes
        """
        return await self.store.query_episodes(event_type=event_type, limit=limit)

    async def get_by_outcome(self, outcome: str, limit: int = 50) -> list[StoredEpisode]:
        """Get episodes by outcome.

        Args:
            outcome: Outcome to filter by (success/failure/pending)
            limit: Maximum number of results

        Returns:
            List of matching episodes
        """
        return await self.store.query_episodes(outcome=outcome, limit=limit)

    async def update_outcome(self, episode_id: int, outcome: str) -> bool:
        """Update the outcome of an episode.

        This is called when we learn whether a past decision was successful.

        Args:
            episode_id: The episode ID
            outcome: New outcome (success/failure)

        Returns:
            True if updated, False if episode not found
        """
        return await self.store.update_episode_outcome(episode_id, outcome)

    async def get_success_rate(
        self,
        event_type: str | None = None,
        action_taken: str | None = None,
    ) -> dict[str, Any]:
        """Calculate success rate for actions.

        Args:
            event_type: Optional filter by event type
            action_taken: Optional filter by action type

        Returns:
            Dictionary with success rate statistics
        """
        episodes = await self.store.query_episodes(event_type=event_type, limit=10000)

        if action_taken:
            episodes = [e for e in episodes if e.action_taken == action_taken]

        total = len(episodes)
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "failure": 0,
                "pending": 0,
                "success_rate": 0.0,
            }

        success = sum(1 for e in episodes if e.outcome == "success")
        failure = sum(1 for e in episodes if e.outcome == "failure")
        pending = sum(1 for e in episodes if e.outcome == "pending")

        return {
            "total": total,
            "success": success,
            "failure": failure,
            "pending": pending,
            "success_rate": success / total if total > 0 else 0.0,
        }

    async def add_embedding(self, episode_id: int) -> bool:
        """Generate and add embedding to an existing episode.

        Args:
            episode_id: The episode ID

        Returns:
            True if embedding was added, False otherwise
        """
        if not self.embedding_provider:
            return False

        episode = await self.store.get_episode_by_id(episode_id)
        if not episode or episode.embedding:
            return False

        try:
            text = self._stored_episode_to_text(episode)
            embedding = await self.embedding_provider.embed(text)
            return await self.store.update_episode_embedding(episode_id, embedding)
        except Exception as e:
            logger.warning("Failed to add embedding to episode %d: %s", episode_id, e)
            return False

    async def backfill_embeddings(self, batch_size: int = 100) -> int:
        """Backfill embeddings for episodes that don't have them.

        Args:
            batch_size: Number of episodes to process at once

        Returns:
            Number of episodes updated
        """
        if not self.embedding_provider:
            return 0

        updated = 0
        offset = 0

        while True:
            # Get episodes without embeddings
            episodes = await self.store.query_episodes(limit=batch_size, offset=offset)

            if not episodes:
                break

            for episode in episodes:
                if episode.embedding is None:
                    if await self.add_embedding(episode.id):
                        updated += 1

            offset += batch_size
            if len(episodes) < batch_size:
                break

        logger.info("Backfilled embeddings for %d episodes", updated)
        return updated

    def _episode_to_text(self, episode: Episode) -> str:
        """Convert an episode to text for embedding."""
        parts = [
            f"Event: {episode.event_type}",
            f"Context: {episode.context}",
            f"Action: {episode.action_taken}",
            f"Reason: {episode.action_reason}",
        ]

        # Add relevant metadata
        if episode.metadata:
            if "tool_name" in episode.metadata:
                parts.append(f"Tool: {episode.metadata['tool_name']}")
            if "question" in episode.metadata:
                parts.append(f"Question: {episode.metadata['question']}")

        return " | ".join(parts)

    def _stored_episode_to_text(self, episode: StoredEpisode) -> str:
        """Convert a stored episode to text for embedding."""
        parts = [
            f"Event: {episode.event_type}",
            f"Context: {episode.context}",
            f"Action: {episode.action_taken}",
            f"Reason: {episode.action_reason}",
        ]

        if episode.metadata:
            if "tool_name" in episode.metadata:
                parts.append(f"Tool: {episode.metadata['tool_name']}")
            if "question" in episode.metadata:
                parts.append(f"Question: {episode.metadata['question']}")

        return " | ".join(parts)
