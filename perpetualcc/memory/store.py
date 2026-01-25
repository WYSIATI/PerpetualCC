"""SQLite persistence layer for memory systems.

This module provides the base storage infrastructure using SQLite with async support.
All memory subsystems (episodic, procedural, semantic) share this common persistence layer.

The database is stored at ~/.perpetualcc/data/perpetualcc.db by default.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DATA_DIR = Path.home() / ".perpetualcc" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "perpetualcc.db"

# Schema version for migrations
SCHEMA_VERSION = 1

# SQL schema definitions
SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Episodes: what happened and how we responded
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,
    context TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    action_reason TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    embedding BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for episodes
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_event_type ON episodes(event_type);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);

-- Procedures: learned rules with confidence
CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_used_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for procedures
CREATE INDEX IF NOT EXISTS idx_procedures_trigger_type ON procedures(trigger_type);
CREATE INDEX IF NOT EXISTS idx_procedures_confidence ON procedures(confidence);

-- Unique constraint on trigger_type + trigger_pattern
CREATE UNIQUE INDEX IF NOT EXISTS idx_procedures_unique_trigger
    ON procedures(trigger_type, trigger_pattern);

-- Semantic facts: project knowledge
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL,
    category TEXT NOT NULL,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for facts
CREATE INDEX IF NOT EXISTS idx_facts_project_path ON facts(project_path);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_project_category ON facts(project_path, category);
"""


@dataclass(frozen=True)
class MemoryStoreConfig:
    """Configuration for the memory store.

    Attributes:
        db_path: Path to the SQLite database file
        pool_size: Number of connections to maintain (for future use)
        busy_timeout: Timeout in milliseconds for busy connections
        journal_mode: SQLite journal mode (WAL recommended)
    """

    db_path: Path = DEFAULT_DB_PATH
    pool_size: int = 5
    busy_timeout: int = 5000
    journal_mode: str = "WAL"


@dataclass
class StoredEpisode:
    """An episode retrieved from storage.

    Attributes:
        id: Database row ID
        session_id: Session that generated this episode
        timestamp: When the episode occurred
        event_type: Type of event (permission_request, question, etc.)
        context: Summary of what was happening
        action_taken: What action was taken
        action_reason: Why this action was taken
        outcome: Result (success/failure/pending)
        confidence: Confidence level of the action (0.0-1.0)
        metadata: Additional structured data
        embedding: Vector embedding for similarity search
        created_at: When this record was created
    """

    id: int
    session_id: str
    timestamp: datetime
    event_type: str
    context: str
    action_taken: str
    action_reason: str
    outcome: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime | None = None


@dataclass
class StoredProcedure:
    """A learned procedure retrieved from storage.

    Attributes:
        id: Database row ID
        trigger_type: Type of trigger (tool, question, etc.)
        trigger_pattern: Pattern to match (regex or glob)
        action: What to do when triggered
        confidence: Current confidence level (0.1-0.99)
        success_count: Number of successful applications
        failure_count: Number of failed applications
        last_used_at: When this procedure was last applied
        created_at: When this procedure was created
        updated_at: When this procedure was last modified
    """

    id: int
    trigger_type: str
    trigger_pattern: str
    action: str
    confidence: float
    success_count: int = 0
    failure_count: int = 0
    last_used_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class StoredFact:
    """A semantic fact retrieved from storage.

    Attributes:
        id: Database row ID
        project_path: Path to the project this fact belongs to
        category: Fact category (convention, architecture, decision, preference)
        fact: The fact content
        source: Where this fact was learned from
        confidence: Confidence level (0.0-1.0)
        created_at: When this fact was created
        updated_at: When this fact was last modified
    """

    id: int
    project_path: str
    category: str
    fact: str
    source: str | None = None
    confidence: float = 1.0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryStore:
    """SQLite-based persistence for memory systems.

    This class provides async database operations for storing and retrieving
    episodes, procedures, and facts. It handles schema migrations and
    connection management.

    Usage:
        store = MemoryStore()
        await store.initialize()

        # Store an episode
        episode_id = await store.insert_episode(...)

        # Query episodes
        episodes = await store.query_episodes(session_id="abc123")

        # Close when done
        await store.close()
    """

    def __init__(self, config: MemoryStoreConfig | None = None):
        """Initialize the memory store.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or MemoryStoreConfig()
        self._connection = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and schema.

        This must be called before using the store. It:
        1. Creates the data directory if needed
        2. Opens the database connection
        3. Applies schema migrations
        """
        if self._initialized:
            return

        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite is required for memory systems. Install with: pip install aiosqlite"
            ) from e

        # Ensure data directory exists
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._connection = await aiosqlite.connect(
            self.config.db_path,
            timeout=self.config.busy_timeout / 1000.0,
        )

        # Configure connection
        await self._connection.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        await self._connection.execute(f"PRAGMA busy_timeout={self.config.busy_timeout}")
        await self._connection.execute("PRAGMA foreign_keys=ON")

        # Row factory for dict-like access
        self._connection.row_factory = aiosqlite.Row

        # Apply schema
        await self._apply_schema()
        self._initialized = True

        logger.info("Memory store initialized: %s", self.config.db_path)

    async def _apply_schema(self) -> None:
        """Apply database schema and migrations."""
        # Check current version
        try:
            async with self._connection.execute(
                "SELECT MAX(version) FROM schema_version"
            ) as cursor:
                row = await cursor.fetchone()
                current_version = row[0] if row and row[0] else 0
        except Exception:
            # Table doesn't exist yet
            current_version = 0

        if current_version < SCHEMA_VERSION:
            # Apply schema
            await self._connection.executescript(SCHEMA_SQL)

            # Record schema version
            await self._connection.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await self._connection.commit()
            logger.info("Applied schema version %d", SCHEMA_VERSION)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False
            logger.info("Memory store closed")

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for database transactions.

        Usage:
            async with store.transaction():
                await store.insert_episode(...)
                await store.insert_episode(...)
                # Both inserts commit together or roll back together
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized. Call initialize() first.")

        try:
            yield
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    # Episode operations

    async def insert_episode(
        self,
        session_id: str,
        timestamp: datetime,
        event_type: str,
        context: str,
        action_taken: str,
        action_reason: str,
        outcome: str = "pending",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert a new episode.

        Args:
            session_id: Session that generated this episode
            timestamp: When the episode occurred
            event_type: Type of event
            context: Summary of what was happening
            action_taken: What action was taken
            action_reason: Why this action was taken
            outcome: Result (success/failure/pending)
            confidence: Confidence level (0.0-1.0)
            metadata: Additional structured data
            embedding: Vector embedding for similarity search

        Returns:
            The ID of the inserted episode
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        metadata_json = json.dumps(metadata or {})
        embedding_blob = bytes(json.dumps(embedding), "utf-8") if embedding else None

        async with self._connection.execute(
            """
            INSERT INTO episodes (
                session_id, timestamp, event_type, context, action_taken,
                action_reason, outcome, confidence, metadata, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                timestamp.isoformat(),
                event_type,
                context,
                action_taken,
                action_reason,
                outcome,
                confidence,
                metadata_json,
                embedding_blob,
            ),
        ) as cursor:
            await self._connection.commit()
            return cursor.lastrowid

    async def query_episodes(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredEpisode]:
        """Query episodes with optional filters.

        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            outcome: Filter by outcome
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching episodes
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        conditions = []
        params = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if outcome:
            conditions.append("outcome = ?")
            params.append(outcome)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        async with self._connection.execute(
            f"""
            SELECT id, session_id, timestamp, event_type, context, action_taken,
                   action_reason, outcome, confidence, metadata, embedding, created_at
            FROM episodes
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_episode(row) for row in rows]

    async def get_episode_by_id(self, episode_id: int) -> StoredEpisode | None:
        """Get a specific episode by ID.

        Args:
            episode_id: The episode ID

        Returns:
            The episode if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        async with self._connection.execute(
            """
            SELECT id, session_id, timestamp, event_type, context, action_taken,
                   action_reason, outcome, confidence, metadata, embedding, created_at
            FROM episodes
            WHERE id = ?
            """,
            (episode_id,),
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_episode(row) if row else None

    async def update_episode_outcome(self, episode_id: int, outcome: str) -> bool:
        """Update the outcome of an episode.

        Args:
            episode_id: The episode ID
            outcome: New outcome value

        Returns:
            True if updated, False if episode not found
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        async with self._connection.execute(
            "UPDATE episodes SET outcome = ? WHERE id = ?",
            (outcome, episode_id),
        ) as cursor:
            await self._connection.commit()
            return cursor.rowcount > 0

    async def update_episode_embedding(self, episode_id: int, embedding: list[float]) -> bool:
        """Update the embedding of an episode.

        Args:
            episode_id: The episode ID
            embedding: New embedding vector

        Returns:
            True if updated, False if episode not found
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        embedding_blob = bytes(json.dumps(embedding), "utf-8")

        async with self._connection.execute(
            "UPDATE episodes SET embedding = ? WHERE id = ?",
            (embedding_blob, episode_id),
        ) as cursor:
            await self._connection.commit()
            return cursor.rowcount > 0

    async def get_recent_episodes(
        self, limit: int = 20, session_id: str | None = None
    ) -> list[StoredEpisode]:
        """Get the most recent episodes.

        Args:
            limit: Maximum number of results
            session_id: Optional filter by session

        Returns:
            List of recent episodes
        """
        return await self.query_episodes(session_id=session_id, limit=limit, offset=0)

    def _row_to_episode(self, row) -> StoredEpisode:
        """Convert a database row to a StoredEpisode."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"].decode("utf-8"))

        return StoredEpisode(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=row["event_type"],
            context=row["context"],
            action_taken=row["action_taken"],
            action_reason=row["action_reason"],
            outcome=row["outcome"],
            confidence=row["confidence"],
            metadata=metadata,
            embedding=embedding,
            created_at=(datetime.fromisoformat(row["created_at"]) if row["created_at"] else None),
        )

    # Procedure operations

    async def insert_procedure(
        self,
        trigger_type: str,
        trigger_pattern: str,
        action: str,
        confidence: float = 0.5,
    ) -> int:
        """Insert a new procedure.

        Args:
            trigger_type: Type of trigger (tool, question, etc.)
            trigger_pattern: Pattern to match
            action: What to do when triggered
            confidence: Initial confidence (0.1-0.99)

        Returns:
            The ID of the inserted procedure
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        now = datetime.now().isoformat()

        async with self._connection.execute(
            """
            INSERT INTO procedures (
                trigger_type, trigger_pattern, action, confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trigger_type, trigger_pattern, action, confidence, now, now),
        ) as cursor:
            await self._connection.commit()
            return cursor.lastrowid

    async def get_procedure(
        self, trigger_type: str, trigger_pattern: str
    ) -> StoredProcedure | None:
        """Get a procedure by its trigger.

        Args:
            trigger_type: Type of trigger
            trigger_pattern: Pattern to match

        Returns:
            The procedure if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        async with self._connection.execute(
            """
            SELECT id, trigger_type, trigger_pattern, action, confidence,
                   success_count, failure_count, last_used_at, created_at, updated_at
            FROM procedures
            WHERE trigger_type = ? AND trigger_pattern = ?
            """,
            (trigger_type, trigger_pattern),
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_procedure(row) if row else None

    async def query_procedures(
        self,
        trigger_type: str | None = None,
        min_confidence: float | None = None,
        limit: int = 100,
    ) -> list[StoredProcedure]:
        """Query procedures with optional filters.

        Args:
            trigger_type: Filter by trigger type
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            List of matching procedures
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        conditions = []
        params = []

        if trigger_type:
            conditions.append("trigger_type = ?")
            params.append(trigger_type)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        async with self._connection.execute(
            f"""
            SELECT id, trigger_type, trigger_pattern, action, confidence,
                   success_count, failure_count, last_used_at, created_at, updated_at
            FROM procedures
            WHERE {where_clause}
            ORDER BY confidence DESC
            LIMIT ?
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_procedure(row) for row in rows]

    async def update_procedure_outcome(
        self, procedure_id: int, success: bool
    ) -> StoredProcedure | None:
        """Update a procedure's confidence based on outcome.

        Success increases confidence by 0.05 (capped at 0.99).
        Failure decreases confidence by 0.1 (floor at 0.1).

        Args:
            procedure_id: The procedure ID
            success: Whether the procedure application was successful

        Returns:
            The updated procedure, or None if not found
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        now = datetime.now().isoformat()

        if success:
            # Increase confidence, cap at 0.99
            await self._connection.execute(
                """
                UPDATE procedures
                SET confidence = MIN(0.99, confidence + 0.05),
                    success_count = success_count + 1,
                    last_used_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, procedure_id),
            )
        else:
            # Decrease confidence, floor at 0.1
            await self._connection.execute(
                """
                UPDATE procedures
                SET confidence = MAX(0.1, confidence - 0.1),
                    failure_count = failure_count + 1,
                    last_used_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, procedure_id),
            )

        await self._connection.commit()

        # Return the updated procedure
        async with self._connection.execute(
            """
            SELECT id, trigger_type, trigger_pattern, action, confidence,
                   success_count, failure_count, last_used_at, created_at, updated_at
            FROM procedures
            WHERE id = ?
            """,
            (procedure_id,),
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_procedure(row) if row else None

    async def upsert_procedure(
        self,
        trigger_type: str,
        trigger_pattern: str,
        action: str,
        confidence: float = 0.5,
    ) -> int:
        """Insert or update a procedure.

        If a procedure with the same trigger_type and trigger_pattern exists,
        updates its action and confidence. Otherwise, inserts a new procedure.

        Args:
            trigger_type: Type of trigger
            trigger_pattern: Pattern to match
            action: What to do when triggered
            confidence: Confidence level

        Returns:
            The procedure ID
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        now = datetime.now().isoformat()

        await self._connection.execute(
            """
            INSERT INTO procedures (trigger_type, trigger_pattern, action, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(trigger_type, trigger_pattern)
            DO UPDATE SET action = excluded.action,
                          confidence = excluded.confidence,
                          updated_at = excluded.updated_at
            """,
            (trigger_type, trigger_pattern, action, confidence, now, now),
        )
        await self._connection.commit()

        # Get the ID
        async with self._connection.execute(
            "SELECT id FROM procedures WHERE trigger_type = ? AND trigger_pattern = ?",
            (trigger_type, trigger_pattern),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0]

    def _row_to_procedure(self, row) -> StoredProcedure:
        """Convert a database row to a StoredProcedure."""
        return StoredProcedure(
            id=row["id"],
            trigger_type=row["trigger_type"],
            trigger_pattern=row["trigger_pattern"],
            action=row["action"],
            confidence=row["confidence"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            last_used_at=(
                datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
            ),
            created_at=(datetime.fromisoformat(row["created_at"]) if row["created_at"] else None),
            updated_at=(datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None),
        )

    # Fact operations

    async def insert_fact(
        self,
        project_path: str,
        category: str,
        fact: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Insert a new fact.

        Args:
            project_path: Path to the project
            category: Fact category (convention, architecture, decision, preference)
            fact: The fact content
            source: Where this fact was learned from
            confidence: Confidence level (0.0-1.0)

        Returns:
            The ID of the inserted fact
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        now = datetime.now().isoformat()

        async with self._connection.execute(
            """
            INSERT INTO facts (project_path, category, fact, source, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project_path, category, fact, source, confidence, now, now),
        ) as cursor:
            await self._connection.commit()
            return cursor.lastrowid

    async def query_facts(
        self,
        project_path: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[StoredFact]:
        """Query facts with optional filters.

        Args:
            project_path: Filter by project path
            category: Filter by category
            limit: Maximum number of results

        Returns:
            List of matching facts
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        conditions = []
        params = []

        if project_path:
            conditions.append("project_path = ?")
            params.append(project_path)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        async with self._connection.execute(
            f"""
            SELECT id, project_path, category, fact, source, confidence, created_at, updated_at
            FROM facts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_fact(row) for row in rows]

    async def get_fact_by_id(self, fact_id: int) -> StoredFact | None:
        """Get a specific fact by ID.

        Args:
            fact_id: The fact ID

        Returns:
            The fact if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        async with self._connection.execute(
            """
            SELECT id, project_path, category, fact, source, confidence, created_at, updated_at
            FROM facts
            WHERE id = ?
            """,
            (fact_id,),
        ) as cursor:
            row = await cursor.fetchone()

        return self._row_to_fact(row) if row else None

    async def update_fact(self, fact_id: int, fact: str, confidence: float | None = None) -> bool:
        """Update a fact's content and optionally confidence.

        Args:
            fact_id: The fact ID
            fact: New fact content
            confidence: New confidence level (optional)

        Returns:
            True if updated, False if fact not found
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        now = datetime.now().isoformat()

        if confidence is not None:
            async with self._connection.execute(
                "UPDATE facts SET fact = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (fact, confidence, now, fact_id),
            ) as cursor:
                await self._connection.commit()
                return cursor.rowcount > 0
        else:
            async with self._connection.execute(
                "UPDATE facts SET fact = ?, updated_at = ? WHERE id = ?",
                (fact, now, fact_id),
            ) as cursor:
                await self._connection.commit()
                return cursor.rowcount > 0

    async def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact.

        Args:
            fact_id: The fact ID

        Returns:
            True if deleted, False if fact not found
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        async with self._connection.execute("DELETE FROM facts WHERE id = ?", (fact_id,)) as cursor:
            await self._connection.commit()
            return cursor.rowcount > 0

    def _row_to_fact(self, row) -> StoredFact:
        """Convert a database row to a StoredFact."""
        return StoredFact(
            id=row["id"],
            project_path=row["project_path"],
            category=row["category"],
            fact=row["fact"],
            source=row["source"],
            confidence=row["confidence"],
            created_at=(datetime.fromisoformat(row["created_at"]) if row["created_at"] else None),
            updated_at=(datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None),
        )

    # Utility methods

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the memory store.

        Returns:
            Dictionary with counts and other statistics
        """
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        stats = {}

        # Episode counts
        async with self._connection.execute("SELECT COUNT(*) FROM episodes") as cursor:
            stats["total_episodes"] = (await cursor.fetchone())[0]

        async with self._connection.execute(
            "SELECT outcome, COUNT(*) FROM episodes GROUP BY outcome"
        ) as cursor:
            stats["episodes_by_outcome"] = {row[0]: row[1] for row in await cursor.fetchall()}

        # Procedure counts
        async with self._connection.execute("SELECT COUNT(*) FROM procedures") as cursor:
            stats["total_procedures"] = (await cursor.fetchone())[0]

        async with self._connection.execute("SELECT AVG(confidence) FROM procedures") as cursor:
            avg = (await cursor.fetchone())[0]
            stats["avg_procedure_confidence"] = round(avg, 3) if avg else 0.0

        # Fact counts
        async with self._connection.execute("SELECT COUNT(*) FROM facts") as cursor:
            stats["total_facts"] = (await cursor.fetchone())[0]

        async with self._connection.execute(
            "SELECT category, COUNT(*) FROM facts GROUP BY category"
        ) as cursor:
            stats["facts_by_category"] = {row[0]: row[1] for row in await cursor.fetchall()}

        return stats

    async def clear_all(self) -> None:
        """Clear all data from the store. Use with caution!"""
        if not self._initialized:
            raise RuntimeError("Memory store not initialized")

        await self._connection.execute("DELETE FROM episodes")
        await self._connection.execute("DELETE FROM procedures")
        await self._connection.execute("DELETE FROM facts")
        await self._connection.commit()
        logger.warning("Cleared all data from memory store")
