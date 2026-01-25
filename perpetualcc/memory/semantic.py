"""Semantic memory - project knowledge and facts.

Semantic memory stores learned facts about projects, including:
- Conventions (coding style, naming, patterns)
- Architecture (structure, dependencies, design decisions)
- Decisions (past architectural/design choices and rationale)
- Preferences (user/team preferences learned over time)

This enables the system to:
- Remember project-specific conventions
- Maintain consistency with past decisions
- Provide context-aware answers
- Avoid repeating questions about known facts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from perpetualcc.memory.store import MemoryStore, StoredFact

logger = logging.getLogger(__name__)


# Fact categories
class FactCategory:
    """Standard categories for facts."""

    CONVENTION = "convention"  # Coding style, naming, patterns
    ARCHITECTURE = "architecture"  # Structure, dependencies, design
    DECISION = "decision"  # Past choices and rationale
    PREFERENCE = "preference"  # User/team preferences
    DEPENDENCY = "dependency"  # Package/library information
    TOOL = "tool"  # Tool configurations and usage
    TESTING = "testing"  # Testing conventions and requirements


@dataclass
class Fact:
    """A semantic fact about a project.

    Attributes:
        project_path: Path to the project
        category: Fact category (convention, architecture, etc.)
        fact: The fact content
        source: Where this fact was learned from
        confidence: Confidence level (0.0-1.0)
    """

    project_path: str
    category: str
    fact: str
    source: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class SemanticMemoryConfig:
    """Configuration for semantic memory.

    Attributes:
        max_facts_per_project: Maximum facts to store per project
        min_confidence: Minimum confidence to keep a fact
        search_limit: Default limit for search queries
    """

    max_facts_per_project: int = 500
    min_confidence: float = 0.3
    search_limit: int = 50


class SemanticMemory:
    """Semantic memory system for project knowledge.

    This class manages project-specific facts that inform decision making.
    Facts are categorized and can be queried by project and category.

    Usage:
        memory = SemanticMemory(store)

        # Add a fact
        await memory.add_fact(
            project_path="/path/to/project",
            category="convention",
            fact="Uses pytest for testing with 80% coverage requirement",
            source="CLAUDE.md",
        )

        # Get facts about a project
        facts = await memory.get_facts("/path/to/project", category="testing")

        # Search facts
        results = await memory.search("testing framework", "/path/to/project")
    """

    def __init__(
        self,
        store: MemoryStore,
        config: SemanticMemoryConfig | None = None,
    ):
        """Initialize semantic memory.

        Args:
            store: The underlying memory store
            config: Optional configuration
        """
        self.store = store
        self.config = config or SemanticMemoryConfig()

    async def add_fact(
        self,
        project_path: str,
        category: str,
        fact: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Add a fact about a project.

        Args:
            project_path: Path to the project
            category: Fact category (use FactCategory constants)
            fact: The fact content
            source: Where this fact was learned from
            confidence: Confidence level (0.0-1.0)

        Returns:
            The ID of the stored fact
        """
        # Normalize project path
        project_path = self._normalize_path(project_path)

        # Check for duplicate or similar fact
        existing = await self._find_similar_fact(project_path, category, fact)
        if existing:
            # Update existing fact if new one has higher confidence
            if confidence > existing.confidence:
                await self.store.update_fact(existing.id, fact, confidence)
                logger.debug("Updated existing fact %d with higher confidence", existing.id)
            return existing.id

        # Add new fact
        fact_id = await self.store.insert_fact(
            project_path=project_path,
            category=category,
            fact=fact,
            source=source,
            confidence=confidence,
        )

        logger.debug(
            "Added fact %d for project %s: [%s] %s",
            fact_id,
            project_path,
            category,
            fact[:50],
        )

        return fact_id

    async def get_facts(
        self,
        project_path: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[StoredFact]:
        """Get facts about a project.

        Args:
            project_path: Path to the project
            category: Optional category filter
            limit: Maximum number of results

        Returns:
            List of matching facts
        """
        project_path = self._normalize_path(project_path)
        limit = limit or self.config.search_limit

        return await self.store.query_facts(
            project_path=project_path,
            category=category,
            limit=limit,
        )

    async def get_conventions(self, project_path: str) -> list[StoredFact]:
        """Get coding conventions for a project.

        Args:
            project_path: Path to the project

        Returns:
            List of convention facts
        """
        return await self.get_facts(project_path, FactCategory.CONVENTION)

    async def get_architecture(self, project_path: str) -> list[StoredFact]:
        """Get architecture facts for a project.

        Args:
            project_path: Path to the project

        Returns:
            List of architecture facts
        """
        return await self.get_facts(project_path, FactCategory.ARCHITECTURE)

    async def get_decisions(self, project_path: str) -> list[StoredFact]:
        """Get past decisions for a project.

        Args:
            project_path: Path to the project

        Returns:
            List of decision facts
        """
        return await self.get_facts(project_path, FactCategory.DECISION)

    async def get_preferences(self, project_path: str) -> list[StoredFact]:
        """Get preferences for a project.

        Args:
            project_path: Path to the project

        Returns:
            List of preference facts
        """
        return await self.get_facts(project_path, FactCategory.PREFERENCE)

    async def search(
        self,
        query: str,
        project_path: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[StoredFact]:
        """Search facts by keyword.

        This performs a simple keyword search on fact content.
        For semantic search, use with embeddings (future enhancement).

        Args:
            query: Search query
            project_path: Optional project path filter
            category: Optional category filter
            limit: Maximum number of results

        Returns:
            List of matching facts
        """
        project_path = self._normalize_path(project_path) if project_path else None
        limit = limit or self.config.search_limit

        # Get all matching facts
        facts = await self.store.query_facts(
            project_path=project_path,
            category=category,
            limit=1000,  # Get more for filtering
        )

        # Filter by query keywords
        query_words = set(query.lower().split())
        results = []

        for fact in facts:
            fact_words = set(fact.fact.lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                # Score based on overlap
                score = overlap / max(len(query_words), 1)
                results.append((fact, score))

        # Sort by score and return
        results.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in results[:limit]]

    async def update_fact(
        self,
        fact_id: int,
        fact: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """Update a fact.

        Args:
            fact_id: The fact ID
            fact: New fact content (optional)
            confidence: New confidence level (optional)

        Returns:
            True if updated, False if not found
        """
        if fact is not None:
            return await self.store.update_fact(fact_id, fact, confidence)
        return False

    async def remove_fact(self, fact_id: int) -> bool:
        """Remove a fact.

        Args:
            fact_id: The fact ID

        Returns:
            True if removed, False if not found
        """
        return await self.store.delete_fact(fact_id)

    async def learn_from_answer(
        self,
        project_path: str,
        question: str,
        answer: str,
        source: str = "user_answer",
    ) -> int | None:
        """Learn a fact from a user's answer to a question.

        This extracts a fact from the question/answer pair and stores it.

        Args:
            project_path: Path to the project
            question: The question that was asked
            answer: The answer provided
            source: Source of the fact

        Returns:
            Fact ID if a fact was learned, None otherwise
        """
        # Try to extract a meaningful fact
        fact_info = self._extract_fact_from_qa(question, answer)
        if not fact_info:
            return None

        category, fact = fact_info
        return await self.add_fact(
            project_path=project_path,
            category=category,
            fact=fact,
            source=source,
            confidence=0.8,  # User-provided facts have good confidence
        )

    async def learn_from_file(
        self,
        project_path: str,
        file_path: str,
        file_content: str,
    ) -> int:
        """Learn facts from a file (e.g., CLAUDE.md, README.md).

        This parses the file and extracts relevant facts.

        Args:
            project_path: Path to the project
            file_path: Path to the file being processed
            file_content: Content of the file

        Returns:
            Number of facts learned
        """
        facts_learned = 0
        source = file_path

        # Extract facts from structured content
        if file_path.endswith("CLAUDE.md") or file_path.endswith("claude.md"):
            facts = self._extract_facts_from_claude_md(file_content)
        elif file_path.endswith("README.md"):
            facts = self._extract_facts_from_readme(file_content)
        elif file_path.endswith("pyproject.toml"):
            facts = self._extract_facts_from_pyproject(file_content)
        elif file_path.endswith("package.json"):
            facts = self._extract_facts_from_package_json(file_content)
        else:
            return 0

        for category, fact in facts:
            await self.add_fact(
                project_path=project_path,
                category=category,
                fact=fact,
                source=source,
                confidence=1.0,  # Facts from project files have high confidence
            )
            facts_learned += 1

        if facts_learned > 0:
            logger.info(
                "Learned %d facts from %s for project %s",
                facts_learned,
                file_path,
                project_path,
            )

        return facts_learned

    async def get_context_for_question(self, project_path: str, question: str) -> list[StoredFact]:
        """Get relevant facts for answering a question.

        This searches for facts that might help answer the question.

        Args:
            project_path: Path to the project
            question: The question being asked

        Returns:
            List of relevant facts
        """
        # Search for facts related to the question
        facts = await self.search(question, project_path, limit=10)

        # Also get general project facts if we don't have enough
        if len(facts) < 5:
            all_facts = await self.get_facts(project_path, limit=20)
            seen_ids = {f.id for f in facts}
            for fact in all_facts:
                if fact.id not in seen_ids:
                    facts.append(fact)
                    if len(facts) >= 10:
                        break

        return facts

    async def get_statistics(self, project_path: str | None = None) -> dict[str, Any]:
        """Get statistics about semantic memory.

        Args:
            project_path: Optional project filter

        Returns:
            Dictionary with statistics
        """
        project_path = self._normalize_path(project_path) if project_path else None
        facts = await self.store.query_facts(project_path=project_path, limit=10000)

        if not facts:
            return {
                "total": 0,
                "by_category": {},
                "by_project": {},
                "avg_confidence": 0.0,
            }

        by_category: dict[str, int] = {}
        by_project: dict[str, int] = {}
        total_confidence = 0.0

        for fact in facts:
            by_category[fact.category] = by_category.get(fact.category, 0) + 1
            by_project[fact.project_path] = by_project.get(fact.project_path, 0) + 1
            total_confidence += fact.confidence

        return {
            "total": len(facts),
            "by_category": by_category,
            "by_project": by_project,
            "avg_confidence": total_confidence / len(facts),
        }

    async def _find_similar_fact(
        self, project_path: str, category: str, fact: str
    ) -> StoredFact | None:
        """Find an existing fact similar to the given one."""
        facts = await self.store.query_facts(
            project_path=project_path, category=category, limit=100
        )

        # Simple duplicate detection
        fact_lower = fact.lower().strip()
        for existing in facts:
            existing_lower = existing.fact.lower().strip()
            # Exact or near-exact match
            if fact_lower == existing_lower:
                return existing
            # High similarity (80% word overlap)
            fact_words = set(fact_lower.split())
            existing_words = set(existing_lower.split())
            overlap = len(fact_words & existing_words)
            total = len(fact_words | existing_words)
            if total > 0 and overlap / total > 0.8:
                return existing

        return None

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a project path for consistent storage."""
        # Remove trailing slashes and resolve
        import os

        return os.path.normpath(path)

    def _extract_fact_from_qa(self, question: str, answer: str) -> tuple[str, str] | None:
        """Extract a fact from a question/answer pair."""
        question_lower = question.lower()

        # Database questions
        if "database" in question_lower or "db" in question_lower:
            return (
                FactCategory.DECISION,
                f"Database choice: {answer}",
            )

        # Framework questions
        if "framework" in question_lower or "library" in question_lower:
            return (
                FactCategory.DECISION,
                f"Framework/library choice: {answer}",
            )

        # Testing questions
        if "test" in question_lower:
            return (
                FactCategory.TESTING,
                f"Testing approach: {answer}",
            )

        # Architecture questions
        if "architecture" in question_lower or "structure" in question_lower:
            return (
                FactCategory.ARCHITECTURE,
                f"Architecture decision: {answer}",
            )

        # Style questions
        if "style" in question_lower or "convention" in question_lower:
            return (
                FactCategory.CONVENTION,
                f"Code style: {answer}",
            )

        # Generic preference
        if "prefer" in question_lower or "should" in question_lower:
            return (
                FactCategory.PREFERENCE,
                f"{question.strip('?')} -> {answer}",
            )

        return None

    def _extract_facts_from_claude_md(self, content: str) -> list[tuple[str, str]]:
        """Extract facts from a CLAUDE.md file."""
        facts = []
        lines = content.split("\n")

        # Look for key sections
        current_section = None
        section_content = []

        for line in lines:
            if line.startswith("## "):
                # Save previous section
                if current_section and section_content:
                    fact = self._section_to_fact(current_section, section_content)
                    if fact:
                        facts.append(fact)
                # Start new section
                current_section = line[3:].strip()
                section_content = []
            elif current_section:
                section_content.append(line)

        # Save last section
        if current_section and section_content:
            fact = self._section_to_fact(current_section, section_content)
            if fact:
                facts.append(fact)

        return facts

    def _extract_facts_from_readme(self, content: str) -> list[tuple[str, str]]:
        """Extract facts from a README.md file."""
        facts = []

        # Look for project description
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("# "):
                # Project name
                project_name = line[2:].strip()
                facts.append((FactCategory.ARCHITECTURE, f"Project name: {project_name}"))
            elif "install" in line.lower() and "```" in content[content.find(line) :]:
                # Installation instructions
                facts.append((FactCategory.TOOL, "Has installation instructions in README"))
                break

        return facts

    def _extract_facts_from_pyproject(self, content: str) -> list[tuple[str, str]]:
        """Extract facts from a pyproject.toml file."""
        facts = []

        # Simple parsing without toml library
        if "python" in content:
            import re

            match = re.search(r'requires-python\s*=\s*"([^"]+)"', content)
            if match:
                facts.append((FactCategory.DEPENDENCY, f"Requires Python {match.group(1)}"))

        if "pytest" in content:
            facts.append((FactCategory.TESTING, "Uses pytest for testing"))

        if "ruff" in content:
            facts.append((FactCategory.TOOL, "Uses ruff for linting/formatting"))

        return facts

    def _extract_facts_from_package_json(self, content: str) -> list[tuple[str, str]]:
        """Extract facts from a package.json file."""
        facts = []

        try:
            import json

            data = json.loads(content)

            if "name" in data:
                facts.append((FactCategory.ARCHITECTURE, f"Package name: {data['name']}"))

            # Check for common tools/frameworks
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }

            if "react" in deps:
                facts.append((FactCategory.ARCHITECTURE, "Uses React"))
            if "vue" in deps:
                facts.append((FactCategory.ARCHITECTURE, "Uses Vue"))
            if "typescript" in deps:
                facts.append((FactCategory.TOOL, "Uses TypeScript"))
            if "jest" in deps:
                facts.append((FactCategory.TESTING, "Uses Jest for testing"))
            if "vitest" in deps:
                facts.append((FactCategory.TESTING, "Uses Vitest for testing"))
            if "eslint" in deps:
                facts.append((FactCategory.TOOL, "Uses ESLint"))
            if "prettier" in deps:
                facts.append((FactCategory.TOOL, "Uses Prettier"))

        except Exception:
            pass

        return facts

    def _section_to_fact(self, section: str, content: list[str]) -> tuple[str, str] | None:
        """Convert a CLAUDE.md section to a fact."""
        section_lower = section.lower()
        content_text = "\n".join(content[:5]).strip()  # First 5 lines

        if not content_text:
            return None

        if "overview" in section_lower or "about" in section_lower:
            return (FactCategory.ARCHITECTURE, f"Overview: {content_text[:200]}")

        if "testing" in section_lower or "test" in section_lower:
            return (FactCategory.TESTING, f"Testing: {content_text[:200]}")

        if "convention" in section_lower or "style" in section_lower:
            return (FactCategory.CONVENTION, f"Conventions: {content_text[:200]}")

        if "structure" in section_lower or "architecture" in section_lower:
            return (FactCategory.ARCHITECTURE, f"Structure: {content_text[:200]}")

        if "dependencies" in section_lower:
            return (FactCategory.DEPENDENCY, f"Dependencies: {content_text[:200]}")

        return None
