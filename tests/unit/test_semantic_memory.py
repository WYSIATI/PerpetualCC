"""Unit tests for semantic memory.

Tests cover:
- Fact storage and retrieval
- Fact categories
- Keyword search
- Learning from files and Q&A
- Real-world Claude Code scenarios
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from perpetualcc.memory.semantic import (
    Fact,
    FactCategory,
    SemanticMemory,
    SemanticMemoryConfig,
)
from perpetualcc.memory.store import MemoryStore, MemoryStoreConfig


class TestFactCategory:
    """Tests for FactCategory constants."""

    def test_categories_defined(self):
        """All expected categories should be defined."""
        assert FactCategory.CONVENTION == "convention"
        assert FactCategory.ARCHITECTURE == "architecture"
        assert FactCategory.DECISION == "decision"
        assert FactCategory.PREFERENCE == "preference"
        assert FactCategory.DEPENDENCY == "dependency"
        assert FactCategory.TOOL == "tool"
        assert FactCategory.TESTING == "testing"


class TestFact:
    """Tests for Fact dataclass."""

    def test_fact_basic(self):
        """Fact should store project knowledge."""
        fact = Fact(
            project_path="/path/to/project",
            category=FactCategory.CONVENTION,
            fact="Uses pytest for testing",
        )

        assert fact.project_path == "/path/to/project"
        assert fact.category == FactCategory.CONVENTION
        assert fact.fact == "Uses pytest for testing"
        assert fact.confidence == 1.0  # Default

    def test_fact_with_source(self):
        """Fact should store source information."""
        fact = Fact(
            project_path="/project",
            category=FactCategory.TOOL,
            fact="Uses ruff for linting",
            source="pyproject.toml",
            confidence=0.9,
        )

        assert fact.source == "pyproject.toml"
        assert fact.confidence == 0.9


class TestSemanticMemoryConfig:
    """Tests for SemanticMemoryConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = SemanticMemoryConfig()

        assert config.max_facts_per_project == 500
        assert config.min_confidence == 0.3
        assert config.search_limit == 50

    def test_custom_values(self):
        """Config should accept custom values."""
        config = SemanticMemoryConfig(
            max_facts_per_project=1000,
            search_limit=100,
        )

        assert config.max_facts_per_project == 1000
        assert config.search_limit == 100


class TestSemanticMemoryBasics:
    """Tests for basic fact operations."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def semantic_memory(self, store: MemoryStore) -> SemanticMemory:
        """Create semantic memory."""
        return SemanticMemory(store)

    @pytest.mark.asyncio
    async def test_add_fact(self, semantic_memory: SemanticMemory):
        """Should add a fact."""
        fact_id = await semantic_memory.add_fact(
            project_path="/path/to/project",
            category=FactCategory.CONVENTION,
            fact="Uses pytest for testing",
            source="CLAUDE.md",
        )

        assert fact_id > 0

        facts = await semantic_memory.get_facts("/path/to/project")
        assert len(facts) == 1
        assert facts[0].fact == "Uses pytest for testing"

    @pytest.mark.asyncio
    async def test_add_duplicate_fact_updates(self, semantic_memory: SemanticMemory):
        """Adding duplicate fact should update if higher confidence."""
        project = "/project"

        # Add initial fact
        fact_id1 = await semantic_memory.add_fact(
            project_path=project,
            category=FactCategory.CONVENTION,
            fact="Uses pytest for testing",
            confidence=0.8,
        )

        # Add same fact with higher confidence
        fact_id2 = await semantic_memory.add_fact(
            project_path=project,
            category=FactCategory.CONVENTION,
            fact="Uses pytest for testing",
            confidence=0.95,
        )

        # Should be same fact ID
        assert fact_id1 == fact_id2

        # Should only have one fact
        facts = await semantic_memory.get_facts(project)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_get_facts_by_category(self, semantic_memory: SemanticMemory):
        """Should filter facts by category."""
        project = "/project"

        await semantic_memory.add_fact(
            project, FactCategory.CONVENTION, "Convention fact"
        )
        await semantic_memory.add_fact(
            project, FactCategory.ARCHITECTURE, "Architecture fact"
        )
        await semantic_memory.add_fact(
            project, FactCategory.TESTING, "Testing fact"
        )

        conventions = await semantic_memory.get_conventions(project)
        architecture = await semantic_memory.get_architecture(project)
        testing = await semantic_memory.get_facts(project, FactCategory.TESTING)

        assert len(conventions) == 1
        assert len(architecture) == 1
        assert len(testing) == 1


class TestSemanticMemorySearch:
    """Tests for fact searching."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def semantic_memory(self, store: MemoryStore) -> SemanticMemory:
        """Create semantic memory."""
        return SemanticMemory(store)

    @pytest.mark.asyncio
    async def test_search_by_keywords(self, semantic_memory: SemanticMemory):
        """Should search facts by keywords."""
        project = "/project"

        await semantic_memory.add_fact(
            project, FactCategory.TESTING, "Uses pytest with 80% coverage requirement"
        )
        await semantic_memory.add_fact(
            project, FactCategory.TOOL, "Uses ruff for linting"
        )
        await semantic_memory.add_fact(
            project, FactCategory.CONVENTION, "All code must have type hints"
        )

        # Search for pytest
        results = await semantic_memory.search("pytest", project)
        assert len(results) >= 1
        assert any("pytest" in r.fact.lower() for r in results)

        # Search for type hints
        results = await semantic_memory.search("type hints", project)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_by_category(self, semantic_memory: SemanticMemory):
        """Should filter search by category."""
        project = "/project"

        await semantic_memory.add_fact(
            project, FactCategory.TESTING, "Uses pytest"
        )
        await semantic_memory.add_fact(
            project, FactCategory.TOOL, "Uses pytest-cov plugin"
        )

        # Search only in TESTING category
        results = await semantic_memory.search(
            "pytest", project, category=FactCategory.TESTING
        )
        assert len(results) == 1
        assert results[0].category == FactCategory.TESTING

    @pytest.mark.asyncio
    async def test_get_context_for_question(self, semantic_memory: SemanticMemory):
        """Should get relevant facts for a question."""
        project = "/project"

        # Add various facts
        await semantic_memory.add_fact(
            project, FactCategory.TESTING, "Uses pytest for testing"
        )
        await semantic_memory.add_fact(
            project, FactCategory.TESTING, "Requires 80% coverage"
        )
        await semantic_memory.add_fact(
            project, FactCategory.TOOL, "Uses TypeScript"
        )

        # Get context for a testing question
        facts = await semantic_memory.get_context_for_question(
            project, "How should I run the tests?"
        )

        # Should return relevant facts
        assert len(facts) >= 1


class TestSemanticMemoryLearning:
    """Tests for learning from various sources."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def semantic_memory(self, store: MemoryStore) -> SemanticMemory:
        """Create semantic memory."""
        return SemanticMemory(store)

    @pytest.mark.asyncio
    async def test_learn_from_database_qa(self, semantic_memory: SemanticMemory):
        """Should learn fact from database question/answer."""
        project = "/project"

        fact_id = await semantic_memory.learn_from_answer(
            project_path=project,
            question="Which database should I use?",
            answer="PostgreSQL",
            source="user_answer",
        )

        assert fact_id is not None

        # Should have created a decision fact
        decisions = await semantic_memory.get_decisions(project)
        assert len(decisions) >= 1
        assert any("PostgreSQL" in d.fact for d in decisions)

    @pytest.mark.asyncio
    async def test_learn_from_framework_qa(self, semantic_memory: SemanticMemory):
        """Should learn fact from framework question/answer."""
        project = "/project"

        fact_id = await semantic_memory.learn_from_answer(
            project_path=project,
            question="Which testing framework?",
            answer="pytest",
        )

        assert fact_id is not None

    @pytest.mark.asyncio
    async def test_learn_from_pyproject_toml(self, semantic_memory: SemanticMemory):
        """Should learn facts from pyproject.toml."""
        project = "/project"

        pyproject_content = """
[project]
name = "myproject"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
]
"""

        facts_learned = await semantic_memory.learn_from_file(
            project_path=project,
            file_path="pyproject.toml",
            file_content=pyproject_content,
        )

        assert facts_learned >= 2  # At least Python version and pytest

        facts = await semantic_memory.get_facts(project)
        fact_texts = [f.fact for f in facts]

        assert any("Python" in f for f in fact_texts)
        assert any("pytest" in f for f in fact_texts)

    @pytest.mark.asyncio
    async def test_learn_from_package_json(self, semantic_memory: SemanticMemory):
        """Should learn facts from package.json."""
        project = "/project"

        package_json = """
{
    "name": "my-app",
    "dependencies": {
        "react": "^18.0.0"
    },
    "devDependencies": {
        "typescript": "^5.0.0",
        "jest": "^29.0.0",
        "eslint": "^8.0.0"
    }
}
"""

        facts_learned = await semantic_memory.learn_from_file(
            project_path=project,
            file_path="package.json",
            file_content=package_json,
        )

        assert facts_learned >= 3  # React, TypeScript, Jest

        facts = await semantic_memory.get_facts(project)
        fact_texts = [f.fact for f in facts]

        assert any("React" in f for f in fact_texts)
        assert any("TypeScript" in f for f in fact_texts)
        assert any("Jest" in f for f in fact_texts)

    @pytest.mark.asyncio
    async def test_learn_from_claude_md(self, semantic_memory: SemanticMemory):
        """Should learn facts from CLAUDE.md."""
        project = "/project"

        claude_md = """
# MyProject

## Overview
A web application for managing tasks.

## Testing
Uses pytest with coverage requirements.
Run with: pytest --cov=src

## Conventions
- All functions must have docstrings
- Use type hints throughout
"""

        facts_learned = await semantic_memory.learn_from_file(
            project_path=project,
            file_path="CLAUDE.md",
            file_content=claude_md,
        )

        assert facts_learned >= 1

        facts = await semantic_memory.get_facts(project)
        assert len(facts) >= 1


class TestRealWorldScenarios:
    """Tests simulating real-world Claude Code scenarios."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> MemoryStore:
        """Create and initialize a memory store."""
        config = MemoryStoreConfig(db_path=tmp_path / "test.db")
        store = MemoryStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def semantic_memory(self, store: MemoryStore) -> SemanticMemory:
        """Create semantic memory."""
        return SemanticMemory(store)

    @pytest.mark.asyncio
    async def test_scenario_python_project_setup(
        self, semantic_memory: SemanticMemory
    ):
        """Scenario: Learning about a Python project."""
        project = "/Users/dev/myapp"

        # Learn from project files
        pyproject = """
[project]
name = "myapp"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
"""
        await semantic_memory.learn_from_file(project, "pyproject.toml", pyproject)

        # Learn from CLAUDE.md
        claude_md = """
## Testing
Run tests with: pytest
Coverage requirement: 80%

## Conventions
Use async/await for all I/O
"""
        await semantic_memory.learn_from_file(project, "CLAUDE.md", claude_md)

        # Verify learned facts
        facts = await semantic_memory.get_facts(project)
        assert len(facts) >= 3

        # Should be able to answer testing questions
        testing_facts = await semantic_memory.get_context_for_question(
            project, "How do I run tests?"
        )
        assert len(testing_facts) >= 1

    @pytest.mark.asyncio
    async def test_scenario_node_project_setup(
        self, semantic_memory: SemanticMemory
    ):
        """Scenario: Learning about a Node.js project."""
        project = "/Users/dev/frontend"

        package_json = """
{
    "name": "frontend",
    "scripts": {
        "dev": "next dev",
        "build": "next build",
        "test": "vitest"
    },
    "dependencies": {
        "next": "^14.0.0",
        "react": "^18.0.0"
    },
    "devDependencies": {
        "typescript": "^5.0.0",
        "vitest": "^1.0.0",
        "prettier": "^3.0.0"
    }
}
"""
        await semantic_memory.learn_from_file(project, "package.json", package_json)

        facts = await semantic_memory.get_facts(project)

        # Should recognize React, TypeScript, Vitest, Prettier
        fact_texts = " ".join(f.fact for f in facts)
        assert "React" in fact_texts
        assert "TypeScript" in fact_texts

    @pytest.mark.asyncio
    async def test_scenario_user_preferences(
        self, semantic_memory: SemanticMemory
    ):
        """Scenario: Learning user preferences from Q&A."""
        project = "/project"

        # User answers various questions
        qa_pairs = [
            ("Which database?", "PostgreSQL"),
            ("Which testing framework?", "pytest"),
            ("Preferred style?", "functional over OOP"),
        ]

        for question, answer in qa_pairs:
            await semantic_memory.learn_from_answer(project, question, answer)

        # Get all preferences
        prefs = await semantic_memory.get_preferences(project)
        decisions = await semantic_memory.get_decisions(project)

        # Should have recorded the preferences
        all_facts = prefs + decisions
        assert len(all_facts) >= 2

    @pytest.mark.asyncio
    async def test_scenario_multi_project(
        self, semantic_memory: SemanticMemory
    ):
        """Scenario: Maintaining facts for multiple projects."""
        project1 = "/projects/backend"
        project2 = "/projects/frontend"

        # Backend facts
        await semantic_memory.add_fact(
            project1, FactCategory.TOOL, "Uses Python 3.11"
        )
        await semantic_memory.add_fact(
            project1, FactCategory.TESTING, "Uses pytest"
        )

        # Frontend facts
        await semantic_memory.add_fact(
            project2, FactCategory.TOOL, "Uses TypeScript"
        )
        await semantic_memory.add_fact(
            project2, FactCategory.TESTING, "Uses Vitest"
        )

        # Query by project
        backend_facts = await semantic_memory.get_facts(project1)
        frontend_facts = await semantic_memory.get_facts(project2)

        assert len(backend_facts) == 2
        assert len(frontend_facts) == 2

        # Verify isolation
        backend_texts = " ".join(f.fact for f in backend_facts)
        frontend_texts = " ".join(f.fact for f in frontend_facts)

        assert "Python" in backend_texts
        assert "TypeScript" in frontend_texts

    @pytest.mark.asyncio
    async def test_scenario_statistics(self, semantic_memory: SemanticMemory):
        """Scenario: Getting semantic memory statistics."""
        project = "/project"

        # Add various facts
        await semantic_memory.add_fact(project, FactCategory.CONVENTION, "Fact 1")
        await semantic_memory.add_fact(project, FactCategory.CONVENTION, "Fact 2")
        await semantic_memory.add_fact(project, FactCategory.ARCHITECTURE, "Fact 3")
        await semantic_memory.add_fact(project, FactCategory.TESTING, "Fact 4")

        stats = await semantic_memory.get_statistics(project)

        assert stats["total"] == 4
        assert stats["by_category"][FactCategory.CONVENTION] == 2
        assert stats["by_category"][FactCategory.ARCHITECTURE] == 1
        assert stats["by_category"][FactCategory.TESTING] == 1

    @pytest.mark.asyncio
    async def test_scenario_fact_removal(self, semantic_memory: SemanticMemory):
        """Scenario: Removing outdated facts."""
        project = "/project"

        # Add a fact
        fact_id = await semantic_memory.add_fact(
            project, FactCategory.TOOL, "Uses Python 3.10"
        )

        # Update to Python 3.11
        await semantic_memory.store.update_fact(
            fact_id, "Uses Python 3.11", confidence=1.0
        )

        facts = await semantic_memory.get_facts(project)
        assert len(facts) == 1
        assert "3.11" in facts[0].fact

    @pytest.mark.asyncio
    async def test_scenario_architecture_decisions(
        self, semantic_memory: SemanticMemory
    ):
        """Scenario: Recording architectural decisions."""
        project = "/project"

        decisions = [
            "Use microservices architecture",
            "PostgreSQL for primary database",
            "Redis for caching",
            "REST API with OpenAPI spec",
            "JWT for authentication",
        ]

        for decision in decisions:
            await semantic_memory.add_fact(
                project,
                FactCategory.DECISION,
                decision,
                source="architecture_review",
            )

        # Query architecture
        arch_facts = await semantic_memory.get_decisions(project)

        assert len(arch_facts) == 5

        # Search for specific decision
        db_facts = await semantic_memory.search("database", project)
        assert len(db_facts) >= 1
        assert any("PostgreSQL" in f.fact for f in db_facts)
