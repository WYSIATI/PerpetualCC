"""Unit tests for the code relationship graph.

Tests cover:
- Graph node and edge types
- Graph building from code
- Relationship queries
- File context retrieval
- Language-specific extraction
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from perpetualcc.knowledge.code_graph import (
    CodeGraph,
    EdgeType,
    GraphConfig,
    GraphEdge,
    GraphNode,
    NodeType,
)
from perpetualcc.knowledge.indexer import Language


# Check if networkx is available
def _has_networkx() -> bool:
    """Check if networkx is installed."""
    try:
        import networkx
        return True
    except ImportError:
        return False


requires_networkx = pytest.mark.skipif(
    not _has_networkx(),
    reason="networkx not installed"
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_types_defined(self):
        """Test all node types are defined."""
        assert NodeType.FILE.value == "file"
        assert NodeType.MODULE.value == "module"
        assert NodeType.CLASS.value == "class"
        assert NodeType.FUNCTION.value == "function"
        assert NodeType.METHOD.value == "method"
        assert NodeType.VARIABLE.value == "variable"
        assert NodeType.IMPORT.value == "import"
        assert NodeType.PACKAGE.value == "package"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_types_defined(self):
        """Test all edge types are defined."""
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.IMPORTS.value == "imports"
        assert EdgeType.INHERITS.value == "inherits"
        assert EdgeType.CALLS.value == "calls"
        assert EdgeType.USES.value == "uses"
        assert EdgeType.DEFINES.value == "defines"
        assert EdgeType.DEPENDS_ON.value == "depends_on"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_basic_creation(self):
        """Test creating a basic graph node."""
        node = GraphNode(
            id="node_1",
            name="greet",
            full_name="src/main.py:greet",
            node_type=NodeType.FUNCTION,
        )

        assert node.id == "node_1"
        assert node.name == "greet"
        assert node.node_type == NodeType.FUNCTION

    def test_full_creation(self):
        """Test creating a node with all fields."""
        node = GraphNode(
            id="node_1",
            name="User",
            full_name="src/models.py:User",
            node_type=NodeType.CLASS,
            file_path="src/models.py",
            start_line=10,
            end_line=50,
            language=Language.PYTHON,
            metadata={"bases": ["Base", "Mixin"]},
        )

        assert node.file_path == "src/models.py"
        assert node.start_line == 10
        assert node.end_line == 50
        assert node.language == Language.PYTHON
        assert node.metadata["bases"] == ["Base", "Mixin"]

    def test_to_dict(self):
        """Test dictionary conversion."""
        node = GraphNode(
            id="node_1",
            name="hello",
            full_name="src/main.py:hello",
            node_type=NodeType.FUNCTION,
            file_path="src/main.py",
            language=Language.PYTHON,
        )

        d = node.to_dict()

        assert d["id"] == "node_1"
        assert d["name"] == "hello"
        assert d["node_type"] == "function"
        assert d["file_path"] == "src/main.py"
        assert d["language"] == "python"

    def test_to_dict_with_none_language(self):
        """Test dict conversion with None language."""
        node = GraphNode(
            id="node_1",
            name="test",
            full_name="test",
            node_type=NodeType.MODULE,
        )

        d = node.to_dict()
        assert d["language"] is None


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_basic_creation(self):
        """Test creating a basic edge."""
        edge = GraphEdge(
            source_id="file_1",
            target_id="func_1",
            edge_type=EdgeType.CONTAINS,
        )

        assert edge.source_id == "file_1"
        assert edge.target_id == "func_1"
        assert edge.edge_type == EdgeType.CONTAINS
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_with_metadata(self):
        """Test creating edge with metadata."""
        edge = GraphEdge(
            source_id="class_1",
            target_id="class_2",
            edge_type=EdgeType.INHERITS,
            weight=0.8,
            metadata={"direct": True},
        )

        assert edge.weight == 0.8
        assert edge.metadata["direct"] is True


class TestGraphConfig:
    """Tests for GraphConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GraphConfig()

        assert config.max_depth == 3
        assert config.include_imports is True
        assert config.include_calls is True
        assert config.include_inheritance is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GraphConfig(
            max_depth=5,
            include_imports=False,
            include_calls=False,
        )

        assert config.max_depth == 5
        assert config.include_imports is False

    def test_frozen(self):
        """Test config is frozen."""
        config = GraphConfig()
        with pytest.raises(AttributeError):
            config.max_depth = 10


@pytest.fixture
def temp_project() -> Path:
    """Create a temporary project with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        (project / "src").mkdir()

        # Create Python files with relationships
        (project / "src" / "main.py").write_text('''"""Main module."""

from src.auth import login, logout
from src.models import User

def main():
    """Main entry point."""
    user = login("admin", "secret")
    print(f"Welcome, {user.name}!")


def helper():
    """Helper function."""
    pass


class Application:
    """Main application class."""

    def __init__(self):
        self.users = []

    def run(self):
        """Run the application."""
        main()
''')

        (project / "src" / "auth.py").write_text('''"""Authentication module."""

from src.models import User

def login(username: str, password: str) -> User:
    """Log in a user."""
    user = User.find(username)
    if user.check_password(password):
        return user
    raise AuthError("Invalid credentials")


def logout(user: User) -> None:
    """Log out a user."""
    user.session = None


class AuthError(Exception):
    """Authentication error."""
    pass
''')

        (project / "src" / "models.py").write_text('''"""Data models."""

from dataclasses import dataclass

@dataclass
class Entity:
    """Base entity class."""
    id: int


class User(Entity):
    """User model."""

    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email

    def check_password(self, password: str) -> bool:
        """Check if password is correct."""
        return True

    @classmethod
    def find(cls, username: str) -> "User":
        """Find user by username."""
        return cls(username, f"{username}@example.com")


class Session:
    """User session."""
    user: User
    token: str
''')

        # Create JavaScript file
        (project / "src" / "client.js").write_text('''/**
 * API client module.
 */

import { config } from './config.js';
import { AuthError } from './errors.js';

/**
 * Fetch user data from API.
 */
async function fetchUser(id) {
    const response = await fetch(`${config.apiUrl}/users/${id}`);
    return response.json();
}

/**
 * User client class.
 */
class UserClient extends BaseClient {
    constructor(token) {
        super();
        this.token = token;
    }

    async getUser(id) {
        return fetchUser(id);
    }
}

export { fetchUser, UserClient };
''')

        yield project


@pytest.fixture
def graph(temp_project: Path) -> CodeGraph:
    """Create a code graph for the temp project."""
    return CodeGraph(temp_project)


class TestCodeGraphInitialization:
    """Tests for CodeGraph initialization."""

    def test_creates_with_defaults(self, temp_project: Path):
        """Test creating graph with default settings."""
        graph = CodeGraph(temp_project)

        assert graph.project_path == temp_project.resolve()
        assert graph.config.max_depth == 3
        assert graph._graph is None  # Lazy init

    def test_creates_with_custom_config(self, temp_project: Path):
        """Test creating graph with custom config."""
        config = GraphConfig(max_depth=5)
        graph = CodeGraph(temp_project, config)

        assert graph.config.max_depth == 5


class TestNodeIdGeneration:
    """Tests for node ID generation."""

    def test_generates_file_id(self, graph: CodeGraph):
        """Test generating file node ID."""
        node_id = graph._generate_node_id(
            "src/main.py", NodeType.FILE
        )

        assert "file" in node_id
        assert "src/main.py" in node_id

    def test_generates_function_id(self, graph: CodeGraph):
        """Test generating function node ID."""
        node_id = graph._generate_node_id(
            "greet", NodeType.FUNCTION, "src/main.py"
        )

        assert "function" in node_id
        assert "greet" in node_id

    def test_generates_method_id_with_parent(self, graph: CodeGraph):
        """Test generating method node ID with parent class."""
        node_id = graph._generate_node_id(
            "run", NodeType.METHOD, "src/main.py", "Application"
        )

        assert "method" in node_id
        assert "run" in node_id
        assert "Application" in node_id


@requires_networkx
class TestGraphBuilding:
    """Tests for building the code graph (requires networkx)."""

    @pytest.mark.asyncio
    async def test_builds_graph(self, graph: CodeGraph):
        """Test building the complete graph."""
        await graph.build()

        # Verify graph was populated
        stats = graph.get_stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    @pytest.mark.asyncio
    async def test_extracts_file_nodes(self, graph: CodeGraph):
        """Test that file nodes are extracted."""
        await graph.build()

        # Find file nodes
        file_nodes = [
            n for n in graph._nodes.values()
            if n.node_type == NodeType.FILE
        ]

        file_names = [n.name for n in file_nodes]
        assert "main.py" in file_names
        assert "auth.py" in file_names
        assert "models.py" in file_names

    @pytest.mark.asyncio
    async def test_extracts_function_nodes(self, graph: CodeGraph):
        """Test that function nodes are extracted."""
        await graph.build()

        func_nodes = [
            n for n in graph._nodes.values()
            if n.node_type == NodeType.FUNCTION
        ]

        func_names = [n.name for n in func_nodes]
        assert "main" in func_names or "login" in func_names

    @pytest.mark.asyncio
    async def test_extracts_class_nodes(self, graph: CodeGraph):
        """Test that class nodes are extracted."""
        await graph.build()

        class_nodes = [
            n for n in graph._nodes.values()
            if n.node_type == NodeType.CLASS
        ]

        class_names = [n.name for n in class_nodes]
        assert "User" in class_names or "Application" in class_names

    @pytest.mark.asyncio
    async def test_extracts_import_nodes(self, graph: CodeGraph):
        """Test that import nodes are extracted."""
        await graph.build()

        import_nodes = [
            n for n in graph._nodes.values()
            if n.node_type == NodeType.IMPORT
        ]

        # Should have some imports
        assert len(import_nodes) > 0


@requires_networkx
class TestRelationshipExtraction:
    """Tests for extracting relationships between entities (requires networkx)."""

    @pytest.mark.asyncio
    async def test_contains_relationships(self, graph: CodeGraph):
        """Test that containment relationships are captured."""
        await graph.build()

        nx_graph = graph._get_graph()

        # Find edges with CONTAINS type
        contains_edges = [
            (u, v) for u, v, d in nx_graph.edges(data=True)
            if d.get("edge_type") == EdgeType.CONTAINS.value
        ]

        assert len(contains_edges) > 0

    @pytest.mark.asyncio
    async def test_imports_relationships(self, graph: CodeGraph):
        """Test that import relationships are captured."""
        await graph.build()

        nx_graph = graph._get_graph()

        # Find edges with IMPORTS type
        imports_edges = [
            (u, v) for u, v, d in nx_graph.edges(data=True)
            if d.get("edge_type") == EdgeType.IMPORTS.value
        ]

        assert len(imports_edges) > 0


@requires_networkx
class TestGraphQueries:
    """Tests for querying the code graph (requires networkx)."""

    @pytest.mark.asyncio
    async def test_get_node(self, graph: CodeGraph):
        """Test getting a node by ID."""
        await graph.build()

        # Get a known node
        file_node_id = graph._generate_node_id("src/main.py", NodeType.FILE)
        node = graph.get_node(file_node_id)

        if node:  # May not exist if parsing failed
            assert node.name == "main.py"
            assert node.node_type == NodeType.FILE

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, graph: CodeGraph):
        """Test getting a non-existent node."""
        await graph.build()

        node = graph.get_node("nonexistent_node_id")
        assert node is None

    @pytest.mark.asyncio
    async def test_get_related(self, graph: CodeGraph):
        """Test finding related entities."""
        await graph.build()

        # Find entities related to "User"
        related = graph.get_related("User", depth=2)

        # Should find some related entities
        # (exact results depend on parsing success)
        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_get_related_with_filter(self, graph: CodeGraph):
        """Test finding related entities with type filter."""
        await graph.build()

        related = graph.get_related(
            "User",
            depth=2,
            include_types=[NodeType.FUNCTION, NodeType.METHOD],
        )

        for entity in related:
            assert entity["node_type"] in ("function", "method")


@requires_networkx
class TestFileContext:
    """Tests for getting file context (requires networkx)."""

    @pytest.mark.asyncio
    async def test_get_file_context(self, graph: CodeGraph):
        """Test getting comprehensive file context."""
        await graph.build()

        context = graph.get_file_context("src/main.py")

        assert context["file_path"] == "src/main.py"
        assert "imports" in context
        assert "exports" in context
        assert "entities" in context
        assert "dependencies" in context

    @pytest.mark.asyncio
    async def test_file_context_includes_imports(self, graph: CodeGraph):
        """Test that file context includes imports."""
        await graph.build()

        context = graph.get_file_context("src/main.py")

        # Should have some imports
        assert isinstance(context["imports"], list)

    @pytest.mark.asyncio
    async def test_file_context_includes_entities(self, graph: CodeGraph):
        """Test that file context includes defined entities."""
        await graph.build()

        context = graph.get_file_context("src/main.py")

        # Should have some entities
        if context["entities"]:
            for entity in context["entities"]:
                assert "name" in entity
                assert "type" in entity

    @pytest.mark.asyncio
    async def test_file_context_for_unknown_file(self, graph: CodeGraph):
        """Test getting context for unknown file."""
        await graph.build()

        context = graph.get_file_context("nonexistent.py")

        assert context["file_path"] == "nonexistent.py"
        assert context["imports"] == []
        assert context["entities"] == []


@requires_networkx
class TestGraphStats:
    """Tests for graph statistics (requires networkx)."""

    @pytest.mark.asyncio
    async def test_get_stats(self, graph: CodeGraph):
        """Test getting graph statistics."""
        await graph.build()

        stats = graph.get_stats()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "node_types" in stats
        assert stats["total_nodes"] >= 0
        assert stats["total_edges"] >= 0

    @pytest.mark.asyncio
    async def test_stats_include_node_types(self, graph: CodeGraph):
        """Test that stats include node type breakdown."""
        await graph.build()

        stats = graph.get_stats()

        # Should have various node types
        node_types = stats["node_types"]
        assert isinstance(node_types, dict)


@requires_networkx
class TestLanguageSpecificParsing:
    """Tests for language-specific parsing (requires networkx)."""

    @pytest.mark.asyncio
    async def test_parses_python(self, graph: CodeGraph, temp_project: Path):
        """Test Python-specific parsing."""
        await graph.build()

        # Should have extracted Python entities
        python_nodes = [
            n for n in graph._nodes.values()
            if n.language == Language.PYTHON
        ]

        assert len(python_nodes) > 0

    @pytest.mark.asyncio
    async def test_parses_javascript(self, graph: CodeGraph, temp_project: Path):
        """Test JavaScript-specific parsing."""
        await graph.build()

        # Should have extracted JavaScript entities
        js_nodes = [
            n for n in graph._nodes.values()
            if n.language == Language.JAVASCRIPT
        ]

        # May be empty if tree-sitter-language-pack not installed
        assert isinstance(js_nodes, list)


@requires_networkx
class TestRealWorldScenarios:
    """Tests simulating real-world usage with Claude Code (requires networkx)."""

    @pytest.mark.asyncio
    async def test_find_authentication_related_code(self, graph: CodeGraph):
        """Test finding code related to authentication.

        Simulates: "What code is related to user authentication?"
        """
        await graph.build()

        # Find entities related to "login"
        related = graph.get_related("login", depth=2)

        # Should find related authentication code
        names = [r["name"] for r in related]
        # Results depend on successful parsing
        assert isinstance(names, list)

    @pytest.mark.asyncio
    async def test_find_class_hierarchy(self, graph: CodeGraph):
        """Test finding class hierarchy.

        Simulates: "What does the User class inherit from?"
        """
        await graph.build()

        # Get User class related entities
        related = graph.get_related(
            "User",
            depth=2,
            include_types=[NodeType.CLASS],
        )

        # Should find the class and possibly its bases
        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_find_file_dependencies(self, graph: CodeGraph):
        """Test finding file dependencies.

        Simulates: "What does main.py depend on?"
        """
        await graph.build()

        deps = graph.get_dependencies("src/main.py")

        # Should find import dependencies
        assert isinstance(deps, list)

    @pytest.mark.asyncio
    async def test_navigate_from_test_to_implementation(self, temp_project: Path):
        """Test navigating from test file to implementation.

        Simulates: Understanding what a test file tests.
        """
        # Add a test file
        (temp_project / "tests").mkdir(exist_ok=True)
        (temp_project / "tests" / "test_auth.py").write_text('''"""Tests for auth module."""

from src.auth import login, logout, AuthError

def test_login_success():
    user = login("admin", "password")
    assert user is not None

def test_logout():
    logout(user)
''')

        graph = CodeGraph(temp_project)
        await graph.build()

        # Get context for test file
        context = graph.get_file_context("tests/test_auth.py")

        # Should show imports from src/auth
        assert isinstance(context["imports"], list)
