"""Code relationship graph for the Knowledge Engine.

Builds and queries a graph of code relationships using:
- tree-sitter for AST parsing
- NetworkX for graph operations

The code graph tracks:
- Files and their dependencies
- Functions and methods
- Classes and their members
- Import relationships
- Call relationships (where detectable)

This enables:
- Finding related code to a given entity
- Understanding file dependencies
- Navigating code structure
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from perpetualcc.knowledge.indexer import (
    EXTENSION_TO_LANGUAGE,
    CodebaseIndexer,
    GitignoreParser,
    IndexerConfig,
    Language,
)

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Type of node in the code graph."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    PACKAGE = "package"


class EdgeType(Enum):
    """Type of edge (relationship) in the code graph."""

    CONTAINS = "contains"  # File/class contains function/method
    IMPORTS = "imports"  # File imports module
    INHERITS = "inherits"  # Class inherits from class
    CALLS = "calls"  # Function calls function
    USES = "uses"  # Function uses variable/class
    DEFINES = "defines"  # Module defines entity
    DEPENDS_ON = "depends_on"  # File depends on file


@dataclass
class GraphNode:
    """A node in the code graph.

    Attributes:
        id: Unique node identifier
        name: Short name of the entity
        full_name: Fully qualified name
        node_type: Type of node
        file_path: Path to the source file
        start_line: Starting line number
        end_line: Ending line number
        language: Programming language
        metadata: Additional node data
    """

    id: str
    name: str
    full_name: str
    node_type: NodeType
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: Language | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "node_type": self.node_type.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language.value if self.language else None,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge (relationship) in the code graph.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of relationship
        weight: Edge weight (for ranking)
        metadata: Additional edge data
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for the code graph.

    Attributes:
        max_depth: Maximum traversal depth for related queries
        include_imports: Whether to track import relationships
        include_calls: Whether to track call relationships
        include_inheritance: Whether to track inheritance
        indexer_config: Configuration for code indexing
    """

    max_depth: int = 3
    include_imports: bool = True
    include_calls: bool = True
    include_inheritance: bool = True
    indexer_config: IndexerConfig | None = None


class CodeGraph:
    """Code relationship graph for understanding codebase structure.

    Uses NetworkX to build a directed graph of code relationships,
    enabling queries like:
    - What does this function depend on?
    - What calls this function?
    - What classes inherit from this?
    - What files import this module?

    Usage:
        graph = CodeGraph("./my-project")
        await graph.build()
        related = graph.get_related("my_function", depth=2)
    """

    def __init__(
        self,
        project_path: str | Path,
        config: GraphConfig | None = None,
    ):
        """Initialize the code graph.

        Args:
            project_path: Path to the project root
            config: Graph configuration
        """
        self.project_path = Path(project_path).resolve()
        self.config = config or GraphConfig()
        self._graph = None
        self._nodes: dict[str, GraphNode] = {}
        self._indexer = CodebaseIndexer(
            self.project_path,
            self.config.indexer_config,
        )
        self._parser_cache: dict[Language, Any] = {}
        self._gitignore = GitignoreParser(self.project_path)

    def _get_networkx(self):
        """Get NetworkX module, importing lazily."""
        try:
            import networkx as nx

            return nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for code graph functionality. "
                "Install with: pip install networkx"
            ) from e

    def _get_graph(self):
        """Get or create the NetworkX graph."""
        if self._graph is None:
            nx = self._get_networkx()
            self._graph = nx.DiGraph()
        return self._graph

    def _generate_node_id(
        self,
        name: str,
        node_type: NodeType,
        file_path: str | None = None,
        parent: str | None = None,
    ) -> str:
        """Generate a unique node ID.

        Args:
            name: Entity name
            node_type: Type of node
            file_path: Optional file path
            parent: Optional parent name

        Returns:
            Unique node identifier
        """
        parts = [node_type.value]
        if file_path:
            parts.append(file_path)
        if parent:
            parts.append(parent)
        parts.append(name)
        return ":".join(parts)

    def _get_parser(self, language: Language):
        """Get tree-sitter parser for a language."""
        if language in self._parser_cache:
            return self._parser_cache[language]

        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language.value)
            self._parser_cache[language] = parser
            return parser
        except Exception as e:
            logger.debug("Parser not available for %s: %s", language.value, e)
            self._parser_cache[language] = None
            return None

    async def build(self, progress_callback: callable | None = None) -> None:
        """Build the code graph from the project.

        Parses all source files and extracts:
        - File nodes
        - Class/function definitions
        - Import relationships
        - Inheritance (where detectable)
        - Call relationships (limited)

        Args:
            progress_callback: Optional callback for progress updates
        """
        logger.info("Building code graph for: %s", self.project_path)

        graph = self._get_graph()
        graph.clear()
        self._nodes.clear()

        files_processed = 0

        def _build():
            nonlocal files_processed

            for file_path in self._indexer.scan_files():
                self._process_file(file_path)
                files_processed += 1

                if progress_callback:
                    progress_callback(files_processed, -1, str(file_path))

        await asyncio.to_thread(_build)

        logger.info(
            "Code graph built: %d nodes, %d edges, %d files",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            files_processed,
        )

    def _process_file(self, file_path: Path) -> None:
        """Process a single file to extract graph nodes and edges.

        Args:
            file_path: Path to the file
        """
        language = self._get_language(file_path)
        if language == Language.UNKNOWN:
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Failed to read %s: %s", file_path, e)
            return

        relative_path = str(file_path.relative_to(self.project_path))

        # Add file node
        file_node_id = self._generate_node_id(relative_path, NodeType.FILE)
        file_node = GraphNode(
            id=file_node_id,
            name=file_path.name,
            full_name=relative_path,
            node_type=NodeType.FILE,
            file_path=relative_path,
            language=language,
        )
        self._add_node(file_node)

        # Parse with tree-sitter if available
        parser = self._get_parser(language)
        if parser:
            self._parse_with_tree_sitter(parser, content, language, relative_path, file_node_id)
        else:
            self._parse_simple(content, language, relative_path, file_node_id)

    def _get_language(self, file_path: Path) -> Language:
        """Get language for a file."""
        suffix = file_path.suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(suffix, Language.UNKNOWN)

    def _add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        graph = self._get_graph()
        self._nodes[node.id] = node
        graph.add_node(
            node.id,
            **node.to_dict(),
        )

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        **metadata,
    ) -> None:
        """Add an edge to the graph."""
        graph = self._get_graph()

        # Only add edge if both nodes exist
        if source_id in self._nodes and target_id in self._nodes:
            graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type.value,
                weight=weight,
                **metadata,
            )

    def _parse_with_tree_sitter(
        self,
        parser,
        content: str,
        language: Language,
        file_path: str,
        file_node_id: str,
    ) -> None:
        """Parse file with tree-sitter to extract graph nodes.

        Args:
            parser: Tree-sitter parser
            content: File content
            language: Programming language
            file_path: Relative file path
            file_node_id: ID of the file node
        """
        try:
            tree = parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node
        except Exception as e:
            logger.debug("Tree-sitter parse failed for %s: %s", file_path, e)
            return

        lines = content.splitlines()

        if language == Language.PYTHON:
            self._extract_python_graph(root_node, lines, file_path, file_node_id)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            self._extract_js_ts_graph(root_node, lines, file_path, file_node_id, language)
        elif language == Language.RUST:
            self._extract_rust_graph(root_node, lines, file_path, file_node_id)
        elif language == Language.GO:
            self._extract_go_graph(root_node, lines, file_path, file_node_id)
        else:
            # Generic extraction
            self._extract_generic_graph(root_node, lines, file_path, file_node_id, language)

    def _extract_python_graph(
        self,
        root_node,
        lines: list[str],
        file_path: str,
        file_node_id: str,
    ) -> None:
        """Extract graph nodes from Python code."""

        def visit_node(node, parent_id: str | None = None, class_name: str | None = None):
            # Handle imports
            if node.type in ("import_statement", "import_from_statement"):
                self._handle_python_import(node, file_path, file_node_id)

            # Handle class definitions
            elif node.type == "class_definition":
                name = None
                bases = []
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                    elif child.type == "argument_list":
                        # Extract base classes
                        for arg in child.children:
                            if arg.type == "identifier":
                                bases.append(arg.text.decode("utf-8"))

                if name:
                    node_id = self._generate_node_id(name, NodeType.CLASS, file_path)
                    class_node = GraphNode(
                        id=node_id,
                        name=name,
                        full_name=f"{file_path}:{name}",
                        node_type=NodeType.CLASS,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.PYTHON,
                        metadata={"bases": bases},
                    )
                    self._add_node(class_node)
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

                    # Add inheritance edges
                    for base in bases:
                        base_id = self._generate_node_id(base, NodeType.CLASS, file_path)
                        if base_id not in self._nodes:
                            # Create placeholder for base class
                            self._add_node(
                                GraphNode(
                                    id=base_id,
                                    name=base,
                                    full_name=base,
                                    node_type=NodeType.CLASS,
                                    language=Language.PYTHON,
                                )
                            )
                        self._add_edge(node_id, base_id, EdgeType.INHERITS)

                    # Visit class body
                    for child in node.children:
                        if child.type == "block":
                            for stmt in child.children:
                                visit_node(stmt, node_id, name)

            # Handle function/method definitions
            elif node.type == "function_definition":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    node_type = NodeType.METHOD if class_name else NodeType.FUNCTION
                    node_id = self._generate_node_id(func_name, node_type, file_path, class_name)
                    func_node = GraphNode(
                        id=node_id,
                        name=func_name,
                        full_name=(
                            f"{file_path}:{class_name}.{func_name}"
                            if class_name
                            else f"{file_path}:{func_name}"
                        ),
                        node_type=node_type,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.PYTHON,
                    )
                    self._add_node(func_node)

                    # Add containment edge
                    container_id = parent_id or file_node_id
                    self._add_edge(container_id, node_id, EdgeType.CONTAINS)

            # Visit children for top-level
            if parent_id is None:
                for child in node.children:
                    visit_node(child)

        visit_node(root_node)

    def _handle_python_import(self, node, file_path: str, file_node_id: str) -> None:
        """Handle Python import statements."""
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    module_name = child.text.decode("utf-8")
                    import_id = self._generate_node_id(module_name, NodeType.IMPORT, file_path)
                    import_node = GraphNode(
                        id=import_id,
                        name=module_name,
                        full_name=module_name,
                        node_type=NodeType.IMPORT,
                        file_path=file_path,
                        language=Language.PYTHON,
                    )
                    self._add_node(import_node)
                    self._add_edge(file_node_id, import_id, EdgeType.IMPORTS)

        elif node.type == "import_from_statement":
            module_name = None
            imported_names = []
            for child in node.children:
                if child.type == "dotted_name":
                    module_name = child.text.decode("utf-8")
                elif child.type == "import_alias":
                    for name in child.children:
                        if name.type == "identifier":
                            imported_names.append(name.text.decode("utf-8"))
                            break

            if module_name:
                import_id = self._generate_node_id(module_name, NodeType.IMPORT, file_path)
                import_node = GraphNode(
                    id=import_id,
                    name=module_name,
                    full_name=module_name,
                    node_type=NodeType.IMPORT,
                    file_path=file_path,
                    language=Language.PYTHON,
                    metadata={"imported_names": imported_names},
                )
                self._add_node(import_node)
                self._add_edge(file_node_id, import_id, EdgeType.IMPORTS)

    def _extract_js_ts_graph(
        self,
        root_node,
        lines: list[str],
        file_path: str,
        file_node_id: str,
        language: Language,
    ) -> None:
        """Extract graph nodes from JavaScript/TypeScript code."""

        def visit_node(node, parent_id: str | None = None, class_name: str | None = None):
            # Handle imports
            if node.type == "import_statement":
                self._handle_js_import(node, file_path, file_node_id)

            # Handle class declarations
            elif node.type in ("class_declaration", "class"):
                name = None
                extends = None
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                    elif child.type == "heritage":
                        for h in child.children:
                            if h.type == "extends_clause":
                                for ext in h.children:
                                    if ext.type == "identifier":
                                        extends = ext.text.decode("utf-8")

                if name:
                    node_id = self._generate_node_id(name, NodeType.CLASS, file_path)
                    class_node = GraphNode(
                        id=node_id,
                        name=name,
                        full_name=f"{file_path}:{name}",
                        node_type=NodeType.CLASS,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language,
                        metadata={"extends": extends} if extends else {},
                    )
                    self._add_node(class_node)
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

                    # Add inheritance
                    if extends:
                        base_id = self._generate_node_id(extends, NodeType.CLASS, file_path)
                        if base_id not in self._nodes:
                            self._add_node(
                                GraphNode(
                                    id=base_id,
                                    name=extends,
                                    full_name=extends,
                                    node_type=NodeType.CLASS,
                                    language=language,
                                )
                            )
                        self._add_edge(node_id, base_id, EdgeType.INHERITS)

                    # Visit class body
                    for child in node.children:
                        if child.type == "class_body":
                            for stmt in child.children:
                                visit_node(stmt, node_id, name)

            # Handle function declarations
            elif node.type in (
                "function_declaration",
                "method_definition",
                "arrow_function",
            ):
                func_name = None
                for child in node.children:
                    if child.type in ("identifier", "property_identifier"):
                        func_name = child.text.decode("utf-8")
                        break

                # For arrow functions in variable declarations
                if not func_name and node.parent:
                    if node.parent.type == "variable_declarator":
                        for sibling in node.parent.children:
                            if sibling.type == "identifier":
                                func_name = sibling.text.decode("utf-8")
                                break

                if func_name:
                    node_type = NodeType.METHOD if class_name else NodeType.FUNCTION
                    node_id = self._generate_node_id(func_name, node_type, file_path, class_name)
                    func_node = GraphNode(
                        id=node_id,
                        name=func_name,
                        full_name=(
                            f"{file_path}:{class_name}.{func_name}"
                            if class_name
                            else f"{file_path}:{func_name}"
                        ),
                        node_type=node_type,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language,
                    )
                    self._add_node(func_node)

                    container_id = parent_id or file_node_id
                    self._add_edge(container_id, node_id, EdgeType.CONTAINS)

            # Visit children
            for child in node.children:
                visit_node(child, parent_id, class_name)

        visit_node(root_node)

    def _handle_js_import(self, node, file_path: str, file_node_id: str) -> None:
        """Handle JavaScript/TypeScript imports."""
        source = None
        for child in node.children:
            if child.type == "string":
                source = child.text.decode("utf-8").strip("'\"")
                break

        if source:
            import_id = self._generate_node_id(source, NodeType.IMPORT, file_path)
            import_node = GraphNode(
                id=import_id,
                name=source,
                full_name=source,
                node_type=NodeType.IMPORT,
                file_path=file_path,
                language=Language.JAVASCRIPT,
            )
            self._add_node(import_node)
            self._add_edge(file_node_id, import_id, EdgeType.IMPORTS)

    def _extract_rust_graph(
        self,
        root_node,
        lines: list[str],
        file_path: str,
        file_node_id: str,
    ) -> None:
        """Extract graph nodes from Rust code."""

        def visit_node(node, parent_id: str | None = None, impl_type: str | None = None):
            # Handle use statements
            if node.type == "use_declaration":
                path = []
                for child in node.children:
                    if child.type == "scoped_identifier":
                        path.append(child.text.decode("utf-8"))

                if path:
                    import_name = "::".join(path)
                    import_id = self._generate_node_id(import_name, NodeType.IMPORT, file_path)
                    self._add_node(
                        GraphNode(
                            id=import_id,
                            name=import_name,
                            full_name=import_name,
                            node_type=NodeType.IMPORT,
                            file_path=file_path,
                            language=Language.RUST,
                        )
                    )
                    self._add_edge(file_node_id, import_id, EdgeType.IMPORTS)

            # Handle struct/enum/trait definitions
            elif node.type in ("struct_item", "enum_item", "trait_item"):
                name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name = child.text.decode("utf-8")
                        break

                if name:
                    node_id = self._generate_node_id(name, NodeType.CLASS, file_path)
                    type_node = GraphNode(
                        id=node_id,
                        name=name,
                        full_name=f"{file_path}:{name}",
                        node_type=NodeType.CLASS,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.RUST,
                        metadata={"kind": node.type.replace("_item", "")},
                    )
                    self._add_node(type_node)
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

            # Handle function definitions
            elif node.type == "function_item":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    node_type = NodeType.METHOD if impl_type else NodeType.FUNCTION
                    node_id = self._generate_node_id(func_name, node_type, file_path, impl_type)
                    func_node = GraphNode(
                        id=node_id,
                        name=func_name,
                        full_name=(
                            f"{file_path}:{impl_type}::{func_name}"
                            if impl_type
                            else f"{file_path}:{func_name}"
                        ),
                        node_type=node_type,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.RUST,
                    )
                    self._add_node(func_node)

                    container_id = parent_id or file_node_id
                    self._add_edge(container_id, node_id, EdgeType.CONTAINS)

            # Handle impl blocks
            elif node.type == "impl_item":
                type_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        type_name = child.text.decode("utf-8")
                        break

                impl_node_id = self._generate_node_id(
                    type_name or "impl", NodeType.CLASS, file_path
                )

                for child in node.children:
                    if child.type == "declaration_list":
                        for item in child.children:
                            visit_node(item, impl_node_id, type_name)

            # Visit children
            for child in node.children:
                if child.type not in ("declaration_list", "impl_item"):
                    visit_node(child, parent_id, impl_type)

        visit_node(root_node)

    def _extract_go_graph(
        self,
        root_node,
        lines: list[str],
        file_path: str,
        file_node_id: str,
    ) -> None:
        """Extract graph nodes from Go code."""

        def visit_node(node, parent_id: str | None = None):
            # Handle imports
            if node.type == "import_declaration":
                for child in node.children:
                    if child.type == "import_spec":
                        for spec in child.children:
                            if spec.type == "interpreted_string_literal":
                                import_path = spec.text.decode("utf-8").strip('"')
                                import_id = self._generate_node_id(
                                    import_path, NodeType.IMPORT, file_path
                                )
                                self._add_node(
                                    GraphNode(
                                        id=import_id,
                                        name=import_path,
                                        full_name=import_path,
                                        node_type=NodeType.IMPORT,
                                        file_path=file_path,
                                        language=Language.GO,
                                    )
                                )
                                self._add_edge(file_node_id, import_id, EdgeType.IMPORTS)

            # Handle type declarations
            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name = None
                        for subchild in child.children:
                            if subchild.type == "type_identifier":
                                name = subchild.text.decode("utf-8")
                                break

                        if name:
                            node_id = self._generate_node_id(name, NodeType.CLASS, file_path)
                            type_node = GraphNode(
                                id=node_id,
                                name=name,
                                full_name=f"{file_path}:{name}",
                                node_type=NodeType.CLASS,
                                file_path=file_path,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                language=Language.GO,
                            )
                            self._add_node(type_node)
                            self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

            # Handle function declarations
            elif node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    node_id = self._generate_node_id(func_name, NodeType.FUNCTION, file_path)
                    func_node = GraphNode(
                        id=node_id,
                        name=func_name,
                        full_name=f"{file_path}:{func_name}",
                        node_type=NodeType.FUNCTION,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.GO,
                    )
                    self._add_node(func_node)
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

            # Handle method declarations
            elif node.type == "method_declaration":
                func_name = None
                receiver_type = None

                for child in node.children:
                    if child.type == "field_identifier":
                        func_name = child.text.decode("utf-8")
                    elif child.type == "parameter_list":
                        for param in child.children:
                            if param.type == "parameter_declaration":
                                for p in param.children:
                                    if p.type == "type_identifier":
                                        receiver_type = p.text.decode("utf-8")
                                        break

                if func_name:
                    node_id = self._generate_node_id(
                        func_name, NodeType.METHOD, file_path, receiver_type
                    )
                    method_node = GraphNode(
                        id=node_id,
                        name=func_name,
                        full_name=(
                            f"{file_path}:{receiver_type}.{func_name}"
                            if receiver_type
                            else f"{file_path}:{func_name}"
                        ),
                        node_type=NodeType.METHOD,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=Language.GO,
                        metadata={"receiver": receiver_type} if receiver_type else {},
                    )
                    self._add_node(method_node)
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

            # Visit children
            for child in node.children:
                visit_node(child, parent_id)

        visit_node(root_node)

    def _extract_generic_graph(
        self,
        root_node,
        lines: list[str],
        file_path: str,
        file_node_id: str,
        language: Language,
    ) -> None:
        """Generic graph extraction for languages without specific handling."""
        # For unsupported languages, we just have the file node
        pass

    def _parse_simple(
        self,
        content: str,
        language: Language,
        file_path: str,
        file_node_id: str,
    ) -> None:
        """Simple fallback parsing without tree-sitter."""
        import re

        lines = content.splitlines()

        # Python pattern matching
        if language == Language.PYTHON:
            for i, line in enumerate(lines):
                # Match class definitions
                class_match = re.match(r"^class\s+(\w+)", line)
                if class_match:
                    name = class_match.group(1)
                    node_id = self._generate_node_id(name, NodeType.CLASS, file_path)
                    self._add_node(
                        GraphNode(
                            id=node_id,
                            name=name,
                            full_name=f"{file_path}:{name}",
                            node_type=NodeType.CLASS,
                            file_path=file_path,
                            start_line=i + 1,
                            language=language,
                        )
                    )
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

                # Match function definitions
                func_match = re.match(r"^def\s+(\w+)", line)
                if func_match:
                    name = func_match.group(1)
                    node_id = self._generate_node_id(name, NodeType.FUNCTION, file_path)
                    self._add_node(
                        GraphNode(
                            id=node_id,
                            name=name,
                            full_name=f"{file_path}:{name}",
                            node_type=NodeType.FUNCTION,
                            file_path=file_path,
                            start_line=i + 1,
                            language=language,
                        )
                    )
                    self._add_edge(file_node_id, node_id, EdgeType.CONTAINS)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            GraphNode or None if not found
        """
        return self._nodes.get(node_id)

    def get_related(
        self,
        entity_name: str,
        depth: int = 2,
        include_types: list[NodeType] | None = None,
    ) -> list[dict[str, Any]]:
        """Find entities related to a given entity.

        Uses BFS to traverse the graph up to specified depth.

        Args:
            entity_name: Name of the entity to find relations for
            depth: Maximum traversal depth
            include_types: Only include these node types (None for all)

        Returns:
            List of related entity dictionaries
        """
        graph = self._get_graph()
        nx = self._get_networkx()

        # Find nodes matching the entity name
        matching_nodes = [nid for nid, node in self._nodes.items() if node.name == entity_name]

        if not matching_nodes:
            return []

        related = set()

        for start_node in matching_nodes:
            # Get successors (what this node depends on)
            try:
                successors = nx.single_source_shortest_path_length(graph, start_node, cutoff=depth)
                related.update(successors.keys())
            except Exception:
                pass

            # Get predecessors (what depends on this node)
            try:
                predecessors = nx.single_source_shortest_path_length(
                    graph.reverse(), start_node, cutoff=depth
                )
                related.update(predecessors.keys())
            except Exception:
                pass

        # Filter and convert to dict
        results = []
        for node_id in related:
            node = self._nodes.get(node_id)
            if node:
                if include_types and node.node_type not in include_types:
                    continue
                results.append(node.to_dict())

        return results

    def get_file_context(self, file_path: str) -> dict[str, Any]:
        """Get comprehensive context for a file.

        Args:
            file_path: Path to the file (relative to project root)

        Returns:
            Dictionary with file context
        """
        graph = self._get_graph()

        # Find the file node
        file_node_id = self._generate_node_id(file_path, NodeType.FILE)
        if file_node_id not in self._nodes:
            return {
                "file_path": file_path,
                "imports": [],
                "exports": [],
                "dependencies": [],
                "entities": [],
            }

        context: dict[str, Any] = {
            "file_path": file_path,
            "imports": [],
            "exports": [],
            "dependencies": [],
            "entities": [],
        }

        # Get all edges from this file
        for _, target, data in graph.out_edges(file_node_id, data=True):
            target_node = self._nodes.get(target)
            if not target_node:
                continue

            edge_type = data.get("edge_type")

            if edge_type == EdgeType.IMPORTS.value:
                context["imports"].append(target_node.name)
            elif edge_type == EdgeType.CONTAINS.value:
                context["entities"].append(
                    {
                        "name": target_node.name,
                        "type": target_node.node_type.value,
                        "line": target_node.start_line,
                    }
                )

        # Get files that import this one
        for source, _, data in graph.in_edges(file_node_id, data=True):
            source_node = self._nodes.get(source)
            if source_node and source_node.node_type == NodeType.FILE:
                context["dependencies"].append(source_node.name)

        return context

    def get_callers(self, func_name: str) -> list[dict[str, Any]]:
        """Find functions that call a given function.

        Args:
            func_name: Name of the function

        Returns:
            List of caller dictionaries
        """
        graph = self._get_graph()

        # Find the function node
        func_nodes = [
            nid
            for nid, node in self._nodes.items()
            if node.name == func_name and node.node_type in (NodeType.FUNCTION, NodeType.METHOD)
        ]

        callers = []
        for func_node_id in func_nodes:
            for source, _, data in graph.in_edges(func_node_id, data=True):
                if data.get("edge_type") == EdgeType.CALLS.value:
                    caller = self._nodes.get(source)
                    if caller:
                        callers.append(caller.to_dict())

        return callers

    def get_dependencies(self, file_path: str) -> list[str]:
        """Get files that the given file depends on.

        Args:
            file_path: Path to the file

        Returns:
            List of dependency file paths
        """
        graph = self._get_graph()
        file_node_id = self._generate_node_id(file_path, NodeType.FILE)

        if file_node_id not in self._nodes:
            return []

        deps = []
        for _, target, data in graph.out_edges(file_node_id, data=True):
            target_node = self._nodes.get(target)
            if target_node and target_node.node_type == NodeType.IMPORT:
                deps.append(target_node.name)

        return deps

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        graph = self._get_graph()

        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "node_types": type_counts,
        }
