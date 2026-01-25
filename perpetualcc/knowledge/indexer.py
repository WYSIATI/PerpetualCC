"""Codebase indexer for the Knowledge Engine.

Scans a codebase to extract code entities and generate text chunks
suitable for embedding and retrieval. Uses tree-sitter for accurate
AST parsing across multiple languages.

Key responsibilities:
- Walking the codebase respecting .gitignore
- Parsing source files with tree-sitter
- Extracting meaningful code chunks (functions, classes, etc.)
- Generating metadata for each chunk
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Type of code chunk."""

    FILE = "file"  # Entire file (for small files)
    MODULE = "module"  # Module/package docstring
    CLASS = "class"  # Class definition
    FUNCTION = "function"  # Function/method definition
    IMPORT = "import"  # Import statements block
    COMMENT = "comment"  # Documentation comment
    CODE_BLOCK = "code_block"  # Generic code block


class Language(Enum):
    """Supported programming languages for parsing."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "c_sharp"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    LUA = "lua"
    BASH = "bash"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


# File extension to language mapping
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".rs": Language.RUST,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".lua": Language.LUA,
    ".sh": Language.BASH,
    ".bash": Language.BASH,
    ".zsh": Language.BASH,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".css": Language.CSS,
    ".scss": Language.CSS,
    ".sass": Language.CSS,
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".toml": Language.TOML,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
}


@dataclass(frozen=True)
class IndexerConfig:
    """Configuration for the codebase indexer.

    Attributes:
        max_chunk_tokens: Maximum tokens per chunk (approximate)
        min_chunk_tokens: Minimum tokens per chunk
        overlap_tokens: Token overlap between chunks
        include_comments: Whether to include comment blocks
        include_imports: Whether to include import sections
        respect_gitignore: Whether to respect .gitignore files
        additional_ignore_patterns: Extra patterns to ignore
        max_file_size_bytes: Maximum file size to process
    """

    max_chunk_tokens: int = 500
    min_chunk_tokens: int = 50
    overlap_tokens: int = 50
    include_comments: bool = True
    include_imports: bool = True
    respect_gitignore: bool = True
    additional_ignore_patterns: tuple[str, ...] = (
        "__pycache__",
        "*.pyc",
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "vendor",
        "venv",
        ".venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.min.js",
        "*.min.css",
        "*.map",
        "*.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Cargo.lock",
        "*.egg-info",
        ".eggs",
        ".tox",
        ".coverage",
        "htmlcov",
        ".idea",
        ".vscode",
        "*.log",
        "*.tmp",
        "*.temp",
    )
    max_file_size_bytes: int = 1_000_000  # 1MB


@dataclass
class CodeChunk:
    """A chunk of code extracted from the codebase.

    Attributes:
        id: Unique identifier for this chunk
        file_path: Path to the source file (relative to project root)
        language: Programming language
        chunk_type: Type of chunk (function, class, etc.)
        name: Name of the entity (function name, class name, etc.)
        content: The actual code content
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        docstring: Documentation string if available
        signature: Function/method signature if applicable
        parent_name: Name of parent class/module if applicable
        metadata: Additional metadata
    """

    id: str
    file_path: str
    language: Language
    chunk_type: ChunkType
    name: str
    content: str
    start_line: int
    end_line: int
    docstring: str | None = None
    signature: str | None = None
    parent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Creates a searchable text that includes:
        - File path and location
        - Entity type and name
        - Docstring (if available)
        - Code content

        Returns:
            Text suitable for embedding
        """
        parts = []

        # Add file context
        parts.append(f"File: {self.file_path}")

        # Add entity info
        type_str = self.chunk_type.value.replace("_", " ").title()
        if self.parent_name:
            parts.append(f"{type_str}: {self.parent_name}.{self.name}")
        else:
            parts.append(f"{type_str}: {self.name}")

        # Add signature if available
        if self.signature:
            parts.append(f"Signature: {self.signature}")

        # Add docstring if available
        if self.docstring:
            parts.append(f"Description: {self.docstring}")

        # Add code content
        parts.append(f"Code:\n{self.content}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "language": self.language.value,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "signature": self.signature,
            "parent_name": self.parent_name,
            "metadata": self.metadata,
        }


@dataclass
class CodeFile:
    """Represents a parsed code file.

    Attributes:
        path: Path to the file (relative to project root)
        absolute_path: Absolute path to the file
        language: Programming language
        chunks: Extracted code chunks
        imports: List of imports found
        total_lines: Total number of lines
        metadata: Additional file metadata
    """

    path: str
    absolute_path: Path
    language: Language
    chunks: list[CodeChunk]
    imports: list[str] = field(default_factory=list)
    total_lines: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GitignoreParser:
    """Parser for .gitignore files."""

    def __init__(self, project_root: Path):
        """Initialize with project root."""
        self.project_root = project_root
        self.patterns: list[str] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        self.patterns.append(line)
            except Exception as e:
                logger.warning("Failed to read .gitignore: %s", e)

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on gitignore patterns.

        Args:
            path: Path to check (relative to project root)

        Returns:
            True if path should be ignored
        """
        path_str = str(path)
        path_parts = path.parts

        for pattern in self.patterns:
            # Handle negation patterns
            if pattern.startswith("!"):
                continue  # Simplified: skip negation

            # Handle directory-only patterns
            is_dir_pattern = pattern.endswith("/")
            if is_dir_pattern:
                pattern = pattern[:-1]

            # Handle patterns starting with /
            if pattern.startswith("/"):
                pattern = pattern[1:]
                # Match only from root
                if fnmatch.fnmatch(path_str, pattern):
                    return True
            else:
                # Match anywhere in path
                if fnmatch.fnmatch(path_str, pattern):
                    return True
                if fnmatch.fnmatch(path_str, f"**/{pattern}"):
                    return True
                # Check each path component
                for part in path_parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True

        return False


class CodebaseIndexer:
    """Indexes a codebase for the Knowledge Engine.

    Walks the codebase, parses source files using tree-sitter,
    and extracts meaningful code chunks for embedding.
    """

    def __init__(
        self,
        project_path: str | Path,
        config: IndexerConfig | None = None,
    ):
        """Initialize the codebase indexer.

        Args:
            project_path: Path to the project root
            config: Indexer configuration
        """
        self.project_path = Path(project_path).resolve()
        self.config = config or IndexerConfig()
        self._gitignore = (
            GitignoreParser(self.project_path) if self.config.respect_gitignore else None
        )
        self._parser_cache: dict[Language, Any] = {}
        self._chunk_counter = 0

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Absolute path to check

        Returns:
            True if path should be ignored
        """
        # Resolve both paths to handle macOS symlinks consistently
        # (e.g., /var/folders vs /private/var/folders)
        resolved_path = path.resolve()
        resolved_project = self.project_path.resolve()

        try:
            relative_path = resolved_path.relative_to(resolved_project)
        except ValueError:
            # Paths may still differ due to symlinks, try realpath
            real_path = Path(os.path.realpath(path))
            real_project = Path(os.path.realpath(self.project_path))
            try:
                relative_path = real_path.relative_to(real_project)
            except ValueError:
                return True

        # Check additional ignore patterns
        path_str = str(relative_path)
        for pattern in self.config.additional_ignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True
            # Check if any part of the path matches
            for part in relative_path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        # Check gitignore
        if self._gitignore and self._gitignore.should_ignore(relative_path):
            return True

        return False

    def _get_language(self, file_path: Path) -> Language:
        """Determine the language of a file.

        Args:
            file_path: Path to the file

        Returns:
            Language enum value
        """
        suffix = file_path.suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(suffix, Language.UNKNOWN)

    def _generate_chunk_id(self, file_path: str, name: str, chunk_type: ChunkType) -> str:
        """Generate a unique chunk ID.

        Args:
            file_path: Relative file path
            name: Entity name
            chunk_type: Type of chunk

        Returns:
            Unique chunk identifier
        """
        self._chunk_counter += 1
        # Create a deterministic ID based on content
        base = f"{file_path}:{chunk_type.value}:{name}"
        return f"chunk_{self._chunk_counter}_{hash(base) & 0xFFFFFFFF:08x}"

    def scan_files(self) -> Iterator[Path]:
        """Scan the project for source files.

        Yields:
            Paths to source files to process
        """
        for path in self.project_path.rglob("*"):
            if not path.is_file():
                continue

            if self._should_ignore(path):
                continue

            # Check file size
            try:
                if path.stat().st_size > self.config.max_file_size_bytes:
                    logger.debug("Skipping large file: %s", path)
                    continue
            except OSError:
                continue

            # Check if we can process this file type
            language = self._get_language(path)
            if language == Language.UNKNOWN:
                continue

            yield path

    def _get_parser(self, language: Language):
        """Get or create a tree-sitter parser for the language.

        Args:
            language: Programming language

        Returns:
            Tree-sitter parser or None if unavailable
        """
        if language in self._parser_cache:
            return self._parser_cache[language]

        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            logger.warning(
                "tree-sitter-language-pack not available. "
                "Install with: pip install tree-sitter-language-pack"
            )
            return None

        try:
            # Map our language enum to tree-sitter-language-pack names
            lang_name = language.value
            parser = get_parser(lang_name)
            self._parser_cache[language] = parser
            return parser
        except Exception as e:
            logger.debug("Parser not available for %s: %s", language.value, e)
            self._parser_cache[language] = None
            return None

    def _parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a single file and extract code chunks.

        Args:
            file_path: Path to the file

        Returns:
            List of code chunks extracted from the file
        """
        language = self._get_language(file_path)
        # Handle macOS symlinks consistently
        real_file = Path(os.path.realpath(file_path))
        real_project = Path(os.path.realpath(self.project_path))
        relative_path = str(real_file.relative_to(real_project))

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read file %s: %s", file_path, e)
            return []

        lines = content.splitlines()
        total_lines = len(lines)

        # Try tree-sitter parsing first
        parser = self._get_parser(language)
        if parser:
            chunks = self._parse_with_tree_sitter(
                parser, content, language, relative_path, total_lines
            )
            if chunks:
                return chunks

        # Fall back to simple chunking
        return self._parse_simple(content, language, relative_path, total_lines)

    def _parse_with_tree_sitter(
        self,
        parser,
        content: str,
        language: Language,
        file_path: str,
        total_lines: int,
    ) -> list[CodeChunk]:
        """Parse content using tree-sitter.

        Args:
            parser: Tree-sitter parser
            content: File content
            language: Programming language
            file_path: Relative file path
            total_lines: Total number of lines

        Returns:
            List of code chunks
        """
        chunks = []

        try:
            tree = parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node
        except Exception as e:
            logger.debug("Tree-sitter parsing failed for %s: %s", file_path, e)
            return []

        # Extract chunks based on language
        if language == Language.PYTHON:
            chunks.extend(self._extract_python_chunks(root_node, content, file_path))
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            chunks.extend(self._extract_js_ts_chunks(root_node, content, file_path))
        elif language == Language.RUST:
            chunks.extend(self._extract_rust_chunks(root_node, content, file_path))
        elif language == Language.GO:
            chunks.extend(self._extract_go_chunks(root_node, content, file_path))
        else:
            # Generic extraction for other languages
            chunks.extend(self._extract_generic_chunks(root_node, content, file_path, language))

        # If no specific chunks found, create a file-level chunk for small files
        if not chunks and total_lines <= 100:
            chunks.append(
                CodeChunk(
                    id=self._generate_chunk_id(file_path, file_path, ChunkType.FILE),
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.FILE,
                    name=file_path,
                    content=content,
                    start_line=1,
                    end_line=total_lines,
                )
            )

        return chunks

    def _extract_python_chunks(self, root_node, content: str, file_path: str) -> list[CodeChunk]:
        """Extract chunks from Python code.

        Args:
            root_node: Tree-sitter root node
            content: File content
            file_path: Relative file path

        Returns:
            List of code chunks
        """
        chunks = []
        lines = content.splitlines()

        def get_docstring(node) -> str | None:
            """Extract docstring from a function/class node."""
            for child in node.children:
                if child.type == "expression_statement":
                    for grandchild in child.children:
                        if grandchild.type == "string":
                            text = grandchild.text.decode("utf-8")
                            # Remove quotes
                            if text.startswith('"""') or text.startswith("'''"):
                                return text[3:-3].strip()
                            elif text.startswith('"') or text.startswith("'"):
                                return text[1:-1].strip()
            return None

        def get_signature(node) -> str | None:
            """Extract function signature."""
            if node.type == "function_definition":
                # Find the name and parameters
                name = None
                params = None
                return_type = None

                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                    elif child.type == "parameters":
                        params = child.text.decode("utf-8")
                    elif child.type == "type":
                        return_type = child.text.decode("utf-8")

                if name and params:
                    sig = f"def {name}{params}"
                    if return_type:
                        sig += f" -> {return_type}"
                    return sig
            return None

        def visit_node(node, parent_class: str | None = None):
            """Recursively visit nodes to extract chunks."""
            if node.type == "class_definition":
                # Extract class name
                class_name = None
                for child in node.children:
                    if child.type == "identifier":
                        class_name = child.text.decode("utf-8")
                        break

                if class_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    docstring = get_docstring(node)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, class_name, ChunkType.CLASS),
                            file_path=file_path,
                            language=Language.PYTHON,
                            chunk_type=ChunkType.CLASS,
                            name=class_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=docstring,
                        )
                    )

                    # Visit class body for methods
                    for child in node.children:
                        if child.type == "block":
                            for stmt in child.children:
                                visit_node(stmt, class_name)

            elif node.type == "function_definition":
                # Extract function name
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    docstring = get_docstring(node)
                    signature = get_signature(node)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, func_name, ChunkType.FUNCTION),
                            file_path=file_path,
                            language=Language.PYTHON,
                            chunk_type=ChunkType.FUNCTION,
                            name=func_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=docstring,
                            signature=signature,
                            parent_name=parent_class,
                        )
                    )

            # Visit children for top-level nodes
            if parent_class is None:
                for child in node.children:
                    if child.type in ("class_definition", "function_definition"):
                        visit_node(child)

        visit_node(root_node)
        return chunks

    def _extract_js_ts_chunks(self, root_node, content: str, file_path: str) -> list[CodeChunk]:
        """Extract chunks from JavaScript/TypeScript code.

        Args:
            root_node: Tree-sitter root node
            content: File content
            file_path: Relative file path

        Returns:
            List of code chunks
        """
        chunks = []
        lines = content.splitlines()
        language = (
            Language.TYPESCRIPT if file_path.endswith((".ts", ".tsx")) else Language.JAVASCRIPT
        )

        def get_jsdoc(node, lines: list[str]) -> str | None:
            """Extract JSDoc comment before a node."""
            start_line = node.start_point[0]
            if start_line > 0:
                prev_line = lines[start_line - 1].strip()
                if prev_line.endswith("*/"):
                    # Find start of JSDoc
                    for i in range(start_line - 1, max(0, start_line - 20), -1):
                        if "/**" in lines[i]:
                            return "\n".join(lines[i:start_line])
            return None

        def visit_node(node, parent_class: str | None = None):
            """Recursively visit nodes."""
            if node.type in ("class_declaration", "class"):
                class_name = None
                for child in node.children:
                    if child.type == "identifier":
                        class_name = child.text.decode("utf-8")
                        break

                if class_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    jsdoc = get_jsdoc(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, class_name, ChunkType.CLASS),
                            file_path=file_path,
                            language=language,
                            chunk_type=ChunkType.CLASS,
                            name=class_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=jsdoc,
                        )
                    )

                    # Visit class body
                    for child in node.children:
                        if child.type == "class_body":
                            for stmt in child.children:
                                visit_node(stmt, class_name)

            elif node.type in (
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function",
            ):
                func_name = None

                # Try to get function name
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
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    jsdoc = get_jsdoc(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, func_name, ChunkType.FUNCTION),
                            file_path=file_path,
                            language=language,
                            chunk_type=ChunkType.FUNCTION,
                            name=func_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=jsdoc,
                            parent_name=parent_class,
                        )
                    )

            # Continue visiting children
            for child in node.children:
                visit_node(child, parent_class)

        visit_node(root_node)
        return chunks

    def _extract_rust_chunks(self, root_node, content: str, file_path: str) -> list[CodeChunk]:
        """Extract chunks from Rust code."""
        chunks = []
        lines = content.splitlines()

        def get_doc_comment(node, lines: list[str]) -> str | None:
            """Extract /// or //! doc comments."""
            start_line = node.start_point[0]
            doc_lines = []
            for i in range(start_line - 1, max(0, start_line - 50), -1):
                line = lines[i].strip()
                if line.startswith("///") or line.startswith("//!"):
                    doc_lines.insert(0, line[3:].strip())
                elif not line or line.startswith("//"):
                    continue
                else:
                    break
            return "\n".join(doc_lines) if doc_lines else None

        def visit_node(node, parent_impl: str | None = None):
            if node.type in ("struct_item", "enum_item", "trait_item"):
                name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name = child.text.decode("utf-8")
                        break

                if name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    doc = get_doc_comment(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, name, ChunkType.CLASS),
                            file_path=file_path,
                            language=Language.RUST,
                            chunk_type=ChunkType.CLASS,
                            name=name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=doc,
                        )
                    )

            elif node.type == "function_item":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    doc = get_doc_comment(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, func_name, ChunkType.FUNCTION),
                            file_path=file_path,
                            language=Language.RUST,
                            chunk_type=ChunkType.FUNCTION,
                            name=func_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=doc,
                            parent_name=parent_impl,
                        )
                    )

            elif node.type == "impl_item":
                # Get the type being implemented
                impl_type = None
                for child in node.children:
                    if child.type == "type_identifier":
                        impl_type = child.text.decode("utf-8")
                        break

                # Visit impl body
                for child in node.children:
                    if child.type == "declaration_list":
                        for item in child.children:
                            visit_node(item, impl_type)
                return

            for child in node.children:
                visit_node(child, parent_impl)

        visit_node(root_node)
        return chunks

    def _extract_go_chunks(self, root_node, content: str, file_path: str) -> list[CodeChunk]:
        """Extract chunks from Go code."""
        chunks = []
        lines = content.splitlines()

        def get_go_doc(node, lines: list[str]) -> str | None:
            """Extract Go doc comments."""
            start_line = node.start_point[0]
            doc_lines = []
            for i in range(start_line - 1, max(0, start_line - 50), -1):
                line = lines[i].strip()
                if line.startswith("//"):
                    doc_lines.insert(0, line[2:].strip())
                elif not line:
                    continue
                else:
                    break
            return "\n".join(doc_lines) if doc_lines else None

        def visit_node(node, receiver_type: str | None = None):
            if node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name = None
                        for subchild in child.children:
                            if subchild.type == "type_identifier":
                                name = subchild.text.decode("utf-8")
                                break

                        if name:
                            start_line = node.start_point[0] + 1
                            end_line = node.end_point[0] + 1
                            doc = get_go_doc(node, lines)

                            chunk_content = "\n".join(lines[start_line - 1 : end_line])
                            chunks.append(
                                CodeChunk(
                                    id=self._generate_chunk_id(file_path, name, ChunkType.CLASS),
                                    file_path=file_path,
                                    language=Language.GO,
                                    chunk_type=ChunkType.CLASS,
                                    name=name,
                                    content=chunk_content,
                                    start_line=start_line,
                                    end_line=end_line,
                                    docstring=doc,
                                )
                            )

            elif node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        break

                if func_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    doc = get_go_doc(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, func_name, ChunkType.FUNCTION),
                            file_path=file_path,
                            language=Language.GO,
                            chunk_type=ChunkType.FUNCTION,
                            name=func_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=doc,
                        )
                    )

            elif node.type == "method_declaration":
                func_name = None
                receiver = None

                for child in node.children:
                    if child.type == "field_identifier":
                        func_name = child.text.decode("utf-8")
                    elif child.type == "parameter_list":
                        # First parameter list is receiver
                        for param in child.children:
                            if param.type == "parameter_declaration":
                                for p in param.children:
                                    if p.type == "type_identifier":
                                        receiver = p.text.decode("utf-8")
                                        break

                if func_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    doc = get_go_doc(node, lines)

                    chunk_content = "\n".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, func_name, ChunkType.FUNCTION),
                            file_path=file_path,
                            language=Language.GO,
                            chunk_type=ChunkType.FUNCTION,
                            name=func_name,
                            content=chunk_content,
                            start_line=start_line,
                            end_line=end_line,
                            docstring=doc,
                            parent_name=receiver,
                        )
                    )

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return chunks

    def _extract_generic_chunks(
        self, root_node, content: str, file_path: str, language: Language
    ) -> list[CodeChunk]:
        """Generic chunk extraction for languages without specific handling.

        Falls back to simple text chunking based on blank lines and size.
        """
        chunks = []
        lines = content.splitlines()

        # For small files, just create a single chunk
        if len(lines) <= 100:
            return [
                CodeChunk(
                    id=self._generate_chunk_id(file_path, file_path, ChunkType.FILE),
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.FILE,
                    name=file_path,
                    content=content,
                    start_line=1,
                    end_line=len(lines),
                )
            ]

        # Split by blank lines into logical sections
        current_chunk_lines: list[str] = []
        current_start = 1
        blank_count = 0

        for i, line in enumerate(lines, 1):
            if not line.strip():
                blank_count += 1
                if blank_count >= 2 and len(current_chunk_lines) > 10:
                    # Create a chunk
                    chunk_content = "\n".join(current_chunk_lines)
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(
                                file_path, f"block_{current_start}", ChunkType.CODE_BLOCK
                            ),
                            file_path=file_path,
                            language=language,
                            chunk_type=ChunkType.CODE_BLOCK,
                            name=f"lines_{current_start}-{i - blank_count}",
                            content=chunk_content,
                            start_line=current_start,
                            end_line=i - blank_count,
                        )
                    )
                    current_chunk_lines = []
                    current_start = i + 1
                    blank_count = 0
            else:
                blank_count = 0
                current_chunk_lines.append(line)

        # Add remaining lines
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                CodeChunk(
                    id=self._generate_chunk_id(
                        file_path, f"block_{current_start}", ChunkType.CODE_BLOCK
                    ),
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.CODE_BLOCK,
                    name=f"lines_{current_start}-{len(lines)}",
                    content=chunk_content,
                    start_line=current_start,
                    end_line=len(lines),
                )
            )

        return chunks

    def _parse_simple(
        self, content: str, language: Language, file_path: str, total_lines: int
    ) -> list[CodeChunk]:
        """Simple fallback parsing without tree-sitter.

        Uses regex-based extraction for common patterns.
        """
        chunks = []
        lines = content.splitlines()

        # For small files, return single chunk
        if total_lines <= 100:
            return [
                CodeChunk(
                    id=self._generate_chunk_id(file_path, file_path, ChunkType.FILE),
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.FILE,
                    name=file_path,
                    content=content,
                    start_line=1,
                    end_line=total_lines,
                )
            ]

        # Try to find function/class definitions with regex
        if language == Language.PYTHON:
            # Match Python functions and classes
            pattern = r"^(class|def)\s+(\w+)"
            for i, line in enumerate(lines):
                match = re.match(pattern, line)
                if match:
                    kind = match.group(1)
                    name = match.group(2)
                    chunk_type = ChunkType.CLASS if kind == "class" else ChunkType.FUNCTION

                    # Find end of block (next def/class at same indentation or end)
                    indent = len(line) - len(line.lstrip())
                    end_line = i + 1
                    for j in range(i + 1, min(i + 200, total_lines)):
                        if lines[j].strip() and not lines[j].startswith(" " * (indent + 1)):
                            if lines[j].startswith(" " * indent) and re.match(
                                pattern, lines[j].lstrip()
                            ):
                                break
                        end_line = j + 1

                    chunk_content = "\n".join(lines[i:end_line])
                    chunks.append(
                        CodeChunk(
                            id=self._generate_chunk_id(file_path, name, chunk_type),
                            file_path=file_path,
                            language=language,
                            chunk_type=chunk_type,
                            name=name,
                            content=chunk_content,
                            start_line=i + 1,
                            end_line=end_line,
                        )
                    )

        # If no chunks found, split by size
        if not chunks:
            chunk_size = 50  # lines per chunk
            for i in range(0, total_lines, chunk_size):
                end = min(i + chunk_size, total_lines)
                chunk_content = "\n".join(lines[i:end])
                chunks.append(
                    CodeChunk(
                        id=self._generate_chunk_id(
                            file_path, f"block_{i + 1}", ChunkType.CODE_BLOCK
                        ),
                        file_path=file_path,
                        language=language,
                        chunk_type=ChunkType.CODE_BLOCK,
                        name=f"lines_{i + 1}-{end}",
                        content=chunk_content,
                        start_line=i + 1,
                        end_line=end,
                    )
                )

        return chunks

    def index_file(self, file_path: Path) -> CodeFile | None:
        """Index a single file.

        Args:
            file_path: Path to the file

        Returns:
            CodeFile with extracted chunks, or None if failed
        """
        if not file_path.is_file():
            return None

        if self._should_ignore(file_path):
            return None

        language = self._get_language(file_path)
        if language == Language.UNKNOWN:
            return None

        try:
            # Handle macOS symlinks consistently
            real_file = Path(os.path.realpath(file_path))
            real_project = Path(os.path.realpath(self.project_path))
            relative_path = str(real_file.relative_to(real_project))
            content = file_path.read_text(encoding="utf-8", errors="replace")
            total_lines = len(content.splitlines())

            chunks = self._parse_file(file_path)

            return CodeFile(
                path=relative_path,
                absolute_path=file_path,
                language=language,
                chunks=chunks,
                total_lines=total_lines,
            )
        except Exception as e:
            logger.warning("Failed to index file %s: %s", file_path, e)
            return None

    def index_project(self) -> Iterator[CodeFile]:
        """Index the entire project.

        Yields:
            CodeFile objects for each processed file
        """
        logger.info("Indexing project: %s", self.project_path)
        file_count = 0
        chunk_count = 0

        for file_path in self.scan_files():
            code_file = self.index_file(file_path)
            if code_file:
                file_count += 1
                chunk_count += len(code_file.chunks)
                yield code_file

        logger.info(
            "Indexed %d files, extracted %d chunks",
            file_count,
            chunk_count,
        )

    def get_all_chunks(self) -> list[CodeChunk]:
        """Get all code chunks from the project.

        Returns:
            List of all extracted code chunks
        """
        chunks = []
        for code_file in self.index_project():
            chunks.extend(code_file.chunks)
        return chunks
