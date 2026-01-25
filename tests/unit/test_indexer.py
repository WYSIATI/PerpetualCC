"""Unit tests for the codebase indexer.

Tests cover:
- File scanning and filtering
- Language detection
- Code chunk extraction
- Gitignore handling
- Tree-sitter parsing (when available)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from perpetualcc.knowledge.indexer import (
    ChunkType,
    CodebaseIndexer,
    CodeChunk,
    CodeFile,
    GitignoreParser,
    IndexerConfig,
    Language,
    EXTENSION_TO_LANGUAGE,
)


# Check if tree-sitter is available
def _has_tree_sitter() -> bool:
    """Check if tree-sitter-language-pack is installed."""
    try:
        import tree_sitter_language_pack
        return True
    except ImportError:
        return False


requires_tree_sitter = pytest.mark.skipif(
    not _has_tree_sitter(),
    reason="tree-sitter-language-pack not installed"
)


@pytest.fixture
def temp_project() -> Path:
    """Create a temporary project directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        # Create directory structure
        (project / "src").mkdir()
        (project / "tests").mkdir()
        (project / "node_modules").mkdir()
        (project / ".git").mkdir()

        # Create Python files
        (project / "src" / "main.py").write_text('''"""Main module."""

def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        """Initialize the calculator."""
        self.result = 0

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        self.result = a + b
        return self.result

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        self.result = a - b
        return self.result


def _private_helper():
    """A private helper function."""
    pass
''')

        (project / "src" / "utils.py").write_text('''"""Utility functions."""

import os
from pathlib import Path
from typing import List, Optional

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    with open(path, 'r') as f:
        return f.read()


def list_files(directory: str) -> List[str]:
    """List all files in a directory."""
    return os.listdir(directory)
''')

        (project / "tests" / "test_main.py").write_text('''"""Tests for main module."""

import pytest
from src.main import greet, Calculator


def test_greet():
    """Test the greet function."""
    assert greet("World") == "Hello, World!"


class TestCalculator:
    """Tests for Calculator class."""

    def test_add(self):
        """Test addition."""
        calc = Calculator()
        assert calc.add(2, 3) == 5

    def test_subtract(self):
        """Test subtraction."""
        calc = Calculator()
        assert calc.subtract(5, 3) == 2
''')

        # Create JavaScript file
        (project / "src" / "app.js").write_text('''/**
 * Main application module.
 */

import { helper } from './helper.js';

/**
 * Process user input.
 * @param {string} input - The user input
 * @returns {string} The processed result
 */
function processInput(input) {
    return input.trim().toLowerCase();
}

/**
 * User class representing a system user.
 */
class User {
    /**
     * Create a new user.
     * @param {string} name - The user's name
     * @param {number} age - The user's age
     */
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    /**
     * Get the user's greeting.
     */
    getGreeting() {
        return `Hello, I am ${this.name}`;
    }
}

const arrowFunc = (x) => x * 2;

export { processInput, User, arrowFunc };
''')

        # Create TypeScript file
        (project / "src" / "types.ts").write_text('''/**
 * Type definitions for the application.
 */

interface UserConfig {
    name: string;
    email: string;
    preferences: {
        theme: 'light' | 'dark';
        notifications: boolean;
    };
}

type StatusCode = 200 | 400 | 404 | 500;

/**
 * API response wrapper.
 */
interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    statusCode: StatusCode;
}

/**
 * Create a success response.
 */
function createSuccessResponse<T>(data: T): ApiResponse<T> {
    return {
        success: true,
        data,
        statusCode: 200,
    };
}

export { UserConfig, ApiResponse, createSuccessResponse };
''')

        # Create files that should be ignored
        (project / "node_modules" / "lodash.js").write_text("// lodash")
        (project / ".git" / "config").write_text("[core]")

        # Create .gitignore
        (project / ".gitignore").write_text("""# Python
__pycache__
*.pyc

# Node
node_modules/

# Build
dist/
build/
""")

        yield project


@pytest.fixture
def indexer(temp_project: Path) -> CodebaseIndexer:
    """Create an indexer for the temp project."""
    return CodebaseIndexer(temp_project)


class TestLanguageDetection:
    """Tests for language detection from file extensions."""

    def test_python_detection(self):
        """Test Python file detection."""
        assert EXTENSION_TO_LANGUAGE[".py"] == Language.PYTHON
        assert EXTENSION_TO_LANGUAGE[".pyi"] == Language.PYTHON

    def test_javascript_detection(self):
        """Test JavaScript file detection."""
        assert EXTENSION_TO_LANGUAGE[".js"] == Language.JAVASCRIPT
        assert EXTENSION_TO_LANGUAGE[".mjs"] == Language.JAVASCRIPT
        assert EXTENSION_TO_LANGUAGE[".jsx"] == Language.JAVASCRIPT

    def test_typescript_detection(self):
        """Test TypeScript file detection."""
        assert EXTENSION_TO_LANGUAGE[".ts"] == Language.TYPESCRIPT
        assert EXTENSION_TO_LANGUAGE[".tsx"] == Language.TSX

    def test_rust_detection(self):
        """Test Rust file detection."""
        assert EXTENSION_TO_LANGUAGE[".rs"] == Language.RUST

    def test_go_detection(self):
        """Test Go file detection."""
        assert EXTENSION_TO_LANGUAGE[".go"] == Language.GO


class TestGitignoreParser:
    """Tests for gitignore parsing."""

    def test_loads_patterns(self, temp_project: Path):
        """Test that patterns are loaded from .gitignore."""
        parser = GitignoreParser(temp_project)
        assert len(parser.patterns) > 0
        assert "__pycache__" in parser.patterns
        assert "node_modules/" in parser.patterns

    def test_ignores_patterns(self, temp_project: Path):
        """Test that patterns are correctly matched."""
        parser = GitignoreParser(temp_project)

        # Should ignore
        assert parser.should_ignore(Path("__pycache__"))
        assert parser.should_ignore(Path("node_modules"))

        # Should not ignore
        assert not parser.should_ignore(Path("src/main.py"))
        assert not parser.should_ignore(Path("tests"))

    def test_handles_missing_gitignore(self, temp_project: Path):
        """Test behavior when .gitignore doesn't exist."""
        (temp_project / ".gitignore").unlink()
        parser = GitignoreParser(temp_project)
        assert parser.patterns == []


class TestFileScanning:
    """Tests for file scanning functionality."""

    def test_scans_source_files(self, indexer: CodebaseIndexer):
        """Test that source files are scanned."""
        files = list(indexer.scan_files())
        file_names = [f.name for f in files]

        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names
        assert "app.js" in file_names
        assert "types.ts" in file_names

    def test_ignores_node_modules(self, indexer: CodebaseIndexer):
        """Test that node_modules is ignored."""
        files = list(indexer.scan_files())
        file_paths = [str(f) for f in files]

        for path in file_paths:
            assert "node_modules" not in path

    def test_ignores_git_directory(self, indexer: CodebaseIndexer):
        """Test that .git is ignored."""
        files = list(indexer.scan_files())
        file_paths = [str(f) for f in files]

        for path in file_paths:
            assert ".git" not in path

    def test_custom_ignore_patterns(self, temp_project: Path):
        """Test custom ignore patterns."""
        config = IndexerConfig(
            additional_ignore_patterns=("*test*",)
        )
        indexer = CodebaseIndexer(temp_project, config)
        files = list(indexer.scan_files())
        file_names = [f.name for f in files]

        assert "test_main.py" not in file_names
        assert "main.py" in file_names


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_to_embedding_text(self):
        """Test embedding text generation."""
        chunk = CodeChunk(
            id="chunk_1",
            file_path="src/main.py",
            language=Language.PYTHON,
            chunk_type=ChunkType.FUNCTION,
            name="greet",
            content="def greet(name):\n    return f'Hello, {name}!'",
            start_line=1,
            end_line=2,
            docstring="Greet a person by name.",
            signature="def greet(name: str) -> str",
        )

        text = chunk.to_embedding_text()

        assert "src/main.py" in text
        assert "Function: greet" in text
        assert "Signature: def greet(name: str) -> str" in text
        assert "Description: Greet a person by name." in text
        assert "def greet(name)" in text

    def test_to_embedding_text_with_parent(self):
        """Test embedding text for method with parent class."""
        chunk = CodeChunk(
            id="chunk_2",
            file_path="src/main.py",
            language=Language.PYTHON,
            chunk_type=ChunkType.FUNCTION,
            name="add",
            content="def add(self, a, b):\n    return a + b",
            start_line=10,
            end_line=11,
            parent_name="Calculator",
        )

        text = chunk.to_embedding_text()
        assert "Calculator.add" in text

    def test_to_dict(self):
        """Test dictionary conversion."""
        chunk = CodeChunk(
            id="chunk_1",
            file_path="src/main.py",
            language=Language.PYTHON,
            chunk_type=ChunkType.FUNCTION,
            name="greet",
            content="def greet(name): pass",
            start_line=1,
            end_line=1,
        )

        d = chunk.to_dict()

        assert d["id"] == "chunk_1"
        assert d["file_path"] == "src/main.py"
        assert d["language"] == "python"
        assert d["chunk_type"] == "function"
        assert d["name"] == "greet"


@requires_tree_sitter
class TestPythonParsing:
    """Tests for Python file parsing (requires tree-sitter)."""

    def test_extracts_functions(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test function extraction from Python files."""
        code_file = indexer.index_file(temp_project / "src" / "main.py")
        assert code_file is not None

        chunk_names = [c.name for c in code_file.chunks]
        assert "greet" in chunk_names
        assert "_private_helper" in chunk_names

    def test_extracts_classes(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test class extraction from Python files."""
        code_file = indexer.index_file(temp_project / "src" / "main.py")
        assert code_file is not None

        class_chunks = [c for c in code_file.chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        calc_class = next((c for c in class_chunks if c.name == "Calculator"), None)
        assert calc_class is not None
        assert calc_class.docstring is not None

    def test_extracts_methods(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test method extraction from classes."""
        code_file = indexer.index_file(temp_project / "src" / "main.py")
        assert code_file is not None

        method_chunks = [
            c for c in code_file.chunks
            if c.chunk_type == ChunkType.FUNCTION and c.parent_name == "Calculator"
        ]

        method_names = [c.name for c in method_chunks]
        assert "add" in method_names
        assert "subtract" in method_names
        assert "__init__" in method_names

    def test_extracts_docstrings(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test docstring extraction."""
        code_file = indexer.index_file(temp_project / "src" / "main.py")
        assert code_file is not None

        greet_chunk = next((c for c in code_file.chunks if c.name == "greet"), None)
        assert greet_chunk is not None
        assert greet_chunk.docstring is not None
        assert "Greet a person" in greet_chunk.docstring

    def test_line_numbers(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test that line numbers are correctly captured."""
        code_file = indexer.index_file(temp_project / "src" / "main.py")
        assert code_file is not None

        greet_chunk = next((c for c in code_file.chunks if c.name == "greet"), None)
        assert greet_chunk is not None
        assert greet_chunk.start_line > 0
        assert greet_chunk.end_line >= greet_chunk.start_line


@requires_tree_sitter
class TestJavaScriptParsing:
    """Tests for JavaScript file parsing (requires tree-sitter)."""

    def test_extracts_functions(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test function extraction from JavaScript files."""
        code_file = indexer.index_file(temp_project / "src" / "app.js")
        assert code_file is not None
        assert code_file.language == Language.JAVASCRIPT

        func_chunks = [c for c in code_file.chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.name for c in func_chunks]
        assert "processInput" in func_names

    def test_extracts_classes(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test class extraction from JavaScript files."""
        code_file = indexer.index_file(temp_project / "src" / "app.js")
        assert code_file is not None

        class_chunks = [c for c in code_file.chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        user_class = next((c for c in class_chunks if c.name == "User"), None)
        assert user_class is not None


@requires_tree_sitter
class TestTypeScriptParsing:
    """Tests for TypeScript file parsing (requires tree-sitter)."""

    def test_extracts_functions(self, indexer: CodebaseIndexer, temp_project: Path):
        """Test function extraction from TypeScript files."""
        code_file = indexer.index_file(temp_project / "src" / "types.ts")
        assert code_file is not None
        assert code_file.language == Language.TYPESCRIPT

        func_chunks = [c for c in code_file.chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.name for c in func_chunks]
        assert "createSuccessResponse" in func_names


class TestProjectIndexing:
    """Tests for full project indexing."""

    def test_indexes_entire_project(self, indexer: CodebaseIndexer):
        """Test full project indexing."""
        files = list(indexer.index_project())

        assert len(files) >= 4  # At least our test files
        file_paths = [f.path for f in files]
        assert "src/main.py" in file_paths
        assert "src/utils.py" in file_paths
        assert "src/app.js" in file_paths

    @requires_tree_sitter
    def test_get_all_chunks(self, indexer: CodebaseIndexer):
        """Test getting all chunks from project (requires tree-sitter for FUNCTION chunks)."""
        chunks = indexer.get_all_chunks()

        assert len(chunks) > 0

        # Should include various chunk types
        chunk_types = set(c.chunk_type for c in chunks)
        assert ChunkType.FUNCTION in chunk_types

        # Should include multiple languages
        languages = set(c.language for c in chunks)
        assert Language.PYTHON in languages
        assert Language.JAVASCRIPT in languages


class TestIndexerConfig:
    """Tests for indexer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IndexerConfig()

        assert config.max_chunk_tokens == 500
        assert config.min_chunk_tokens == 50
        assert config.respect_gitignore is True
        assert "node_modules" in config.additional_ignore_patterns

    def test_custom_safe_directories(self, temp_project: Path):
        """Test custom safe directories configuration."""
        config = IndexerConfig(
            additional_ignore_patterns=()  # Don't ignore anything extra
        )
        indexer = CodebaseIndexer(temp_project, config)

        files = list(indexer.scan_files())
        assert len(files) > 0

    def test_max_file_size(self, temp_project: Path):
        """Test max file size filtering."""
        # Create a large file
        large_content = "x" * 2_000_000  # 2MB
        (temp_project / "src" / "large.py").write_text(large_content)

        config = IndexerConfig(max_file_size_bytes=1_000_000)
        indexer = CodebaseIndexer(temp_project, config)

        files = list(indexer.scan_files())
        file_names = [f.name for f in files]
        assert "large.py" not in file_names


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_file(self, temp_project: Path):
        """Test handling of empty files."""
        (temp_project / "src" / "empty.py").write_text("")

        indexer = CodebaseIndexer(temp_project)
        code_file = indexer.index_file(temp_project / "src" / "empty.py")

        # Should handle gracefully
        assert code_file is not None
        assert len(code_file.chunks) == 0 or (
            len(code_file.chunks) == 1 and code_file.chunks[0].content == ""
        )

    def test_binary_file(self, temp_project: Path):
        """Test handling of binary files."""
        (temp_project / "src" / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

        indexer = CodebaseIndexer(temp_project)
        files = list(indexer.scan_files())
        file_names = [f.name for f in files]

        # Binary files should be filtered out (unknown extension)
        assert "binary.bin" not in file_names

    def test_unicode_content(self, temp_project: Path):
        """Test handling of files with unicode content."""
        (temp_project / "src" / "unicode.py").write_text('''"""Module with unicode: 你好世界 🎉"""

def greet():
    """Greet in Chinese."""
    return "你好"
''')

        indexer = CodebaseIndexer(temp_project)
        code_file = indexer.index_file(temp_project / "src" / "unicode.py")

        assert code_file is not None
        assert len(code_file.chunks) > 0

    def test_syntax_error_file(self, temp_project: Path):
        """Test handling of files with syntax errors."""
        (temp_project / "src" / "broken.py").write_text('''def broken(
    # Missing closing paren and colon
    print("oops"
''')

        indexer = CodebaseIndexer(temp_project)
        code_file = indexer.index_file(temp_project / "src" / "broken.py")

        # Should handle gracefully, might use fallback parsing
        assert code_file is not None

    @requires_tree_sitter
    def test_deeply_nested_code(self, temp_project: Path):
        """Test handling of deeply nested code (requires tree-sitter)."""
        (temp_project / "src" / "nested.py").write_text('''class Outer:
    class Inner:
        class DeepInner:
            def deep_method(self):
                def local_func():
                    pass
                return local_func
''')

        indexer = CodebaseIndexer(temp_project)
        code_file = indexer.index_file(temp_project / "src" / "nested.py")

        assert code_file is not None
        # Should extract at least the outer class
        chunk_names = [c.name for c in code_file.chunks]
        assert "Outer" in chunk_names
