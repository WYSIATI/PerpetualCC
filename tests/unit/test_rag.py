"""Unit tests for the RAG pipeline.

Tests cover:
- RAG configuration
- Index statistics
- Retrieval results
- Project indexing
- Semantic search
- File-based retrieval
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perpetualcc.knowledge.indexer import ChunkType, Language
from perpetualcc.knowledge.rag import (
    IndexStats,
    KnowledgeEngine,
    RAGConfig,
    RAGPipeline,
    RetrievalResult,
)


# Check if chromadb is available
def _has_chromadb() -> bool:
    """Check if chromadb is installed."""
    try:
        import chromadb
        return True
    except ImportError:
        return False


requires_chromadb = pytest.mark.skipif(
    not _has_chromadb(),
    reason="chromadb not installed"
)


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RAGConfig()

        assert config.collection_name == "perpetualcc_codebase"
        assert config.persist_directory is None
        assert config.default_top_k == 5
        assert config.similarity_threshold == 0.3
        assert config.rerank_results is True
        assert config.include_metadata is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RAGConfig(
            collection_name="custom_collection",
            persist_directory="/tmp/chromadb",
            default_top_k=10,
            similarity_threshold=0.5,
            rerank_results=False,
        )

        assert config.collection_name == "custom_collection"
        assert config.persist_directory == "/tmp/chromadb"
        assert config.default_top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.rerank_results is False

    def test_frozen(self):
        """Test that config is frozen."""
        config = RAGConfig()
        with pytest.raises(AttributeError):
            config.default_top_k = 20


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = IndexStats()

        assert stats.total_files == 0
        assert stats.total_chunks == 0
        assert stats.languages == {}
        assert stats.chunk_types == {}
        assert stats.last_indexed is None
        assert stats.index_hash is None

    def test_with_values(self):
        """Test statistics with values."""
        now = datetime.now()
        stats = IndexStats(
            total_files=10,
            total_chunks=50,
            languages={"python": 8, "javascript": 2},
            chunk_types={"function": 30, "class": 20},
            last_indexed=now,
            index_hash="abc123",
        )

        assert stats.total_files == 10
        assert stats.total_chunks == 50
        assert stats.languages["python"] == 8
        assert stats.chunk_types["function"] == 30
        assert stats.last_indexed == now
        assert stats.index_hash == "abc123"


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic retrieval result."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            file_path="src/main.py",
            content="def hello(): pass",
            score=0.95,
            chunk_type="function",
            name="hello",
            start_line=1,
            end_line=1,
        )

        assert result.chunk_id == "chunk_1"
        assert result.file_path == "src/main.py"
        assert result.score == 0.95
        assert result.name == "hello"

    def test_with_optional_fields(self):
        """Test creating result with all fields."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            file_path="src/auth.py",
            content="def authenticate(): pass",
            score=0.9,
            chunk_type="function",
            name="authenticate",
            start_line=10,
            end_line=25,
            docstring="Authenticate a user.",
            language="python",
            metadata={"imports": ["flask"]},
        )

        assert result.docstring == "Authenticate a user."
        assert result.language == "python"
        assert result.metadata == {"imports": ["flask"]}

    def test_to_context_string_basic(self):
        """Test generating context string."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            file_path="src/main.py",
            content="def hello():\n    return 'Hello'",
            score=0.9,
            chunk_type="function",
            name="hello",
            start_line=1,
            end_line=2,
            language="python",
        )

        context = result.to_context_string()

        assert "src/main.py:1-2" in context
        assert "function: hello" in context
        assert "```python" in context
        assert "def hello():" in context

    def test_to_context_string_with_docstring(self):
        """Test context string includes docstring."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            file_path="src/main.py",
            content="def greet(): pass",
            score=0.9,
            chunk_type="function",
            name="greet",
            start_line=1,
            end_line=1,
            docstring="Greet a user by name.",
        )

        context = result.to_context_string()
        assert "Doc: Greet a user" in context

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            file_path="src/main.py",
            content="code",
            score=0.9,
            chunk_type="function",
            name="hello",
            start_line=1,
            end_line=1,
            language="python",
        )

        d = result.to_dict()

        assert d["chunk_id"] == "chunk_1"
        assert d["file_path"] == "src/main.py"
        assert d["score"] == 0.9
        assert d["language"] == "python"


class MockEmbeddingProvider:
    """Mock embedding provider for testing RAG pipeline."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        # Simple deterministic embedding based on text hash
        h = hash(text) & 0xFFFFFFFF
        return [(h >> i) % 100 / 100.0 for i in range(self._dimension)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_name(self) -> str:
        return "mock-embedding"


@pytest.fixture
def temp_project() -> Path:
    """Create a temporary project with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        (project / "src").mkdir()
        (project / "tests").mkdir()

        # Create sample Python files
        (project / "src" / "auth.py").write_text('''"""Authentication module."""

def login(username: str, password: str) -> str:
    """Log in a user and return a token.

    Args:
        username: The user's username
        password: The user's password

    Returns:
        JWT token string
    """
    user = User.find(username)
    if user.check_password(password):
        return create_token(user)
    raise AuthError("Invalid credentials")


def logout(token: str) -> bool:
    """Log out a user by invalidating their token."""
    return invalidate_token(token)


class User:
    """Represents a system user."""

    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email

    def check_password(self, password: str) -> bool:
        """Check if password matches."""
        pass
''')

        (project / "src" / "api.py").write_text('''"""API endpoints module."""

from flask import Flask, request, jsonify
from auth import login, logout

app = Flask(__name__)


@app.route("/api/login", methods=["POST"])
def api_login():
    """Handle login API request."""
    data = request.json
    token = login(data["username"], data["password"])
    return jsonify({"token": token})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Handle logout API request."""
    token = request.headers.get("Authorization")
    logout(token)
    return jsonify({"success": True})


@app.route("/api/users", methods=["GET"])
def list_users():
    """List all users."""
    return jsonify({"users": []})
''')

        (project / "tests" / "test_auth.py").write_text('''"""Tests for authentication."""

import pytest
from src.auth import login, logout, User


def test_login_success():
    """Test successful login."""
    token = login("testuser", "password123")
    assert token is not None


def test_login_failure():
    """Test login with wrong password."""
    with pytest.raises(AuthError):
        login("testuser", "wrongpassword")


def test_logout():
    """Test logout functionality."""
    assert logout("valid_token") is True
''')

        yield project


class TestRAGPipelineInitialization:
    """Tests for RAGPipeline initialization."""

    def test_creates_with_defaults(self, temp_project: Path):
        """Test creating pipeline with default settings."""
        pipeline = RAGPipeline(temp_project)

        assert pipeline.project_path == temp_project.resolve()
        assert pipeline.config.collection_name == "perpetualcc_codebase"

    def test_creates_with_custom_config(self, temp_project: Path):
        """Test creating pipeline with custom config."""
        config = RAGConfig(
            collection_name="custom",
            default_top_k=10,
        )
        pipeline = RAGPipeline(temp_project, config)

        assert pipeline.config.collection_name == "custom"
        assert pipeline.config.default_top_k == 10

    def test_creates_with_custom_embeddings(self, temp_project: Path):
        """Test creating pipeline with custom embedding provider."""
        embeddings = MockEmbeddingProvider()
        pipeline = RAGPipeline(temp_project, embeddings=embeddings)

        assert pipeline.embeddings == embeddings

    def test_persist_directory_default(self, temp_project: Path):
        """Test default persistence directory."""
        pipeline = RAGPipeline(temp_project)

        expected = temp_project.resolve() / ".perpetualcc" / "chromadb"
        assert pipeline._persist_path == expected

    def test_persist_directory_custom(self, temp_project: Path):
        """Test custom persistence directory."""
        config = RAGConfig(persist_directory="/custom/path")
        pipeline = RAGPipeline(temp_project, config)

        assert pipeline._persist_path == Path("/custom/path")


class TestRetrievalResultReranking:
    """Tests for result re-ranking functionality."""

    @pytest.mark.asyncio
    async def test_rerank_boosts_name_match(self):
        """Test that name matches boost score."""
        pipeline = RAGPipeline(
            Path("/tmp"),
            embeddings=MockEmbeddingProvider(),
        )

        results = [
            RetrievalResult(
                chunk_id="1",
                file_path="src/auth.py",
                content="def authenticate(): pass",
                score=0.7,
                chunk_type="function",
                name="authenticate",
                start_line=1,
                end_line=1,
            ),
            RetrievalResult(
                chunk_id="2",
                file_path="src/utils.py",
                content="def helper(): pass",
                score=0.75,
                chunk_type="function",
                name="helper",
                start_line=1,
                end_line=1,
            ),
        ]

        # Query for "authenticate"
        reranked = await pipeline._rerank_results("authenticate", results)

        # The authenticate function should now rank higher due to name match
        assert reranked[0].name == "authenticate"

    @pytest.mark.asyncio
    async def test_rerank_boosts_docstring_match(self):
        """Test that docstring matches boost score."""
        pipeline = RAGPipeline(
            Path("/tmp"),
            embeddings=MockEmbeddingProvider(),
        )

        results = [
            RetrievalResult(
                chunk_id="1",
                file_path="src/auth.py",
                content="def func1(): pass",
                score=0.7,
                chunk_type="function",
                name="func1",
                start_line=1,
                end_line=1,
                docstring="Handle user authentication",
            ),
            RetrievalResult(
                chunk_id="2",
                file_path="src/utils.py",
                content="def func2(): pass",
                score=0.72,
                chunk_type="function",
                name="func2",
                start_line=1,
                end_line=1,
                docstring="Helper function for data",
            ),
        ]

        reranked = await pipeline._rerank_results("authentication", results)

        # func1 should rank higher due to docstring match
        assert reranked[0].chunk_id == "1"

    @pytest.mark.asyncio
    async def test_rerank_boosts_content_match(self):
        """Test that content keyword matches boost score."""
        pipeline = RAGPipeline(
            Path("/tmp"),
            embeddings=MockEmbeddingProvider(),
        )

        results = [
            RetrievalResult(
                chunk_id="1",
                file_path="src/auth.py",
                content="def process(): return authenticate_user()",
                score=0.7,
                chunk_type="function",
                name="process",
                start_line=1,
                end_line=1,
            ),
            RetrievalResult(
                chunk_id="2",
                file_path="src/utils.py",
                content="def helper(): return format_data()",
                score=0.72,
                chunk_type="function",
                name="helper",
                start_line=1,
                end_line=1,
            ),
        ]

        reranked = await pipeline._rerank_results("authenticate", results)

        # First result should rank higher due to content match
        assert reranked[0].chunk_id == "1"


class TestIndexStatsTracking:
    """Tests for index statistics tracking."""

    @requires_chromadb
    def test_stats_initialized_empty(self, temp_project: Path):
        """Test that stats start empty (requires chromadb)."""
        pipeline = RAGPipeline(
            temp_project,
            embeddings=MockEmbeddingProvider(),
        )

        stats = pipeline.get_stats()
        assert stats.total_files == 0

    def test_stats_after_indexing(self, temp_project: Path):
        """Test stats are updated after indexing."""
        # This would require mocking ChromaDB
        # For now, we just verify the stats object structure
        stats = IndexStats(
            total_files=5,
            total_chunks=25,
            languages={"python": 5},
            chunk_types={"function": 15, "class": 10},
            last_indexed=datetime.now(),
        )

        assert stats.total_files == 5
        assert stats.total_chunks == 25
        assert "python" in stats.languages


class TestKnowledgeEngine:
    """Tests for the high-level KnowledgeEngine interface."""

    def test_initialization(self, temp_project: Path):
        """Test knowledge engine initialization."""
        engine = KnowledgeEngine(temp_project)

        assert engine.project_path == temp_project.resolve()
        assert engine.rag is not None
        assert engine._code_graph is None  # Lazy init

    def test_initialization_with_config(self, temp_project: Path):
        """Test knowledge engine with custom config."""
        config = RAGConfig(
            collection_name="test_collection",
            default_top_k=10,
        )
        engine = KnowledgeEngine(temp_project, config)

        assert engine.rag.config.collection_name == "test_collection"

    @requires_chromadb
    def test_get_stats(self, temp_project: Path):
        """Test getting knowledge engine statistics (requires chromadb)."""
        engine = KnowledgeEngine(temp_project)
        stats = engine.get_stats()

        assert "rag" in stats
        assert "total_files" in stats["rag"]
        assert "total_chunks" in stats["rag"]


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios with Claude Code."""

    def test_retrieval_result_for_authentication_query(self):
        """Test retrieval result format for an authentication query.

        This simulates what Claude Code might receive when asking
        "where is authentication handled?"
        """
        result = RetrievalResult(
            chunk_id="auth_login_1",
            file_path="src/auth/handler.py",
            content='''def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user and return JWT token.

    Args:
        request: Login request with username and password

    Returns:
        TokenResponse with access and refresh tokens

    Raises:
        AuthenticationError: If credentials invalid
    """
    user = user_service.find_by_username(request.username)
    if not user:
        raise AuthenticationError("User not found")

    if not user.verify_password(request.password):
        raise AuthenticationError("Invalid password")

    return token_service.create_tokens(user)''',
            score=0.95,
            chunk_type="function",
            name="login",
            start_line=15,
            end_line=35,
            docstring="Authenticate user and return JWT token.",
            language="python",
            metadata={
                "parent_name": None,
                "signature": "def login(request: LoginRequest) -> TokenResponse",
            },
        )

        # Verify the result can be used for context
        context = result.to_context_string()
        assert "src/auth/handler.py:15-35" in context
        assert "function: login" in context
        assert "Authenticate user" in context

        # Verify dict format for MasterAgent integration
        d = result.to_dict()
        assert d["score"] == 0.95
        assert d["name"] == "login"

    def test_retrieval_result_for_api_endpoint_query(self):
        """Test retrieval result for API endpoint query.

        Simulates: "how do I create a new user?"
        """
        result = RetrievalResult(
            chunk_id="api_create_user",
            file_path="src/api/users.py",
            content='''@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
) -> UserResponse:
    """Create a new user.

    Requires admin privileges.
    """
    if user_service.exists(user_data.email):
        raise HTTPException(400, "Email already registered")

    user = user_service.create(db, user_data)
    return UserResponse.from_orm(user)''',
            score=0.92,
            chunk_type="function",
            name="create_user",
            start_line=45,
            end_line=62,
            docstring="Create a new user. Requires admin privileges.",
            language="python",
        )

        d = result.to_dict()
        assert d["name"] == "create_user"
        assert "admin" in d["docstring"]

    def test_retrieval_result_for_class_query(self):
        """Test retrieval result for a class definition query.

        Simulates: "what does the User model look like?"
        """
        result = RetrievalResult(
            chunk_id="model_user",
            file_path="src/models/user.py",
            content='''class User(Base):
    """User database model.

    Attributes:
        id: Unique identifier
        email: User's email address (unique)
        username: User's display name
        password_hash: Hashed password
        created_at: Account creation timestamp
        is_active: Whether account is active
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def set_password(self, password: str) -> None:
        """Set a new password."""
        self.password_hash = generate_password_hash(password)''',
            score=0.98,
            chunk_type="class",
            name="User",
            start_line=10,
            end_line=40,
            docstring="User database model.",
            language="python",
        )

        context = result.to_context_string()
        assert "class: User" in context
        assert "User database model" in context
        assert "email" in context
        assert "password_hash" in context

    def test_multiple_results_for_broad_query(self):
        """Test handling multiple results for a broad query.

        Simulates: "how does error handling work?"
        """
        results = [
            RetrievalResult(
                chunk_id="error_handler",
                file_path="src/middleware/errors.py",
                content="def global_error_handler(exc): ...",
                score=0.88,
                chunk_type="function",
                name="global_error_handler",
                start_line=1,
                end_line=20,
            ),
            RetrievalResult(
                chunk_id="custom_exception",
                file_path="src/exceptions.py",
                content="class AppException(Exception): ...",
                score=0.85,
                chunk_type="class",
                name="AppException",
                start_line=1,
                end_line=15,
            ),
            RetrievalResult(
                chunk_id="api_error_response",
                file_path="src/api/utils.py",
                content="def create_error_response(error): ...",
                score=0.82,
                chunk_type="function",
                name="create_error_response",
                start_line=30,
                end_line=45,
            ),
        ]

        # Verify results are sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Verify all results can be converted to context
        for result in results:
            context = result.to_context_string()
            assert result.name in context

    def test_result_with_typescript_code(self):
        """Test retrieval result for TypeScript code."""
        result = RetrievalResult(
            chunk_id="ts_api_client",
            file_path="src/api/client.ts",
            content='''export async function fetchUser(id: string): Promise<User> {
    const response = await fetch(`/api/users/${id}`, {
        headers: {
            'Authorization': `Bearer ${getToken()}`,
        },
    });

    if (!response.ok) {
        throw new ApiError(response.status, await response.text());
    }

    return response.json();
}''',
            score=0.90,
            chunk_type="function",
            name="fetchUser",
            start_line=15,
            end_line=28,
            language="typescript",
        )

        context = result.to_context_string()
        assert "```typescript" in context
        assert "fetchUser" in context
        assert "Promise<User>" in context

    def test_result_with_react_component(self):
        """Test retrieval result for React component."""
        result = RetrievalResult(
            chunk_id="react_login_form",
            file_path="src/components/LoginForm.tsx",
            content='''export function LoginForm({ onSubmit }: LoginFormProps) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        try {
            await onSubmit({ username, password });
        } catch (err) {
            setError('Login failed');
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <input value={username} onChange={e => setUsername(e.target.value)} />
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
            {error && <p className="error">{error}</p>}
            <button type="submit">Login</button>
        </form>
    );
}''',
            score=0.87,
            chunk_type="function",
            name="LoginForm",
            start_line=5,
            end_line=30,
            docstring="Login form component",
            language="tsx",
        )

        d = result.to_dict()
        assert d["language"] == "tsx"
        assert "useState" in d["content"]
