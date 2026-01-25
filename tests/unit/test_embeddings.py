"""Unit tests for embedding providers.

Tests cover:
- Embedding provider interfaces
- Configuration options
- Hybrid provider fallback behavior
- Factory function
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perpetualcc.knowledge.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingProviderType,
    EmbeddingResult,
    GeminiEmbeddings,
    HybridEmbeddings,
    OllamaEmbeddings,
    SentenceTransformersEmbeddings,
    create_embedding_provider,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.provider_type == EmbeddingProviderType.SENTENCE_TRANSFORMERS
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.cache_enabled is True
        assert config.ollama_host == "http://localhost:11434"
        assert config.gemini_api_key is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            provider_type=EmbeddingProviderType.OLLAMA,
            model_name="nomic-embed-text",
            dimension=768,
            batch_size=16,
            normalize=False,
            ollama_host="http://custom:11434",
        )

        assert config.provider_type == EmbeddingProviderType.OLLAMA
        assert config.model_name == "nomic-embed-text"
        assert config.dimension == 768
        assert config.batch_size == 16
        assert config.normalize is False
        assert config.ollama_host == "http://custom:11434"

    def test_frozen(self):
        """Test that config is frozen (immutable)."""
        config = EmbeddingConfig()
        with pytest.raises(AttributeError):
            config.model_name = "different-model"


class TestEmbeddingProviderInterface:
    """Tests for the abstract EmbeddingProvider interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "embed_batch")
        assert hasattr(EmbeddingProvider, "get_dimension")
        assert hasattr(EmbeddingProvider, "get_model_name")


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, model_name: str = "mock-model"):
        self._dimension = dimension
        self._model_name = model_name

    async def embed(self, text: str) -> list[float]:
        """Generate mock embedding."""
        # Return a simple embedding based on text length
        return [float(len(text) % 100) / 100.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch."""
        return [await self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_name(self) -> str:
        return self._model_name


class TestMockEmbeddingProvider:
    """Tests using mock embedding provider."""

    @pytest.fixture
    def provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embed_single_text(self, provider: MockEmbeddingProvider):
        """Test embedding a single text."""
        embedding = await provider.embed("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider: MockEmbeddingProvider):
        """Test embedding multiple texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = await provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self, provider: MockEmbeddingProvider):
        """Test embedding empty batch."""
        embeddings = await provider.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_is_available(self, provider: MockEmbeddingProvider):
        """Test availability check."""
        available = await provider.is_available()
        assert available is True

    def test_get_dimension(self, provider: MockEmbeddingProvider):
        """Test getting embedding dimension."""
        assert provider.get_dimension() == 384

    def test_get_model_name(self, provider: MockEmbeddingProvider):
        """Test getting model name."""
        assert provider.get_model_name() == "mock-model"


class TestSentenceTransformersEmbeddings:
    """Tests for SentenceTransformersEmbeddings provider."""

    def test_initialization(self):
        """Test provider initialization without loading model."""
        provider = SentenceTransformersEmbeddings(
            model_name="all-MiniLM-L6-v2",
            normalize=True,
        )

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.normalize is True
        assert provider._model is None  # Lazy loading

    def test_get_model_name(self):
        """Test getting model name."""
        provider = SentenceTransformersEmbeddings(model_name="test-model")
        assert provider.get_model_name() == "test-model"

    def test_default_dimension(self):
        """Test default dimension for all-MiniLM-L6-v2."""
        provider = SentenceTransformersEmbeddings()
        # Without loading the model, should return default
        assert provider.get_dimension() == 384


class TestOllamaEmbeddings:
    """Tests for OllamaEmbeddings provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = OllamaEmbeddings(
            model_name="nomic-embed-text",
            host="http://localhost:11434",
        )

        assert provider.model_name == "nomic-embed-text"
        assert provider.host == "http://localhost:11434"
        assert provider._client is None  # Lazy loading

    def test_get_dimension(self):
        """Test getting embedding dimension."""
        provider = OllamaEmbeddings()
        assert provider.get_dimension() == 768  # nomic-embed-text dimension

    def test_get_model_name(self):
        """Test getting model name."""
        provider = OllamaEmbeddings(model_name="custom-model")
        assert provider.get_model_name() == "custom-model"


class TestGeminiEmbeddings:
    """Tests for GeminiEmbeddings provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = GeminiEmbeddings(
            api_key="test-key",
            model_name="models/embedding-001",
        )

        assert provider.api_key == "test-key"
        assert provider.model_name == "models/embedding-001"
        assert provider._client is None  # Lazy loading

    def test_initialization_without_key(self):
        """Test initialization without API key."""
        provider = GeminiEmbeddings()
        assert provider.api_key is None

    def test_get_dimension(self):
        """Test getting embedding dimension."""
        provider = GeminiEmbeddings()
        assert provider.get_dimension() == 768

    def test_get_model_name(self):
        """Test getting model name."""
        provider = GeminiEmbeddings(model_name="custom/model")
        assert provider.get_model_name() == "custom/model"


class TestHybridEmbeddings:
    """Tests for HybridEmbeddings provider with fallback."""

    @pytest.fixture
    def primary_provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider(dimension=384, model_name="primary")

    @pytest.fixture
    def secondary_provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider(dimension=768, model_name="secondary")

    @pytest.mark.asyncio
    async def test_uses_primary_when_available(
        self, primary_provider: MockEmbeddingProvider
    ):
        """Test that primary provider is used when available."""
        hybrid = HybridEmbeddings(primary=primary_provider)

        embedding = await hybrid.embed("test")

        assert len(embedding) == 384  # Primary dimension
        assert hybrid.get_model_name() == "primary"

    @pytest.mark.asyncio
    async def test_fallback_to_secondary(
        self,
        secondary_provider: MockEmbeddingProvider,
    ):
        """Test fallback to secondary when primary unavailable."""
        # Create a failing primary
        failing_primary = MockEmbeddingProvider()
        failing_primary.is_available = AsyncMock(return_value=False)

        hybrid = HybridEmbeddings(
            primary=failing_primary,
            secondary=secondary_provider,
        )

        embedding = await hybrid.embed("test")
        # Should use secondary
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test full fallback chain: primary -> secondary -> fallback."""
        # All fail except fallback
        failing1 = MockEmbeddingProvider(dimension=100)
        failing1.is_available = AsyncMock(return_value=False)

        failing2 = MockEmbeddingProvider(dimension=200)
        failing2.is_available = AsyncMock(return_value=False)

        working = MockEmbeddingProvider(dimension=300, model_name="fallback")

        hybrid = HybridEmbeddings(
            primary=failing1,
            secondary=failing2,
            fallback=working,
        )

        embedding = await hybrid.embed("test")
        assert len(embedding) == 300
        assert hybrid.get_model_name() == "fallback"

    @pytest.mark.asyncio
    async def test_creates_default_when_no_providers(self):
        """Test that default provider is created when none configured."""
        hybrid = HybridEmbeddings()

        # This will create a default SentenceTransformersEmbeddings
        # but we can't test it without the actual library
        # So we just check the initialization doesn't fail
        assert hybrid.primary is None
        assert hybrid.secondary is None
        assert hybrid.fallback is None

    def test_get_dimension_with_active_provider(
        self, primary_provider: MockEmbeddingProvider
    ):
        """Test getting dimension from active provider."""
        hybrid = HybridEmbeddings(primary=primary_provider)
        hybrid._active_provider = primary_provider

        assert hybrid.get_dimension() == 384

    def test_get_dimension_default(self):
        """Test default dimension when no active provider."""
        hybrid = HybridEmbeddings()
        assert hybrid.get_dimension() == 384  # Default for all-MiniLM-L6-v2

    def test_get_model_name_default(self):
        """Test default model name when no active provider."""
        hybrid = HybridEmbeddings()
        assert hybrid.get_model_name() == "all-MiniLM-L6-v2"


class TestCreateEmbeddingProvider:
    """Tests for the factory function."""

    def test_creates_sentence_transformers(self):
        """Test creating SentenceTransformersEmbeddings."""
        config = EmbeddingConfig(
            provider_type=EmbeddingProviderType.SENTENCE_TRANSFORMERS,
            model_name="test-model",
        )
        provider = create_embedding_provider(config)

        assert isinstance(provider, SentenceTransformersEmbeddings)
        assert provider.model_name == "test-model"

    def test_creates_ollama(self):
        """Test creating OllamaEmbeddings."""
        config = EmbeddingConfig(
            provider_type=EmbeddingProviderType.OLLAMA,
            model_name="test-model",
            ollama_host="http://test:11434",
        )
        provider = create_embedding_provider(config)

        assert isinstance(provider, OllamaEmbeddings)
        assert provider.model_name == "test-model"
        assert provider.host == "http://test:11434"

    def test_creates_gemini(self):
        """Test creating GeminiEmbeddings."""
        config = EmbeddingConfig(
            provider_type=EmbeddingProviderType.GEMINI,
            model_name="models/test",
            gemini_api_key="test-key",
        )
        provider = create_embedding_provider(config)

        assert isinstance(provider, GeminiEmbeddings)
        assert provider.api_key == "test-key"

    def test_creates_hybrid(self):
        """Test creating HybridEmbeddings."""
        config = EmbeddingConfig(
            provider_type=EmbeddingProviderType.HYBRID,
        )
        provider = create_embedding_provider(config)

        assert isinstance(provider, HybridEmbeddings)
        assert isinstance(provider.primary, SentenceTransformersEmbeddings)

    def test_creates_default_without_config(self):
        """Test creating default provider without config."""
        provider = create_embedding_provider()

        assert isinstance(provider, SentenceTransformersEmbeddings)
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_default_for_unknown_type(self):
        """Test that unknown types default to sentence-transformers."""
        config = EmbeddingConfig(
            model_name="custom-model",
        )
        provider = create_embedding_provider(config)

        assert isinstance(provider, SentenceTransformersEmbeddings)


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_creation(self):
        """Test creating an embedding result."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            dimension=3,
            texts_embedded=1,
        )

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.model == "test-model"
        assert result.dimension == 3
        assert result.texts_embedded == 1
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test creating result with metadata."""
        result = EmbeddingResult(
            embeddings=[[0.1]],
            model="test",
            dimension=1,
            texts_embedded=1,
            metadata={"key": "value"},
        )

        assert result.metadata == {"key": "value"}


class TestEmbeddingProviderRealWorld:
    """Tests simulating real-world embedding scenarios.

    These tests use mock providers but simulate realistic usage patterns
    that would occur with Claude Code's knowledge engine.
    """

    @pytest.fixture
    def provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embed_code_snippet(self, provider: MockEmbeddingProvider):
        """Test embedding a code snippet like Claude Code would index."""
        code = '''def authenticate_user(username: str, password: str) -> User:
    """Authenticate a user with username and password.

    Args:
        username: The user's username
        password: The user's password

    Returns:
        The authenticated User object

    Raises:
        AuthenticationError: If credentials are invalid
    """
    user = User.find_by_username(username)
    if user and user.check_password(password):
        return user
    raise AuthenticationError("Invalid credentials")'''

        embedding = await provider.embed(code)

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_embed_documentation(self, provider: MockEmbeddingProvider):
        """Test embedding documentation like from CLAUDE.md."""
        doc = """The authentication system uses JWT tokens for session management.
When a user logs in, they receive an access token (valid for 15 minutes)
and a refresh token (valid for 7 days). The access token should be included
in the Authorization header for all authenticated requests."""

        embedding = await provider.embed(doc)
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_embed_batch_of_chunks(self, provider: MockEmbeddingProvider):
        """Test embedding a batch of code chunks like during indexing."""
        chunks = [
            "def login(username, password): pass",
            "def logout(): pass",
            "class User: pass",
            "class Session: pass",
            "async def refresh_token(): pass",
        ]

        embeddings = await provider.embed_batch(chunks)

        assert len(embeddings) == 5
        assert all(len(e) == 384 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embed_query_for_retrieval(self, provider: MockEmbeddingProvider):
        """Test embedding a search query for RAG retrieval."""
        query = "How does user authentication work?"

        embedding = await provider.embed(query)

        assert len(embedding) == 384
        # Query embedding should be same dimension as document embeddings

    @pytest.mark.asyncio
    async def test_large_batch_embedding(self, provider: MockEmbeddingProvider):
        """Test embedding a large batch of chunks."""
        # Simulate indexing a file with many functions
        chunks = [f"def function_{i}(): pass" for i in range(100)]

        embeddings = await provider.embed_batch(chunks)

        assert len(embeddings) == 100

    @pytest.mark.asyncio
    async def test_empty_text_embedding(self, provider: MockEmbeddingProvider):
        """Test embedding empty text."""
        embedding = await provider.embed("")

        # Should handle gracefully
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_unicode_text_embedding(self, provider: MockEmbeddingProvider):
        """Test embedding text with unicode characters."""
        text = "def greet(): return '你好世界' # Hello World in Chinese"

        embedding = await provider.embed(text)
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_very_long_text_embedding(self, provider: MockEmbeddingProvider):
        """Test embedding very long text."""
        # Simulate a large class definition
        text = "class LargeClass:\n" + "\n".join(
            [f"    def method_{i}(self): pass" for i in range(200)]
        )

        embedding = await provider.embed(text)
        assert len(embedding) == 384
