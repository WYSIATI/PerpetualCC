"""Embedding providers for the Knowledge Engine.

Provides a unified interface for generating text embeddings using:
- Local models via sentence-transformers or Ollama
- API-based models via Gemini
- Hybrid approach: try local first, fallback to API

The default and recommended approach is to use sentence-transformers locally
for fast, free, offline-capable embeddings.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingProviderType(Enum):
    """Type of embedding provider."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding providers.

    Attributes:
        provider_type: Type of embedding provider to use
        model_name: Model name/identifier
        dimension: Expected embedding dimension (for validation)
        batch_size: Batch size for embedding multiple texts
        normalize: Whether to normalize embeddings to unit length
        cache_enabled: Whether to cache embeddings
        ollama_host: Ollama server host (if using Ollama)
        gemini_api_key: Gemini API key (if using Gemini)
    """

    provider_type: EmbeddingProviderType = EmbeddingProviderType.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str | None = None


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embeddings: List of embedding vectors
        model: Model used for embedding
        dimension: Dimension of embeddings
        texts_embedded: Number of texts embedded
        metadata: Additional metadata
    """

    embeddings: list[list[float]]
    model: str
    dimension: int
    texts_embedded: int
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    An embedding provider converts text into dense vector representations
    that capture semantic meaning. These vectors are used for:
    - Similarity search in the RAG pipeline
    - Finding related code in the knowledge engine
    - Clustering and classification tasks
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Model name string
        """
        ...

    async def is_available(self) -> bool:
        """Check if the provider is available and working.

        Returns:
            True if provider is available
        """
        try:
            test_embedding = await self.embed("test")
            return len(test_embedding) == self.get_dimension()
        except Exception as e:
            logger.warning("Provider availability check failed: %s", e)
            return False


class SentenceTransformersEmbeddings(EmbeddingProvider):
    """Local embeddings using sentence-transformers library.

    This is the default and recommended provider for local embeddings.
    Uses the all-MiniLM-L6-v2 model which is:
    - Small (22MB)
    - Fast
    - High quality for code and text
    - Works offline

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize the sentence-transformers embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            normalize: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension: int | None = None

    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from e

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses asyncio.to_thread to run encoding in a separate thread
        to avoid blocking the event loop.
        """
        if not texts:
            return []

        def _encode():
            model = self._get_model()
            return model.encode(
                texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ).tolist()

        return await asyncio.to_thread(_encode)

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns the known dimension for common models without loading them.
        For the default all-MiniLM-L6-v2, this is 384.
        """
        if self._dimension is not None:
            return self._dimension

        # Known dimensions for common models
        known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
            "paraphrase-mpnet-base-v2": 768,
        }

        return known_dimensions.get(self.model_name, 384)

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class OllamaEmbeddings(EmbeddingProvider):
    """Local embeddings via Ollama server.

    Uses Ollama to run embedding models locally. Default model is
    nomic-embed-text which provides good quality embeddings.

    Requires:
    - Ollama installed and running: https://ollama.ai
    - Model pulled: ollama pull nomic-embed-text
    - pip install ollama
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        """Initialize the Ollama embedding provider.

        Args:
            model_name: Name of the Ollama embedding model
            host: Ollama server host URL
        """
        self.model_name = model_name
        self.host = host
        self._client = None
        # nomic-embed-text produces 768-dim embeddings
        self._dimension = 768

    def _get_client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                import ollama
            except ImportError as e:
                raise ImportError(
                    "ollama is required for Ollama embeddings. Install with: pip install ollama"
                ) from e

            self._client = ollama.Client(host=self.host)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""

        def _embed():
            client = self._get_client()
            response = client.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]

        return await asyncio.to_thread(_embed)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Ollama doesn't support batch embedding natively, so we
        process texts sequentially (could be parallelized).
        """
        if not texts:
            return []

        results = []
        for text in texts:
            embedding = await self.embed(text)
            results.append(embedding)
        return results

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class GeminiEmbeddings(EmbeddingProvider):
    """API embeddings via Google Gemini.

    Uses Google's Gemini API for high-quality embeddings.
    Requires an API key.

    Requires:
    - pip install google-genai
    - GEMINI_API_KEY environment variable or explicit key
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "models/embedding-001",
    ):
        """Initialize the Gemini embedding provider.

        Args:
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
            model_name: Name of the Gemini embedding model
        """
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        # Gemini embeddings are 768-dimensional
        self._dimension = 768

    def _get_client(self):
        """Lazy load the Gemini client."""
        if self._client is None:
            import os

            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai is required for Gemini embeddings. "
                    "Install with: pip install google-genai"
                ) from e

            api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key required. Set GEMINI_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._client = genai.Client(api_key=api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""

        def _embed():
            client = self._get_client()
            response = client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            return response.embeddings[0].values

        return await asyncio.to_thread(_embed)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        def _embed_batch():
            client = self._get_client()
            results = []
            for text in texts:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                results.append(response.embeddings[0].values)
            return results

        return await asyncio.to_thread(_embed_batch)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class HybridEmbeddings(EmbeddingProvider):
    """Hybrid embedding provider with fallback support.

    Tries providers in order:
    1. Primary provider (default: sentence-transformers)
    2. Secondary provider (optional: Ollama or Gemini)
    3. Fallback provider (optional: Gemini API)

    This ensures embeddings are always available, preferring
    local/free options when possible.
    """

    def __init__(
        self,
        primary: EmbeddingProvider | None = None,
        secondary: EmbeddingProvider | None = None,
        fallback: EmbeddingProvider | None = None,
    ):
        """Initialize the hybrid embedding provider.

        Args:
            primary: Primary provider (default: SentenceTransformersEmbeddings)
            secondary: Secondary provider (optional)
            fallback: Fallback provider (optional)
        """
        self.primary = primary
        self.secondary = secondary
        self.fallback = fallback
        self._active_provider: EmbeddingProvider | None = None

    async def _get_active_provider(self) -> EmbeddingProvider:
        """Get the first available provider."""
        if self._active_provider:
            return self._active_provider

        providers = [p for p in [self.primary, self.secondary, self.fallback] if p]

        for provider in providers:
            if await provider.is_available():
                self._active_provider = provider
                logger.info(
                    "Using embedding provider: %s (%s)",
                    type(provider).__name__,
                    provider.get_model_name(),
                )
                return provider

        # If no providers configured or available, create default
        if not providers:
            logger.info("No providers configured, using sentence-transformers default")
            self.primary = SentenceTransformersEmbeddings()
            self._active_provider = self.primary
            return self.primary

        raise RuntimeError(
            "No embedding providers available. "
            "Install sentence-transformers: pip install sentence-transformers"
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using active provider."""
        provider = await self._get_active_provider()
        return await provider.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using active provider."""
        provider = await self._get_active_provider()
        return await provider.embed_batch(texts)

    def get_dimension(self) -> int:
        """Get dimension from active provider."""
        if self._active_provider:
            return self._active_provider.get_dimension()
        # Default dimension for all-MiniLM-L6-v2
        return 384

    def get_model_name(self) -> str:
        """Get model name from active provider."""
        if self._active_provider:
            return self._active_provider.get_model_name()
        return "all-MiniLM-L6-v2"


def create_embedding_provider(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        config: Embedding configuration (optional)

    Returns:
        Configured EmbeddingProvider instance
    """
    config = config or EmbeddingConfig()

    match config.provider_type:
        case EmbeddingProviderType.SENTENCE_TRANSFORMERS:
            return SentenceTransformersEmbeddings(
                model_name=config.model_name,
                normalize=config.normalize,
            )

        case EmbeddingProviderType.OLLAMA:
            return OllamaEmbeddings(
                model_name=config.model_name,
                host=config.ollama_host,
            )

        case EmbeddingProviderType.GEMINI:
            return GeminiEmbeddings(
                api_key=config.gemini_api_key,
                model_name=config.model_name,
            )

        case EmbeddingProviderType.HYBRID:
            primary = SentenceTransformersEmbeddings(
                model_name="all-MiniLM-L6-v2",
                normalize=config.normalize,
            )
            secondary = None
            fallback = None

            # Add Ollama as secondary if configured
            if config.ollama_host:
                try:
                    secondary = OllamaEmbeddings(
                        model_name="nomic-embed-text",
                        host=config.ollama_host,
                    )
                except Exception:
                    pass

            # Add Gemini as fallback if API key provided
            if config.gemini_api_key:
                try:
                    fallback = GeminiEmbeddings(api_key=config.gemini_api_key)
                except Exception:
                    pass

            return HybridEmbeddings(
                primary=primary,
                secondary=secondary,
                fallback=fallback,
            )

        case _:
            # Default to sentence-transformers
            return SentenceTransformersEmbeddings(
                model_name=config.model_name,
                normalize=config.normalize,
            )
