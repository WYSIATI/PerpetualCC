"""Knowledge engine - RAG, code graph, and embeddings.

Phase 7 of PerpetualCC - provides deep project understanding through:
- Embeddings: Text to vector conversion for semantic search
- Indexer: Codebase scanning and chunk extraction
- RAG: Retrieval-Augmented Generation pipeline
- Code Graph: Relationship graph for code navigation

Usage:
    from perpetualcc.knowledge import KnowledgeEngine

    engine = KnowledgeEngine("./my-project")
    await engine.initialize()
    results = await engine.retrieve("authentication handler")

Dependencies:
    Required: sentence-transformers, chromadb
    Optional: tree-sitter-language-pack, networkx
    Install with: pip install perpetualcc[knowledge]
"""

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
from perpetualcc.knowledge.indexer import (
    ChunkType,
    CodebaseIndexer,
    CodeChunk,
    CodeFile,
    IndexerConfig,
    Language,
)
from perpetualcc.knowledge.rag import (
    IndexStats,
    KnowledgeEngine,
    RAGConfig,
    RAGPipeline,
    RetrievalResult,
)

# Code graph may not be available if networkx isn't installed
try:
    from perpetualcc.knowledge.code_graph import (
        CodeGraph,
        EdgeType,
        GraphConfig,
        GraphEdge,
        GraphNode,
        NodeType,
    )

    __all__ = [
        # Embeddings
        "EmbeddingConfig",
        "EmbeddingProvider",
        "EmbeddingProviderType",
        "EmbeddingResult",
        "SentenceTransformersEmbeddings",
        "OllamaEmbeddings",
        "GeminiEmbeddings",
        "HybridEmbeddings",
        "create_embedding_provider",
        # Indexer
        "CodebaseIndexer",
        "CodeChunk",
        "CodeFile",
        "ChunkType",
        "Language",
        "IndexerConfig",
        # RAG
        "RAGPipeline",
        "RAGConfig",
        "RetrievalResult",
        "IndexStats",
        "KnowledgeEngine",
        # Code Graph
        "CodeGraph",
        "GraphConfig",
        "GraphNode",
        "GraphEdge",
        "NodeType",
        "EdgeType",
    ]
except ImportError:
    # Code graph not available
    __all__ = [
        # Embeddings
        "EmbeddingConfig",
        "EmbeddingProvider",
        "EmbeddingProviderType",
        "EmbeddingResult",
        "SentenceTransformersEmbeddings",
        "OllamaEmbeddings",
        "GeminiEmbeddings",
        "HybridEmbeddings",
        "create_embedding_provider",
        # Indexer
        "CodebaseIndexer",
        "CodeChunk",
        "CodeFile",
        "ChunkType",
        "Language",
        "IndexerConfig",
        # RAG
        "RAGPipeline",
        "RAGConfig",
        "RetrievalResult",
        "IndexStats",
        "KnowledgeEngine",
    ]
