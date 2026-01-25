"""RAG (Retrieval-Augmented Generation) pipeline for the Knowledge Engine.

Provides semantic search over indexed codebase using:
- ChromaDB for vector storage and retrieval
- Code-aware chunking from the indexer
- Hybrid search combining semantic and keyword matching

The RAG pipeline enables context-aware code assistance by finding
relevant code snippets for questions and decisions.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from perpetualcc.knowledge.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    create_embedding_provider,
)
from perpetualcc.knowledge.indexer import (
    ChunkType,
    CodebaseIndexer,
    CodeChunk,
    IndexerConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory for ChromaDB persistence
        embedding_config: Configuration for embeddings
        indexer_config: Configuration for code indexing
        default_top_k: Default number of results to return
        similarity_threshold: Minimum similarity score (0-1)
        rerank_results: Whether to re-rank results
        include_metadata: Whether to include metadata in results
    """

    collection_name: str = "perpetualcc_codebase"
    persist_directory: str | None = None
    embedding_config: EmbeddingConfig | None = None
    indexer_config: IndexerConfig | None = None
    default_top_k: int = 5
    similarity_threshold: float = 0.3
    rerank_results: bool = True
    include_metadata: bool = True


@dataclass
class RetrievalResult:
    """Result from a RAG retrieval query.

    Attributes:
        chunk_id: Unique identifier of the chunk
        file_path: Path to the source file
        content: The retrieved content
        score: Similarity score (0-1, higher is better)
        chunk_type: Type of code chunk
        name: Name of the code entity
        start_line: Starting line number
        end_line: Ending line number
        docstring: Documentation if available
        language: Programming language
        metadata: Additional metadata
    """

    chunk_id: str
    file_path: str
    content: str
    score: float
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    docstring: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format as context string for LLM prompts.

        Returns:
            Formatted string with file path, location, and content
        """
        header = f"# {self.file_path}:{self.start_line}-{self.end_line}"
        if self.name:
            header += f" ({self.chunk_type}: {self.name})"

        parts = [header]

        if self.docstring:
            parts.append(f"# Doc: {self.docstring[:200]}...")

        parts.append(f"```{self.language or ''}")
        parts.append(self.content)
        parts.append("```")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "content": self.content,
            "score": self.score,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class IndexStats:
    """Statistics about the indexed codebase.

    Attributes:
        total_files: Number of files indexed
        total_chunks: Number of chunks stored
        languages: Count per language
        chunk_types: Count per chunk type
        last_indexed: Timestamp of last indexing
        index_hash: Hash of indexed content for change detection
    """

    total_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    chunk_types: dict[str, int] = field(default_factory=dict)
    last_indexed: datetime | None = None
    index_hash: str | None = None


class RAGPipeline:
    """RAG pipeline for semantic code search.

    Provides:
    - Indexing: Scan codebase, embed chunks, store in ChromaDB
    - Retrieval: Find relevant code for queries
    - Update: Incremental updates when code changes

    Usage:
        pipeline = RAGPipeline(project_path="./my-project")
        await pipeline.index_project()
        results = await pipeline.retrieve("authentication handler")
    """

    def __init__(
        self,
        project_path: str | Path,
        config: RAGConfig | None = None,
        embeddings: EmbeddingProvider | None = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            project_path: Path to the project root
            config: RAG configuration
            embeddings: Custom embedding provider (optional)
        """
        self.project_path = Path(project_path).resolve()
        self.config = config or RAGConfig()
        self.embeddings = embeddings or create_embedding_provider(self.config.embedding_config)

        # Set up persistence directory
        if self.config.persist_directory:
            self._persist_path = Path(self.config.persist_directory)
        else:
            self._persist_path = self.project_path / ".perpetualcc" / "chromadb"

        self._client = None
        self._collection = None
        self._indexer = CodebaseIndexer(
            self.project_path,
            self.config.indexer_config,
        )
        self._stats = IndexStats()

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError as e:
                raise ImportError(
                    "chromadb is required for RAG functionality. Install with: pip install chromadb"
                ) from e

            # Ensure persistence directory exists
            self._persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
            )
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()

            # Get or create collection with cosine similarity
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def index_project(
        self,
        force_reindex: bool = False,
        progress_callback: callable | None = None,
    ) -> IndexStats:
        """Index the entire project.

        Scans all source files, extracts chunks, embeds them,
        and stores in ChromaDB for retrieval.

        Args:
            force_reindex: If True, reindex even if unchanged
            progress_callback: Optional callback for progress updates
                              Signature: callback(current: int, total: int, file: str)

        Returns:
            IndexStats with indexing statistics
        """
        logger.info("Indexing project: %s", self.project_path)

        # Calculate content hash to detect changes
        content_hash = await self._calculate_content_hash()

        # Check if reindex is needed
        collection = self._get_collection()
        existing_count = collection.count()

        if not force_reindex and existing_count > 0:
            # Check if content has changed
            try:
                existing_meta = collection.get(limit=1, include=["metadatas"])
                if existing_meta["metadatas"]:
                    stored_hash = existing_meta["metadatas"][0].get("index_hash")
                    if stored_hash == content_hash:
                        logger.info("Index is up to date, skipping reindex")
                        self._stats.index_hash = content_hash
                        self._stats.total_chunks = existing_count
                        return self._stats
            except Exception:
                pass

        # Clear existing data if reindexing
        if existing_count > 0:
            logger.info("Clearing existing index (%d chunks)", existing_count)
            # Delete all existing documents
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

        # Collect all chunks
        all_chunks: list[CodeChunk] = []
        files_processed = 0

        for code_file in self._indexer.index_project():
            files_processed += 1
            all_chunks.extend(code_file.chunks)

            # Update language stats
            lang = code_file.language.value
            self._stats.languages[lang] = self._stats.languages.get(lang, 0) + 1

            if progress_callback:
                progress_callback(files_processed, -1, code_file.path)

        if not all_chunks:
            logger.warning("No chunks extracted from project")
            return self._stats

        logger.info("Extracted %d chunks from %d files", len(all_chunks), files_processed)

        # Embed chunks in batches
        batch_size = 32
        total_chunks = len(all_chunks)

        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i : i + batch_size]

            # Prepare texts for embedding
            texts = [chunk.to_embedding_text() for chunk in batch]

            # Generate embeddings
            embeddings = await self.embeddings.embed_batch(texts)

            # Prepare data for ChromaDB
            ids = [chunk.id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [
                {
                    "file_path": chunk.file_path,
                    "chunk_type": chunk.chunk_type.value,
                    "name": chunk.name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language.value,
                    "docstring": chunk.docstring or "",
                    "signature": chunk.signature or "",
                    "parent_name": chunk.parent_name or "",
                    "index_hash": content_hash,
                }
                for chunk in batch
            ]

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Update chunk type stats
            for chunk in batch:
                ct = chunk.chunk_type.value
                self._stats.chunk_types[ct] = self._stats.chunk_types.get(ct, 0) + 1

            if progress_callback:
                progress_callback(min(i + batch_size, total_chunks), total_chunks, "embedding")

        # Update stats
        self._stats.total_files = files_processed
        self._stats.total_chunks = total_chunks
        self._stats.last_indexed = datetime.now()
        self._stats.index_hash = content_hash

        logger.info(
            "Indexing complete: %d files, %d chunks",
            files_processed,
            total_chunks,
        )

        return self._stats

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_types: list[str] | None = None,
        filter_files: list[str] | None = None,
        filter_language: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant code chunks for a query.

        Args:
            query: The search query
            top_k: Number of results to return (default from config)
            filter_types: Only include these chunk types
            filter_files: Only include chunks from these files
            filter_language: Only include chunks in this language

        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("Index is empty, please run index_project() first")
            return []

        top_k = top_k or self.config.default_top_k

        # Embed the query
        query_embedding = await self.embeddings.embed(query)

        # Build where filter
        where_filter = None
        where_conditions = []

        if filter_types:
            where_conditions.append({"chunk_type": {"$in": filter_types}})
        if filter_language:
            where_conditions.append({"language": filter_language})
        if filter_files:
            where_conditions.append({"file_path": {"$in": filter_files}})

        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        # Query ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more to filter by threshold
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        # Process results
        retrieval_results = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i, chunk_id in enumerate(results["ids"][0]):
            # Convert distance to similarity score
            # ChromaDB returns cosine distance (0-2), convert to similarity (0-1)
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1 - (distance / 2)

            # Apply similarity threshold
            if score < self.config.similarity_threshold:
                continue

            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    file_path=metadata.get("file_path", ""),
                    content=content,
                    score=score,
                    chunk_type=metadata.get("chunk_type", "unknown"),
                    name=metadata.get("name", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    docstring=metadata.get("docstring") or None,
                    language=metadata.get("language"),
                    metadata=metadata,
                )
            )

        # Sort by score and limit to top_k
        retrieval_results.sort(key=lambda r: r.score, reverse=True)
        retrieval_results = retrieval_results[:top_k]

        # Optional re-ranking (could use cross-encoder)
        if self.config.rerank_results and len(retrieval_results) > 1:
            retrieval_results = await self._rerank_results(query, retrieval_results)

        return retrieval_results

    async def retrieve_by_file(
        self,
        file_path: str,
        chunk_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve all chunks from a specific file.

        Args:
            file_path: Path to the file (relative to project root)
            chunk_types: Optional filter by chunk types

        Returns:
            List of RetrievalResult objects from the file
        """
        collection = self._get_collection()

        where_filter: dict[str, Any] = {"file_path": file_path}
        if chunk_types:
            where_filter = {
                "$and": [
                    {"file_path": file_path},
                    {"chunk_type": {"$in": chunk_types}},
                ]
            }

        try:
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error("Failed to retrieve by file: %s", e)
            return []

        retrieval_results = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            content = results["documents"][i] if results["documents"] else ""

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    file_path=metadata.get("file_path", ""),
                    content=content,
                    score=1.0,  # Not from similarity search
                    chunk_type=metadata.get("chunk_type", "unknown"),
                    name=metadata.get("name", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    docstring=metadata.get("docstring") or None,
                    language=metadata.get("language"),
                    metadata=metadata,
                )
            )

        # Sort by line number
        retrieval_results.sort(key=lambda r: r.start_line)
        return retrieval_results

    async def retrieve_related(
        self,
        chunk_id: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Find chunks related to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of related chunks to find

        Returns:
            List of related RetrievalResult objects
        """
        collection = self._get_collection()

        # Get the reference chunk
        try:
            ref_result = collection.get(
                ids=[chunk_id],
                include=["embeddings", "metadatas"],
            )
        except Exception as e:
            logger.error("Failed to get reference chunk: %s", e)
            return []

        if not ref_result["ids"]:
            return []

        # Use its embedding to find similar chunks
        ref_embedding = ref_result["embeddings"][0]
        ref_file = ref_result["metadatas"][0].get("file_path", "")

        results = collection.query(
            query_embeddings=[ref_embedding],
            n_results=top_k + 1,  # +1 to exclude self
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        for i, cid in enumerate(results["ids"][0]):
            if cid == chunk_id:  # Skip self
                continue

            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1 - (distance / 2)

            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=cid,
                    file_path=metadata.get("file_path", ""),
                    content=content,
                    score=score,
                    chunk_type=metadata.get("chunk_type", "unknown"),
                    name=metadata.get("name", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    docstring=metadata.get("docstring") or None,
                    language=metadata.get("language"),
                    metadata=metadata,
                )
            )

        return retrieval_results[:top_k]

    async def search_by_name(
        self,
        name: str,
        exact: bool = False,
    ) -> list[RetrievalResult]:
        """Search for chunks by entity name.

        Args:
            name: Name to search for (function, class, etc.)
            exact: If True, require exact match

        Returns:
            List of matching RetrievalResult objects
        """
        collection = self._get_collection()

        if exact:
            where_filter = {"name": name}
        else:
            # ChromaDB doesn't support regex, so we get all and filter
            # For large codebases, this should use a separate name index
            where_filter = None

        try:
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error("Failed to search by name: %s", e)
            return []

        retrieval_results = []
        name_lower = name.lower()

        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            chunk_name = metadata.get("name", "")

            # Filter by name match
            if exact:
                if chunk_name != name:
                    continue
            else:
                if name_lower not in chunk_name.lower():
                    continue

            content = results["documents"][i] if results["documents"] else ""

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    file_path=metadata.get("file_path", ""),
                    content=content,
                    score=1.0 if exact else 0.8,
                    chunk_type=metadata.get("chunk_type", "unknown"),
                    name=chunk_name,
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    docstring=metadata.get("docstring") or None,
                    language=metadata.get("language"),
                    metadata=metadata,
                )
            )

        return retrieval_results

    async def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-rank results for better relevance.

        Simple re-ranking based on:
        - Keyword match boost
        - Docstring relevance
        - Code structure relevance

        Args:
            query: Original query
            results: Initial results

        Returns:
            Re-ranked results
        """
        query_terms = set(query.lower().split())

        for result in results:
            boost = 0.0

            # Boost for name match
            name_terms = set(result.name.lower().split("_"))
            name_match = len(query_terms & name_terms) / max(len(query_terms), 1)
            boost += name_match * 0.2

            # Boost for docstring match
            if result.docstring:
                doc_terms = set(result.docstring.lower().split())
                doc_match = len(query_terms & doc_terms) / max(len(query_terms), 1)
                boost += doc_match * 0.1

            # Boost for content keyword match
            content_lower = result.content.lower()
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            boost += (keyword_matches / max(len(query_terms), 1)) * 0.1

            # Boost for chunk type relevance
            if result.chunk_type in ("function", "class"):
                boost += 0.05

            # Apply boost (capped at 1.0)
            result.score = min(result.score + boost, 1.0)

        # Re-sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def _calculate_content_hash(self) -> str:
        """Calculate a hash of the codebase content for change detection.

        Returns:
            Hash string representing current codebase state
        """
        hasher = hashlib.sha256()

        def _compute():
            for file_path in sorted(self._indexer.scan_files()):
                try:
                    # Include file path and modification time
                    stat = file_path.stat()
                    hasher.update(str(file_path).encode())
                    hasher.update(str(stat.st_mtime).encode())
                except OSError:
                    continue
            return hasher.hexdigest()[:16]

        return await asyncio.to_thread(_compute)

    def get_stats(self) -> IndexStats:
        """Get current index statistics.

        Returns:
            IndexStats object
        """
        collection = self._get_collection()
        self._stats.total_chunks = collection.count()
        return self._stats

    async def clear_index(self) -> None:
        """Clear the entire index."""
        collection = self._get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        self._stats = IndexStats()
        logger.info("Index cleared")


class KnowledgeEngine:
    """High-level Knowledge Engine interface.

    Combines RAG pipeline with code graph (Phase 7 complete) for
    comprehensive codebase understanding.

    This class implements the KnowledgeEngine protocol expected
    by the MasterAgent.
    """

    def __init__(
        self,
        project_path: str | Path,
        config: RAGConfig | None = None,
    ):
        """Initialize the Knowledge Engine.

        Args:
            project_path: Path to the project root
            config: RAG configuration
        """
        self.project_path = Path(project_path).resolve()
        self.rag = RAGPipeline(project_path, config)
        self._code_graph = None  # Initialized lazily

    async def initialize(self, force_reindex: bool = False) -> IndexStats:
        """Initialize the knowledge engine.

        Indexes the codebase and builds the code graph.

        Args:
            force_reindex: If True, reindex even if unchanged

        Returns:
            IndexStats from indexing
        """
        stats = await self.rag.index_project(force_reindex=force_reindex)

        # Initialize code graph if available
        try:
            from perpetualcc.knowledge.code_graph import CodeGraph

            self._code_graph = CodeGraph(self.project_path)
            await self._code_graph.build()
        except ImportError:
            logger.debug("Code graph not available")
        except Exception as e:
            logger.warning("Failed to build code graph: %s", e)

        return stats

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context for a query.

        This method is used by the MasterAgent for context-aware
        question answering and permission decisions.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with retrieved context
        """
        results = await self.rag.retrieve(query, top_k=top_k)

        # Convert to dict format expected by MasterAgent
        return [result.to_dict() for result in results]

    async def get_file_context(self, file_path: str) -> dict[str, Any]:
        """Get comprehensive context for a file.

        Combines RAG results with code graph information.

        Args:
            file_path: Path to the file (relative to project)

        Returns:
            Dictionary with file context
        """
        context: dict[str, Any] = {
            "file_path": file_path,
            "chunks": [],
            "imports": [],
            "exports": [],
            "dependencies": [],
        }

        # Get chunks from RAG
        chunks = await self.rag.retrieve_by_file(file_path)
        context["chunks"] = [c.to_dict() for c in chunks]

        # Get graph context if available
        if self._code_graph:
            try:
                graph_context = self._code_graph.get_file_context(file_path)
                context.update(graph_context)
            except Exception as e:
                logger.debug("Failed to get graph context: %s", e)

        return context

    async def find_related_code(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Find code related to an entity.

        Args:
            entity_name: Name of the entity (function, class, etc.)
            depth: How many levels of relationships to traverse

        Returns:
            List of related code entities
        """
        # Search by name first
        results = await self.rag.search_by_name(entity_name)

        if not results:
            # Try semantic search
            results = await self.rag.retrieve(entity_name, top_k=3)

        # Get related from code graph
        related = []
        if self._code_graph and results:
            for result in results[:3]:
                try:
                    graph_related = self._code_graph.get_related(
                        result.name,
                        depth=depth,
                    )
                    related.extend(graph_related)
                except Exception:
                    pass

        return [r.to_dict() for r in results] + related

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge engine statistics.

        Returns:
            Dictionary with statistics
        """
        rag_stats = self.rag.get_stats()
        stats = {
            "rag": {
                "total_files": rag_stats.total_files,
                "total_chunks": rag_stats.total_chunks,
                "languages": rag_stats.languages,
                "chunk_types": rag_stats.chunk_types,
                "last_indexed": (
                    rag_stats.last_indexed.isoformat() if rag_stats.last_indexed else None
                ),
            },
        }

        if self._code_graph:
            try:
                stats["graph"] = self._code_graph.get_stats()
            except Exception:
                pass

        return stats
