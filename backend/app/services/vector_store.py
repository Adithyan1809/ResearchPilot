"""ChromaDB vector store service.

Wraps ``langchain_chroma.Chroma`` to provide document storage, similarity
search, and collection management for the RAG pipeline.
"""

from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.services.embedding_service import EmbeddingService
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreService:
    """Manage a persistent ChromaDB collection for research paper chunks."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialise or open an existing ChromaDB collection.

        Parameters
        ----------
        settings:
            Application settings for persist directory and collection name.
        embedding_service:
            Provides the embedding function used by Chroma.
        """
        self._persist_dir = settings.chroma_persist_dir
        self._collection_name = settings.chroma_collection_name
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=embedding_service.get_embeddings(),
            persist_directory=self._persist_dir,
        )
        logger.info(
            "VectorStoreService initialised (dir=%s, collection=%s)",
            self._persist_dir,
            self._collection_name,
        )

    # ── Write operations ─────────────────────────────────────────────────

    @traceable(
        name="vector_store_indexing",
        run_type="tool",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "indexing"],
    )
    def add_documents(self, documents: List[Document]) -> int:
        """Add LangChain ``Document`` objects to the vector store.

        Parameters
        ----------
        documents:
            Documents with ``page_content`` and ``metadata`` to store.

        Returns
        -------
        int
            The number of documents successfully added.
        """
        if not documents:
            logger.warning("add_documents called with empty list")
            return 0

        logger.info("Adding %d documents to ChromaDB", len(documents))
        self._store.add_documents(documents)
        logger.info("Successfully added %d documents", len(documents))
        return len(documents)

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document.

        Parameters
        ----------
        doc_id:
            The unique document identifier whose chunks should be removed.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        logger.info("Deleting chunks for doc_id=%s", doc_id)

        # Retrieve matching chunk IDs via the underlying collection.
        collection = self._store._collection  # noqa: SLF001
        results = collection.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            logger.warning("No chunks found for doc_id=%s", doc_id)
            return 0

        collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunks for doc_id=%s", len(ids_to_delete), doc_id
        )
        return len(ids_to_delete)

    # ── Read / search operations ─────────────────────────────────────────

    @traceable(
        name="chroma_vector_search",
        run_type="retriever",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "vector_search"],
    )
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores.

        Parameters
        ----------
        query:
            The natural-language query to search for.
        k:
            Maximum number of results.
        filter_dict:
            Optional Chroma metadata filter (e.g. ``{"doc_id": "..."}``).

        Returns
        -------
        list[tuple[Document, float]]
            ``(document, score)`` pairs ordered by descending relevance.
        """
        logger.debug(
            "Similarity search: k=%d, filter=%s", k, filter_dict
        )

        kwargs: Dict = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict

        results = self._store.similarity_search_with_relevance_scores(
            query, **kwargs
        )

        logger.debug("Search returned %d results", len(results))
        return results

    # ── Collection introspection ─────────────────────────────────────────

    def get_document_count(self) -> int:
        """Return the total number of chunks in the collection."""
        collection = self._store._collection  # noqa: SLF001
        count: int = collection.count()
        return count

    def get_collection_stats(self) -> Dict:
        """Return descriptive statistics about the collection.

        Returns
        -------
        dict
            Keys: ``collection_name``, ``total_chunks``,
            ``persist_directory``.
        """
        return {
            "collection_name": self._collection_name,
            "total_chunks": self.get_document_count(),
            "persist_directory": self._persist_dir,
        }
