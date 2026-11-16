"""Retrieval service — bridges the vector store and the response schemas.

Translates raw ``(Document, score)`` tuples from ChromaDB into typed
``SourceDocument`` models that the generator and API layer consume.
"""

import time
from typing import List, Optional

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.models.schemas import SourceDocument
from backend.app.services.vector_store import VectorStoreService
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrieverService:
    """Retrieve and rank source chunks for a given query."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        settings: Settings,
    ) -> None:
        """Store service dependencies.

        Parameters
        ----------
        vector_store:
            The vector store to search against.
        settings:
            Application settings (default ``top_k``, etc.).
        """
        self._vector_store = vector_store
        self._settings = settings
        self.last_retrieval_latency_ms: float = 0.0

    @traceable(
        name="similarity_retrieval",
        run_type="retriever",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "retrieval"],
    )
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[SourceDocument]:
        """Retrieve the most relevant source chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language question.
        top_k:
            Number of chunks to retrieve.
        doc_id:
            If provided, restrict retrieval to this document.

        Returns
        -------
        list[SourceDocument]
            Chunks sorted by descending relevance score.
        """
        logger.info(
            "Retrieving top-%d chunks (doc_id=%s)", top_k, doc_id
        )

        filter_dict = {"doc_id": doc_id} if doc_id else None

        start = time.perf_counter()
        results = self._vector_store.similarity_search(
            query=query,
            k=top_k,
            filter_dict=filter_dict,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_retrieval_latency_ms = elapsed_ms

        sources: List[SourceDocument] = []
        for doc, score in results:
            sources.append(
                SourceDocument(
                    content=doc.page_content,
                    page_number=doc.metadata.get("page_number", 1),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=round(float(score), 4),
                    doc_id=doc.metadata.get("doc_id", ""),
                    filename=doc.metadata.get("filename", ""),
                )
            )

        # Sort by descending relevance score
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        logger.info(
            "Retrieved %d sources in %.1f ms", len(sources), elapsed_ms
        )
        return sources
