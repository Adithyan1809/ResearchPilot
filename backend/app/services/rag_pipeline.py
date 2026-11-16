"""RAG pipeline orchestrator — ties every service together.

Provides the high-level operations consumed by the API routes:
  * ``ingest_document`` — PDF → chunks → embeddings → vector store
  * ``query``           — question → retrieval → generation → cited answer
  * ``get_health``      — system health snapshot
  * ``get_documents``   — list ingested documents
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app import __version__
from backend.app.config import Settings
from backend.app.models.schemas import (
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TokenUsage,
)
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.generator import GeneratorService
from backend.app.services.retriever import RetrieverService
from backend.app.services.vector_store import VectorStoreService
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)

_REGISTRY_FILENAME = "document_registry.json"


class RAGPipeline:
    """End-to-end RAG pipeline for research paper analysis."""

    def __init__(self, settings: Settings) -> None:
        """Wire together every service the pipeline depends on.

        Parameters
        ----------
        settings:
            Centralised application configuration.
        """
        logger.info("Initialising RAG pipeline …")

        self._settings = settings

        # Services
        self._doc_processor = DocumentProcessor(settings)
        self._embedding_service = EmbeddingService(settings)
        self._vector_store = VectorStoreService(settings, self._embedding_service)
        self._retriever = RetrieverService(self._vector_store, settings)
        self._generator = GeneratorService(settings)

        # Document registry path
        self._registry_path = os.path.join("data", _REGISTRY_FILENAME)
        self._ensure_registry()

        logger.info("RAG pipeline ready.")

    # ── Registry helpers ─────────────────────────────────────────────────

    def _ensure_registry(self) -> None:
        """Create the document registry file if it does not exist."""
        Path(os.path.dirname(self._registry_path)).mkdir(
            parents=True, exist_ok=True
        )
        if not os.path.exists(self._registry_path):
            self._save_registry([])

    def _load_registry(self) -> List[Dict]:
        """Load the document registry from disk."""
        try:
            with open(self._registry_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_registry(self, registry: List[Dict]) -> None:
        """Persist the document registry to disk."""
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2, default=str)

    def _add_to_registry(
        self,
        doc_id: str,
        filename: str,
        total_pages: int,
        total_chunks: int,
    ) -> None:
        """Append an entry to the document registry."""
        registry = self._load_registry()
        registry.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )
        self._save_registry(registry)

    def _remove_from_registry(self, doc_id: str) -> bool:
        """Remove a document entry from the registry. Returns True if found."""
        registry = self._load_registry()
        new_registry = [d for d in registry if d["doc_id"] != doc_id]
        if len(new_registry) == len(registry):
            return False
        self._save_registry(new_registry)
        return True

    # ── Public operations ────────────────────────────────────────────────

    @traceable(
        name="document_ingestion_pipeline",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "ingestion"],
    )
    def ingest_document(self, file_path: str, filename: str) -> IngestResponse:
        """Ingest a PDF document into the RAG system.

        Parameters
        ----------
        file_path:
            On-disk location of the uploaded PDF.
        filename:
            Original filename from the upload.

        Returns
        -------
        IngestResponse
            Summary of the ingestion result.

        Raises
        ------
        Exception
            Propagated from lower-level services on failure.
        """
        logger.info("Ingesting document: %s", filename)

        try:
            # 1. Extract + chunk
            doc_id, chunks = self._doc_processor.process_pdf(file_path, filename)

            # 2. Store in vector DB
            count = self._vector_store.add_documents(chunks)

            # 3. Determine total pages from chunk metadata
            total_pages = max(
                (c.metadata.get("total_pages", 1) for c in chunks), default=1
            )

            # 4. Update registry
            self._add_to_registry(doc_id, filename, total_pages, count)

            logger.info(
                "Document ingested: doc_id=%s, pages=%d, chunks=%d",
                doc_id,
                total_pages,
                count,
            )

            return IngestResponse(
                doc_id=doc_id,
                filename=filename,
                total_pages=total_pages,
                total_chunks=count,
                message=f"Successfully ingested '{filename}': {total_pages} pages, {count} chunks.",
            )

        except Exception:
            logger.exception("Failed to ingest document: %s", filename)
            raise

    @traceable(
        name="rag_query_pipeline",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "query"],
    )
    def query(self, request: QueryRequest) -> QueryResponse:
        """Answer a research question using the RAG pipeline.

        Parameters
        ----------
        request:
            The query payload containing the question, optional doc filter,
            and top_k.

        Returns
        -------
        QueryResponse
            The generated answer, source documents, timing, and token usage.
        """
        logger.info("Processing query: %s", request.question[:80])
        start = time.perf_counter()

        try:
            # 1. Retrieve
            sources = self._retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                doc_id=request.doc_id,
            )

            # 2. Generate
            result = self._generator.generate(
                question=request.question,
                sources=sources,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            tokens = result.get("tokens_used", {})

            logger.info("Query processed in %.1f ms", elapsed_ms)

            return QueryResponse(
                answer=result["answer"],
                sources=sources,
                query=request.question,
                processing_time_ms=round(elapsed_ms, 2),
                tokens_used=TokenUsage(**tokens) if tokens else None,
            )

        except Exception:
            logger.exception("Query processing failed")
            raise

    def get_health(self) -> HealthResponse:
        """Return a system health snapshot.

        Returns
        -------
        HealthResponse
            Status, version, document count, and chunk count.
        """
        try:
            registry = self._load_registry()
            chunks_count = self._vector_store.get_document_count()

            return HealthResponse(
                status="healthy",
                version=__version__,
                documents_count=len(registry),
                chunks_count=chunks_count,
            )
        except Exception:
            logger.exception("Health check failed")
            return HealthResponse(
                status="unhealthy",
                version=__version__,
                documents_count=0,
                chunks_count=0,
            )

    def get_documents(self) -> DocumentListResponse:
        """List all ingested documents.

        Returns
        -------
        DocumentListResponse
            A list of ``DocumentInfo`` entries with totals.
        """
        registry = self._load_registry()
        documents = [
            DocumentInfo(
                doc_id=entry["doc_id"],
                filename=entry["filename"],
                total_pages=entry["total_pages"],
                total_chunks=entry["total_chunks"],
                ingested_at=datetime.fromisoformat(entry["ingested_at"]),
            )
            for entry in registry
        ]
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store and registry.

        Parameters
        ----------
        doc_id:
            The document to remove.

        Returns
        -------
        bool
            ``True`` if the document was found and deleted.
        """
        logger.info("Deleting document: %s", doc_id)
        deleted_chunks = self._vector_store.delete_document(doc_id)
        removed = self._remove_from_registry(doc_id)

        if removed or deleted_chunks:
            logger.info(
                "Document deleted: doc_id=%s (%d chunks removed)",
                doc_id,
                deleted_chunks,
            )
            return True

        logger.warning("Document not found: doc_id=%s", doc_id)
        return False
