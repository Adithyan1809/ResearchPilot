"""Hybrid retrieval service — Dense Semantic Vector + Sparse BM25 Lexical Search.

Combines ChromaDB vector similarity with BM25Okapi lexical matching using
Reciprocal Rank Fusion (RRF) to eliminate dense-only retrieval blind spots
for technical terms, equations, acronyms, and exact keywords.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

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


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 matching."""
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if len(t) > 1]


class RetrieverService:
    """Retrieve and rank source chunks using Dense, BM25, or Hybrid RRF search."""

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
            Application settings (retrieval mode, RRF constant, top_k, etc.).
        """
        self._vector_store = vector_store
        self._settings = settings
        self.last_retrieval_latency_ms: float = 0.0

    # ── Channel 1: Dense Semantic Vector Search ──────────────────────────────

    @traceable(
        name="dense_vector_search",
        run_type="retriever",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "dense_search"],
    )
    def _dense_search(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K candidates using dense vector cosine similarity."""
        logger.debug("Executing dense vector search (k=%d, filter=%s)", k, filter_dict)
        return self._vector_store.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
        )

    # ── Channel 2: Sparse Lexical BM25 Search ────────────────────────────────

    @traceable(
        name="bm25_lexical_search",
        run_type="retriever",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "bm25_search"],
    )
    def _bm25_search(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K candidates using sparse BM25Okapi lexical matching."""
        logger.debug("Executing BM25 lexical search (k=%d, filter=%s)", k, filter_dict)
        all_docs = self._vector_store.get_all_documents(filter_dict=filter_dict)
        if not all_docs:
            logger.debug("No documents found for BM25 search")
            return []

        tokenized_corpus = [_tokenize(doc.page_content) for doc in all_docs]
        tokenized_query = _tokenize(query)

        if not tokenized_query:
            logger.debug("Query produced 0 tokens for BM25 search")
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Pair documents with non-zero or positive BM25 scores
        scored_docs: List[Tuple[Document, float]] = [
            (doc, float(score))
            for doc, score in zip(all_docs, scores)
            if score > 0.0
        ]

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    # ── Channel 3: Reciprocal Rank Fusion (RRF) ──────────────────────────────

    @traceable(
        name="reciprocal_rank_fusion",
        run_type="tool",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "rank_fusion"],
    )
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int,
        rrf_k: int = 60,
        weight_dense: float = 1.0,
        weight_bm25: float = 1.0,
    ) -> List[SourceDocument]:
        """Fuse dense and lexical candidate lists using Reciprocal Rank Fusion.

        Formula:
            RRF_Score(d) = sum_{m in {dense, bm25}} ( weight_m / (rrf_k + rank_m(d)) )

        Parameters
        ----------
        dense_results:
            Ranked (Document, cosine_similarity) tuples.
        bm25_results:
            Ranked (Document, bm25_score) tuples.
        k:
            Number of top fused documents to return.
        rrf_k:
            Smoothing constant to mitigate high rank bias (standard = 60).
        weight_dense:
            Relative importance weight for dense channel.
        weight_bm25:
            Relative importance weight for lexical channel.

        Returns
        -------
        list[SourceDocument]
            Fused documents ranked by descending RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}
        origin_channels: Dict[str, List[str]] = {}

        def _doc_key(doc: Document) -> str:
            # Unique identifier combining doc_id and chunk_index or content hash
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id", "unknown")
            chunk_idx = meta.get("chunk_index", "0")
            return f"{doc_id}:{chunk_idx}"

        # 1. Process Dense Rankings
        for rank, (doc, _) in enumerate(dense_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight_dense / (rrf_k + rank))
            doc_store[key] = doc
            origin_channels.setdefault(key, []).append("dense")

        # 2. Process BM25 Rankings
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight_bm25 / (rrf_k + rank))
            doc_store[key] = doc
            origin_channels.setdefault(key, []).append("bm25")

        # 3. Sort by descending fused RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]

        fused_sources: List[SourceDocument] = []
        for key in sorted_keys:
            doc = doc_store[key]
            fused_score = rrf_scores[key]
            channels = origin_channels.get(key, [])
            method_label = "hybrid_rrf (dense + bm25)" if len(channels) > 1 else channels[0]

            fused_sources.append(
                SourceDocument(
                    content=doc.page_content,
                    page_number=doc.metadata.get("page_number", 1),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=round(float(fused_score), 4),
                    doc_id=doc.metadata.get("doc_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    retrieval_method=method_label,
                    rrf_score=round(float(fused_score), 5),
                )
            )

        logger.info(
            "RRF fusion complete: %d dense candidates, %d bm25 candidates -> %d fused results",
            len(dense_results),
            len(bm25_results),
            len(fused_sources),
        )
        return fused_sources

    # ── Public API ───────────────────────────────────────────────────────────

    @traceable(
        name="hybrid_similarity_retrieval",
        run_type="retriever",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "retrieval", "hybrid"],
    )
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[SourceDocument]:
        """Retrieve and rank source chunks using the selected retrieval strategy.

        Parameters
        ----------
        query:
            Natural-language question.
        top_k:
            Number of chunks to retrieve.
        doc_id:
            If provided, restrict retrieval to this document.
        retrieval_mode:
            Optional override for retrieval mode: "hybrid", "dense", or "bm25".

        Returns
        -------
        list[SourceDocument]
            Chunks sorted by descending relevance / RRF score.
        """
        mode = (retrieval_mode or self._settings.retrieval_mode or "hybrid").lower()
        filter_dict = {"doc_id": doc_id} if doc_id else None

        logger.info(
            "Retrieving top-%d chunks (mode=%s, doc_id=%s)", top_k, mode, doc_id
        )

        start = time.perf_counter()

        if mode == "hybrid":
            # Multiplier for candidate generation before fusion
            cand_k = top_k * getattr(self._settings, "hybrid_candidate_multiplier", 2)
            dense_candidates = self._dense_search(query, k=cand_k, filter_dict=filter_dict)
            bm25_candidates = self._bm25_search(query, k=cand_k, filter_dict=filter_dict)

            # If BM25 returned no matches (e.g. no exact token overlaps), fall back to dense
            if not bm25_candidates and dense_candidates:
                sources = [
                    SourceDocument(
                        content=doc.page_content,
                        page_number=doc.metadata.get("page_number", 1),
                        chunk_index=doc.metadata.get("chunk_index", 0),
                        relevance_score=round(float(score), 4),
                        doc_id=doc.metadata.get("doc_id", ""),
                        filename=doc.metadata.get("filename", ""),
                        retrieval_method="dense (fallback)",
                        rrf_score=None,
                    )
                    for doc, score in dense_candidates[:top_k]
                ]
            else:
                sources = self._reciprocal_rank_fusion(
                    dense_results=dense_candidates,
                    bm25_results=bm25_candidates,
                    k=top_k,
                    rrf_k=getattr(self._settings, "rrf_k", 60),
                    weight_dense=getattr(self._settings, "hybrid_dense_weight", 1.0),
                    weight_bm25=getattr(self._settings, "hybrid_bm25_weight", 1.0),
                )

        elif mode == "bm25":
            bm25_results = self._bm25_search(query, k=top_k, filter_dict=filter_dict)
            sources = [
                SourceDocument(
                    content=doc.page_content,
                    page_number=doc.metadata.get("page_number", 1),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=round(float(score), 4),
                    doc_id=doc.metadata.get("doc_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    retrieval_method="bm25",
                    rrf_score=None,
                )
                for doc, score in bm25_results
            ]

        else:  # "dense"
            dense_results = self._dense_search(query, k=top_k, filter_dict=filter_dict)
            sources = [
                SourceDocument(
                    content=doc.page_content,
                    page_number=doc.metadata.get("page_number", 1),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=round(float(score), 4),
                    doc_id=doc.metadata.get("doc_id", ""),
                    filename=doc.metadata.get("filename", ""),
                    retrieval_method="dense",
                    rrf_score=None,
                )
                for doc, score in dense_results
            ]

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_retrieval_latency_ms = elapsed_ms

        logger.info(
            "Retrieved %d sources in %.1f ms using %s retrieval",
            len(sources),
            elapsed_ms,
            mode,
        )
        return sources
