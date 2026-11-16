"""Unit tests for Hybrid Retriever (Dense + BM25 RRF)."""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from backend.app.config import Settings
from backend.app.services.retriever import RetrieverService, _tokenize


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    return store


@pytest.fixture
def retriever(mock_vector_store):
    settings = Settings(retrieval_mode="hybrid", rrf_k=60)
    return RetrieverService(mock_vector_store, settings)


def test_tokenize_clean_words():
    text = "The Transformer (Vaswani et al., 2017) uses d_model=512!"
    tokens = _tokenize(text)
    assert "transformer" in tokens
    assert "vaswani" in tokens
    assert "2017" in tokens
    assert "d_model" in tokens or "model" in tokens
    assert "512" in tokens


def test_reciprocal_rank_fusion_boosting(retriever):
    doc1 = Document(page_content="Common chunk", metadata={"doc_id": "d1", "chunk_index": 0})
    doc2 = Document(page_content="Dense only chunk", metadata={"doc_id": "d1", "chunk_index": 1})
    doc3 = Document(page_content="BM25 only chunk", metadata={"doc_id": "d1", "chunk_index": 2})

    dense_candidates = [(doc1, 0.95), (doc2, 0.85)]
    bm25_candidates = [(doc1, 4.2), (doc3, 3.1)]

    fused = retriever._reciprocal_rank_fusion(
        dense_results=dense_candidates,
        bm25_results=bm25_candidates,
        k=3,
        rrf_k=60,
    )

    assert len(fused) == 3
    # doc1 appeared in both rankings, so its RRF score must be highest
    assert fused[0].chunk_index == 0
    assert fused[0].retrieval_method == "hybrid_rrf (dense + bm25)"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_dense_mode_retrieval(retriever, mock_vector_store):
    doc = Document(page_content="Dense test", metadata={"doc_id": "d1", "chunk_index": 0, "page_number": 1})
    mock_vector_store.similarity_search.return_value = [(doc, 0.88)]

    sources = retriever.retrieve(query="attention mechanism", top_k=1, retrieval_mode="dense")

    assert len(sources) == 1
    assert sources[0].retrieval_method == "dense"
    assert sources[0].relevance_score == 0.88
    mock_vector_store.similarity_search.assert_called_once()


def test_bm25_fallback_when_empty_lexical_results(retriever, mock_vector_store):
    doc = Document(page_content="Fallback dense test", metadata={"doc_id": "d1", "chunk_index": 0, "page_number": 1})
    mock_vector_store.similarity_search.return_value = [(doc, 0.75)]
    # All documents have no keyword match with gibberish query
    mock_vector_store.get_all_documents.return_value = [
        Document(page_content="Completely different words", metadata={"doc_id": "d1", "chunk_index": 0})
    ]

    sources = retriever.retrieve(query="xyzzyqwerty", top_k=1, retrieval_mode="hybrid")

    assert len(sources) == 1
    assert "dense" in sources[0].retrieval_method
