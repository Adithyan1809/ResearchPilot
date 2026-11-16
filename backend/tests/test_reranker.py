"""Unit and integration tests for Cross-Encoder Reranker and Two-Stage Retrieval.

Validates:
- Service initialization and configuration loading
- LLM scoring and candidate re-ordering
- Top-K candidate pruning
- Graceful fallbacks for empty candidate pools and LLM failures
- JSON parsing resiliency (markdown code fences, regex fallbacks)
- End-to-end RAGPipeline two-stage retrieval integration
"""

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from backend.app.config import Settings
from backend.app.models.schemas import QueryRequest, SourceDocument
from backend.app.services.rag_pipeline import RAGPipeline
from backend.app.services.reranker import RerankerService


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def default_settings() -> Settings:
    """Settings with reranker enabled and default parameters."""
    return Settings(
        google_api_key="test-api-key",
        reranker_enabled=True,
        reranker_candidates_k=15,
        reranker_top_k=3,
        reranker_model="gemini-3.5-flash-lite",
    )


@pytest.fixture
def reranker(default_settings: Settings) -> RerankerService:
    """RerankerService instance using default test settings."""
    return RerankerService(default_settings)


def create_candidate_documents(count: int = 5) -> List[SourceDocument]:
    """Helper to generate mock retrieved candidate documents."""
    candidates = []
    for i in range(count):
        candidates.append(
            SourceDocument(
                content=f"Candidate chunk content {i} evaluating transformer attention.",
                page_number=i + 1,
                chunk_index=i,
                relevance_score=0.90 - (i * 0.05),
                doc_id=f"doc_{i}",
                filename=f"paper_{i}.pdf",
                retrieval_method="hybrid_rrf",
            )
        )
    return candidates


# ── Unit Tests: RerankerService ──────────────────────────────────────────────

def test_reranker_initialization(default_settings: Settings):
    """Verify RerankerService initializes attributes from settings."""
    service = RerankerService(default_settings)
    assert service.model_name == "gemini-3.5-flash-lite"
    assert service.top_k == 3
    assert service.settings == default_settings
    assert service.llm is not None
    assert service.last_rerank_latency_ms == 0.0

    # Custom configuration
    custom_settings = Settings(
        google_api_key="custom-key",
        reranker_model="gemini-2.0-flash-exp",
        reranker_top_k=5,
    )
    custom_service = RerankerService(custom_settings)
    assert custom_service.model_name == "gemini-2.0-flash-exp"
    assert custom_service.top_k == 5


def test_rerank_scoring_and_ordering(reranker: RerankerService):
    """Verify high-scoring chunks are moved to the top based on LLM output."""
    candidates = create_candidate_documents(3)
    # candidates[0]: chunk_index 0, orig_relevance 0.90
    # candidates[1]: chunk_index 1, orig_relevance 0.85
    # candidates[2]: chunk_index 2, orig_relevance 0.80

    # Mock LLM to score chunk 1 highest (0.96), chunk 2 second (0.72), chunk 0 lowest (0.28)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps([
            {"index": 0, "score": 0.28},
            {"index": 1, "score": 0.96},
            {"index": 2, "score": 0.72},
        ])
    )
    reranker.llm = mock_llm

    query = "What is the key mechanism in transformer architectures?"
    results = reranker.rerank(query=query, candidates=candidates, top_k=3)

    assert len(results) == 3

    # Rank 1: Chunk 1
    assert results[0].chunk_index == 1
    assert results[0].rerank_score == 0.96
    assert results[0].relevance_score == 0.96
    assert results[0].filename == "paper_1.pdf"

    # Rank 2: Chunk 2
    assert results[1].chunk_index == 2
    assert results[1].rerank_score == 0.72
    assert results[1].relevance_score == 0.72

    # Rank 3: Chunk 0
    assert results[2].chunk_index == 0
    assert results[2].rerank_score == 0.28
    assert results[2].relevance_score == 0.28

    mock_llm.invoke.assert_called_once()
    assert reranker.last_rerank_latency_ms >= 0.0


def test_rerank_top_k_trimming(reranker: RerankerService):
    """Verify candidate list is trimmed down to the requested top_k."""
    candidates = create_candidate_documents(10)

    # Mock LLM scoring all 10 candidates
    scores_payload = [{"index": i, "score": round(0.1 * i, 2)} for i in range(10)]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(scores_payload))
    reranker.llm = mock_llm

    # Request top_k = 3 from 10 candidates
    results = reranker.rerank(query="Attention is all you need", candidates=candidates, top_k=3)

    assert len(results) == 3
    # Candidate 9 had score 0.9, candidate 8 had 0.8, candidate 7 had 0.7
    assert results[0].chunk_index == 9
    assert results[0].rerank_score == 0.90
    assert results[1].chunk_index == 8
    assert results[1].rerank_score == 0.80
    assert results[2].chunk_index == 7
    assert results[2].rerank_score == 0.70

    # Test top_k=None defaults to settings.reranker_top_k (3)
    results_default_k = reranker.rerank(query="Attention", candidates=candidates, top_k=None)
    assert len(results_default_k) == 3


def test_rerank_empty_candidates_fallback(reranker: RerankerService):
    """Verify graceful handling and short-circuit when candidate list is empty."""
    mock_llm = MagicMock()
    reranker.llm = mock_llm

    results = reranker.rerank(query="Empty search test", candidates=[], top_k=3)

    assert results == []
    # LLM must not be invoked on empty candidates
    mock_llm.invoke.assert_not_called()


def test_rerank_llm_failure_fallback(reranker: RerankerService):
    """Verify graceful fallback preserving original ranking when LLM raises exception."""
    candidates = create_candidate_documents(5)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Simulated Gemini API timeout / connection error")
    reranker.llm = mock_llm

    results = reranker.rerank(
        query="Robustness fallback test",
        candidates=candidates,
        top_k=3,
    )

    # Must not raise an exception, and must preserve candidate order up to top_k
    assert len(results) == 3
    assert results[0].chunk_index == 0
    assert results[1].chunk_index == 1
    assert results[2].chunk_index == 2
    assert results[0].relevance_score == candidates[0].relevance_score


def test_rerank_json_code_fence_parsing(reranker: RerankerService):
    """Verify LLM responses enclosed in markdown code fences are parsed correctly."""
    candidates = create_candidate_documents(2)

    markdown_response = (
        "```json\n"
        "[\n"
        '  {"index": 0, "score": 0.45},\n'
        '  {"index": 1, "score": 0.89}\n'
        "]\n"
        "```"
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=markdown_response)
    reranker.llm = mock_llm

    results = reranker.rerank(query="Markdown JSON test", candidates=candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_index == 1
    assert results[0].rerank_score == 0.89
    assert results[1].chunk_index == 0
    assert results[1].rerank_score == 0.45


def test_rerank_score_clamping(reranker: RerankerService):
    """Verify scores out of bounds [0.0, 1.0] are safely clamped."""
    candidates = create_candidate_documents(2)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='[{"index": 0, "score": 1.55}, {"index": 1, "score": -0.42}]'
    )
    reranker.llm = mock_llm

    results = reranker.rerank(query="Score clamping test", candidates=candidates, top_k=2)

    assert results[0].chunk_index == 0
    assert results[0].rerank_score == 1.0  # Clamped to 1.0
    assert results[1].chunk_index == 1
    assert results[1].rerank_score == 0.0  # Clamped to 0.0


# ── Integration Tests: RAGPipeline Two-Stage Retrieval ───────────────────────

@patch("backend.app.services.rag_pipeline.DocumentProcessor")
@patch("backend.app.services.rag_pipeline.EmbeddingService")
@patch("backend.app.services.rag_pipeline.VectorStoreService")
@patch("backend.app.services.rag_pipeline.RetrieverService")
@patch("backend.app.services.rag_pipeline.RerankerService")
@patch("backend.app.services.rag_pipeline.GeneratorService")
def test_rag_pipeline_two_stage_integration(
    mock_generator_cls,
    mock_reranker_cls,
    mock_retriever_cls,
    mock_vector_store_cls,
    mock_embedding_cls,
    mock_doc_processor_cls,
    tmp_path,
):
    """Verify RAGPipeline orchestrates Stage 1 (candidates_k) -> Stage 2 reranker -> Generator."""
    settings = Settings(
        google_api_key="mock-key",
        reranker_enabled=True,
        reranker_candidates_k=15,
        reranker_top_k=3,
    )

    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_generator = MagicMock()

    mock_retriever_cls.return_value = mock_retriever
    mock_reranker_cls.return_value = mock_reranker
    mock_generator_cls.return_value = mock_generator

    # 15 candidate documents returned by stage 1 retriever
    candidates_15 = create_candidate_documents(15)
    mock_retriever.retrieve.return_value = candidates_15

    # 3 reranked source documents returned by stage 2 reranker
    reranked_3 = [
        SourceDocument(
            content="Reranked high-relevance chunk 0",
            page_number=1,
            chunk_index=0,
            relevance_score=0.98,
            rerank_score=0.98,
            doc_id="doc_top",
            filename="top_paper.pdf",
        ),
        SourceDocument(
            content="Reranked high-relevance chunk 1",
            page_number=2,
            chunk_index=1,
            relevance_score=0.91,
            rerank_score=0.91,
            doc_id="doc_top",
            filename="top_paper.pdf",
        ),
        SourceDocument(
            content="Reranked high-relevance chunk 2",
            page_number=3,
            chunk_index=2,
            relevance_score=0.85,
            rerank_score=0.85,
            doc_id="doc_top",
            filename="top_paper.pdf",
        ),
    ]
    mock_reranker.rerank.return_value = reranked_3

    mock_generator.generate.return_value = {
        "answer": "Cross-encoder reranking focuses attention on relevant context [Source 1].",
        "tokens_used": {"prompt_tokens": 850, "completion_tokens": 120, "total_tokens": 970},
    }

    with patch.object(RAGPipeline, "_ensure_registry"):
        pipeline = RAGPipeline(settings)
        pipeline._registry_path = str(tmp_path / "document_registry.json")

        request = QueryRequest(
            question="How does cross-encoder reranking improve retrieval quality?",
            top_k=3,
            doc_id="doc_top",
            retrieval_mode="hybrid",
        )

        response = pipeline.query(request)

        # 1. Retriever must be called with candidate pool size (reranker_candidates_k = 15)
        mock_retriever.retrieve.assert_called_once_with(
            query="How does cross-encoder reranking improve retrieval quality?",
            top_k=15,
            doc_id="doc_top",
            retrieval_mode="hybrid",
        )

        # 2. Reranker must be called with candidates and target top_k (3)
        mock_reranker.rerank.assert_called_once_with(
            query="How does cross-encoder reranking improve retrieval quality?",
            candidates=candidates_15,
            top_k=3,
        )

        # 3. Generator must receive the 3 reranked sources
        mock_generator.generate.assert_called_once_with(
            question="How does cross-encoder reranking improve retrieval quality?",
            sources=reranked_3,
        )

        # 4. Verify response fields
        assert "Cross-encoder reranking focuses attention" in response.answer
        assert len(response.sources) == 3
        assert response.sources[0].rerank_score == 0.98
        assert response.tokens_used.total_tokens == 970


@patch("backend.app.services.rag_pipeline.DocumentProcessor")
@patch("backend.app.services.rag_pipeline.EmbeddingService")
@patch("backend.app.services.rag_pipeline.VectorStoreService")
@patch("backend.app.services.rag_pipeline.RetrieverService")
@patch("backend.app.services.rag_pipeline.RerankerService")
@patch("backend.app.services.rag_pipeline.GeneratorService")
def test_rag_pipeline_reranker_disabled_bypass(
    mock_generator_cls,
    mock_reranker_cls,
    mock_retriever_cls,
    mock_vector_store_cls,
    mock_embedding_cls,
    mock_doc_processor_cls,
    tmp_path,
):
    """Verify that when reranker_enabled=False, reranking is bypassed and retriever is called directly."""
    settings = Settings(
        google_api_key="mock-key",
        reranker_enabled=False,
        top_k=5,
    )

    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_generator = MagicMock()

    mock_retriever_cls.return_value = mock_retriever
    mock_reranker_cls.return_value = mock_reranker
    mock_generator_cls.return_value = mock_generator

    stage1_sources = create_candidate_documents(5)
    mock_retriever.retrieve.return_value = stage1_sources
    mock_generator.generate.return_value = {
        "answer": "Direct retrieval response without reranking.",
        "tokens_used": {"prompt_tokens": 1500, "completion_tokens": 100, "total_tokens": 1600},
    }

    with patch.object(RAGPipeline, "_ensure_registry"):
        pipeline = RAGPipeline(settings)
        pipeline._registry_path = str(tmp_path / "document_registry.json")

        request = QueryRequest(
            question="Explain basic RAG without reranking",
            top_k=5,
        )

        response = pipeline.query(request)

        # Retriever called with top_k = 5
        mock_retriever.retrieve.assert_called_once_with(
            query="Explain basic RAG without reranking",
            top_k=5,
            doc_id=None,
            retrieval_mode=None,
        )

        # Reranker must NOT be called
        mock_reranker.rerank.assert_not_called()

        # Generator receives Stage 1 sources directly
        mock_generator.generate.assert_called_once_with(
            question="Explain basic RAG without reranking",
            sources=stage1_sources,
        )
        assert len(response.sources) == 5
