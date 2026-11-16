"""Unit and integration tests for Streaming (SSE) and Conversational Chat History."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

os.environ["GOOGLE_API_KEY"] = "mock-key-for-tests"

for mod in ["langchain_google_genai", "langchain_chroma", "chromadb"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import app
from backend.app.models.schemas import (
    ChatMessage,
    QueryRequest,
    QueryResponse,
    SessionResponse,
    SourceDocument,
)
from backend.app.services.generator import GeneratorService
from backend.app.services.rag_pipeline import RAGPipeline


def test_generator_stream_empty_sources():
    """Test generate_stream with no sources yields fallback message."""
    settings = Settings(google_api_key="mock-key")
    with patch("backend.app.services.generator.ChatGoogleGenerativeAI"):
        gen = GeneratorService(settings)
        chunks = list(gen.generate_stream(question="Hello", sources=[]))
        assert len(chunks) == 1
        assert "don't have enough information" in chunks[0]


def test_generator_stream_tokens():
    """Test generate_stream yields token chunks properly."""
    settings = Settings(google_api_key="mock-key")
    with patch("backend.app.services.generator.ChatGoogleGenerativeAI") as mock_cls:
        mock_llm = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Attention "
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "is all "
        mock_chunk3 = MagicMock()
        mock_chunk3.content = "you need."
        mock_llm.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        mock_cls.return_value = mock_llm

        gen = GeneratorService(settings)
        # Patch the internal chain
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        gen._prompt = MagicMock()
        with patch.object(gen, "_prompt") as mock_prompt:
            mock_prompt.__or__.return_value = mock_chain
            sources = [
                SourceDocument(
                    content="Attention is essential.",
                    page_number=1,
                    chunk_index=0,
                    relevance_score=0.9,
                    doc_id="d1",
                    filename="paper.pdf",
                )
            ]
            history = [ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!")]
            deltas = list(gen.generate_stream("What is attention?", sources, history))
            assert deltas == ["Attention ", "is all ", "you need."]


def test_rag_pipeline_query_stream_sse_structure():
    """Test RAGPipeline.query_stream produces well-formed SSE lines."""
    settings = Settings(google_api_key="mock-key", reranker_enabled=False)
    with patch("backend.app.services.rag_pipeline.DocumentProcessor"), \
         patch("backend.app.services.rag_pipeline.EmbeddingService"), \
         patch("backend.app.services.rag_pipeline.VectorStoreService"), \
         patch("backend.app.services.rag_pipeline.RetrieverService") as mock_retriever_cls, \
         patch("backend.app.services.rag_pipeline.GeneratorService") as mock_gen_cls:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            SourceDocument(
                content="Transformer source content",
                page_number=1,
                chunk_index=0,
                relevance_score=0.88,
                doc_id="doc1",
                filename="transformer.pdf",
            )
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_gen = MagicMock()
        mock_gen.generate_stream.return_value = iter(["Self-attention ", "mechanism."])
        mock_gen_cls.return_value = mock_gen

        pipeline = RAGPipeline(settings)
        req = QueryRequest(question="What mechanism does Transformer use?", session_id="test-session-1")

        events = list(pipeline.query_stream(req))
        assert len(events) >= 3

        # First event: sources
        assert events[0].startswith("data: ")
        data_sources = json.loads(events[0][6:])
        assert data_sources["type"] == "sources"
        assert len(data_sources["sources"]) == 1
        assert data_sources["session_id"] == "test-session-1"

        # Middle events: tokens
        data_tok1 = json.loads(events[1][6:])
        assert data_tok1["type"] == "token"
        assert data_tok1["delta"] == "Self-attention "

        data_tok2 = json.loads(events[2][6:])
        assert data_tok2["type"] == "token"
        assert data_tok2["delta"] == "mechanism."

        # Final event: done
        data_done = json.loads(events[3][6:])
        assert data_done["type"] == "done"
        assert data_done["answer"] == "Self-attention mechanism."
        assert "processing_time_ms" in data_done

        # Verify session was updated
        history = pipeline.session_manager.get_history("test-session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[1].content == "Self-attention mechanism."


def test_api_query_stream_and_session_routes():
    """Integration test for API /query/stream and /sessions endpoints."""
    mock_pipeline = MagicMock()

    def sample_stream(req):
        yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'delta': 'Chunk answer'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': 'Chunk answer', 'processing_time_ms': 50})}\n\n"

    mock_pipeline.query_stream.side_effect = sample_stream
    mock_pipeline.create_session.return_value = "sess-123"
    mock_pipeline.list_sessions.return_value = []
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    mock_pipeline.get_session.return_value = SessionResponse(
        session_id="sess-123",
        messages=[ChatMessage(role="user", content="Hi", timestamp=now)],
        created_at=now,
        updated_at=now,
    )
    mock_pipeline.clear_session.return_value = True

    app.state.pipeline = mock_pipeline
    client = TestClient(app)

    # 1. Test POST /query/stream
    response = client.post(
        "/api/v1/query/stream",
        json={"question": "Tell me about transformers", "session_id": "sess-123"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line for line in response.text.split("\n\n") if line.strip()]
    assert len(lines) == 3
    assert "sources" in lines[0]
    assert "token" in lines[1]
    assert "done" in lines[2]

    # 2. Test POST /sessions
    sess_create = client.post("/api/v1/sessions")
    assert sess_create.status_code == 200
    assert sess_create.json()["session_id"] == "sess-123"

    # 3. Test GET /sessions
    sess_list = client.get("/api/v1/sessions")
    assert sess_list.status_code == 200
    assert "sessions" in sess_list.json()

    # 4. Test GET /sessions/{session_id}
    sess_detail = client.get("/api/v1/sessions/sess-123")
    assert sess_detail.status_code == 200
    assert len(sess_detail.json()["messages"]) == 1

    # 5. Test DELETE /sessions/{session_id}
    sess_delete = client.delete("/api/v1/sessions/sess-123")
    assert sess_delete.status_code == 200
    assert "cleared" in sess_delete.json()["message"]
