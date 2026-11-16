"""Integration tests for FastAPI endpoints."""

import os
import sys
from unittest.mock import MagicMock

os.environ["GOOGLE_API_KEY"] = "mock-key-for-tests"

# Ensure external heavy AI libraries can be mocked during testing
for mod in ["langchain_google_genai", "langchain_chroma", "chromadb"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import (
    HealthResponse,
    IngestResponse,
    QueryResponse,
    SourceDocument,
)


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.get_health.return_value = HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_count=1,
        chunks_count=10,
    )
    pipeline.get_documents.return_value = {
        "documents": [],
        "total": 0,
    }
    pipeline.query.return_value = QueryResponse(
        answer="The Transformer relies on self-attention [Source 1].",
        sources=[
            SourceDocument(
                content="The Transformer is based entirely on attention mechanisms.",
                page_number=1,
                chunk_index=0,
                relevance_score=0.92,
                doc_id="doc_1",
                filename="transformer.pdf",
            )
        ],
        query="What is the Transformer?",
        processing_time_ms=120.5,
    )
    pipeline.ingest_document.return_value = IngestResponse(
        doc_id="test_doc_abc",
        filename="test.pdf",
        total_pages=5,
        total_chunks=15,
        message="Successfully ingested 'test.pdf'",
    )
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    app.state.pipeline = mock_pipeline
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["documents_count"] == 1
    assert data["chunks_count"] == 10


def test_list_documents_endpoint(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


def test_query_validation_short_question(client):
    response = client.post("/api/v1/query", json={"question": "hi"})
    assert response.status_code == 422  # Pydantic validation error: min_length=3


def test_query_successful(client, mock_pipeline):
    response = client.post(
        "/api/v1/query",
        json={"question": "What is the primary contribution?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "Transformer" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["relevance_score"] == 0.92
    assert data["processing_time_ms"] > 0


def test_ingest_rejects_non_pdf(client):
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


def test_nonexistent_endpoint_returns_404(client):
    response = client.get("/api/v1/unknown_route")
    assert response.status_code == 404
