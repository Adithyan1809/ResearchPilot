"""Pydantic v2 schemas for request / response models.

Every API boundary is defined here so that FastAPI auto-generates accurate
OpenAPI documentation and validates payloads at the edge.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Chat & Conversational Models ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single turn in a conversational exchange."""

    role: str = Field(..., description="Message author role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Text content of the message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the message was recorded",
    )


class SessionInfo(BaseModel):
    """Summary of an active conversational session."""

    session_id: str = Field(..., description="Unique session identifier")
    message_count: int = Field(..., ge=0, description="Total messages in session history")
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC last active timestamp")


class SessionResponse(BaseModel):
    """Detailed view of a session including full message history."""

    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Chronological message history",
    )
    created_at: datetime = Field(..., description="UTC creation timestamp")
    updated_at: datetime = Field(..., description="UTC last active timestamp")


class SessionListResponse(BaseModel):
    """List of all tracked active sessions."""

    sessions: List[SessionInfo] = Field(
        default_factory=list,
        description="List of session summaries",
    )
    total: int = Field(..., ge=0, description="Total active sessions")


# ── Document Metadata ────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata attached to every chunk stored in the vector database."""

    doc_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    page_number: int = Field(..., ge=1, description="1-indexed page number in the PDF")
    chunk_index: int = Field(..., ge=0, description="0-indexed chunk position within the page")
    total_pages: int = Field(..., ge=1, description="Total number of pages in the PDF")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when the document was ingested",
    )


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Marker schema for PDF ingestion (file arrives via ``UploadFile``)."""
    pass


class IngestResponse(BaseModel):
    """Response returned after successfully ingesting a PDF."""

    doc_id: str = Field(..., description="Unique identifier assigned to the document")
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., ge=1, description="Number of pages extracted")
    total_chunks: int = Field(..., ge=1, description="Number of text chunks created")
    message: str = Field(..., description="Human-readable status message")


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload for submitting a question to the RAG pipeline."""

    question: str = Field(
        ...,
        min_length=3,
        description="The research question to answer",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Optional document ID to restrict retrieval scope",
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve",
    )
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Retrieval strategy: 'hybrid' (default), 'dense', or 'bm25'",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversational memory continuity",
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Optional list of prior conversation turns for multi-turn reasoning",
    )


class SourceDocument(BaseModel):
    """A single retrieved chunk returned alongside the generated answer."""

    content: str = Field(..., description="Text content of the chunk")
    page_number: int = Field(..., ge=1, description="Page the chunk originates from")
    chunk_index: int = Field(..., ge=0, description="Chunk position index")
    relevance_score: float = Field(..., description="Relevance or RRF score")
    doc_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename of the source document")
    retrieval_method: Optional[str] = Field(
        default="hybrid_rrf",
        description="Retrieval channel: 'hybrid_rrf', 'dense', or 'bm25'",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal rank fusion score if hybrid retrieval was used",
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score (0.0 to 1.0) after reranking",
    )


class TokenUsage(BaseModel):
    """Token consumption breakdown for a single generation call."""

    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens consumed")


class QueryResponse(BaseModel):
    """Full response to a user query, including the answer and sources."""

    answer: str = Field(..., description="Generated answer with inline citations")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved source chunks used for generation",
    )
    query: str = Field(..., description="Original query string")
    processing_time_ms: float = Field(
        ...,
        description="End-to-end processing time in milliseconds",
    )
    tokens_used: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage breakdown if available",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Active conversation session ID if tracking conversation memory",
    )


# ── Health / Admin ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(..., description="Current system status")
    version: str = Field(..., description="Application version string")
    documents_count: int = Field(..., ge=0, description="Number of ingested documents")
    chunks_count: int = Field(..., ge=0, description="Total chunks in vector store")


class DocumentInfo(BaseModel):
    """Summary information for a single ingested document."""

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., ge=1, description="Total pages in the PDF")
    total_chunks: int = Field(..., ge=1, description="Number of chunks stored")
    ingested_at: datetime = Field(..., description="UTC ingestion timestamp")


class DocumentListResponse(BaseModel):
    """Paginated list of all ingested documents."""

    documents: List[DocumentInfo] = Field(
        default_factory=list,
        description="List of document summaries",
    )
    total: int = Field(..., ge=0, description="Total number of documents")


# ── Errors ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned on failure."""

    error: str = Field(..., description="Short error code or title")
    detail: str = Field(..., description="Human-readable error explanation")
