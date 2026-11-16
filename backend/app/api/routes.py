"""API route definitions for the ResearchLens REST API.

All endpoints are mounted under the ``/api/v1`` prefix via the ``router``
instance exported from this module.
"""

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.config import get_settings
from backend.app.models.schemas import (
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    SessionListResponse,
    SessionResponse,
)
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ResearchLens"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_pipeline(request: Request):
    """Retrieve the RAG pipeline singleton from application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Please try again later.",
        )
    return pipeline


def _validate_pdf(file: UploadFile) -> None:
    """Raise 400 if the uploaded file is not a PDF."""
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a file with a .pdf extension.",
        )
    if file.content_type and file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Expected 'application/pdf'.",
        )


# ── Ingestion ────────────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Ingest a PDF document",
    description="Upload a PDF file to be processed, chunked, embedded, and stored in the vector database.",
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    """Ingest a PDF document into the RAG pipeline."""
    settings = get_settings()
    pipeline = _get_pipeline(request)

    # Validate file type
    _validate_pdf(file)

    # Read file content and validate size
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size ({len(content) / (1024 * 1024):.1f} MB) exceeds "
                f"the maximum allowed size of {settings.max_file_size_mb} MB."
            ),
        )

    safe_filename = file.filename or "document.pdf"
    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = Path(settings.upload_dir) / safe_filename

    try:
        save_path.write_bytes(content)
        logger.info("Saved uploaded file to %s (%d bytes)", save_path, len(content))
    except OSError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file to disk.",
        ) from exc

    # Process through the RAG pipeline
    try:
        response: IngestResponse = pipeline.ingest_document(
            file_path=str(save_path),
            filename=safe_filename,
        )
        return response
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", safe_filename, exc)
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        ) from exc


# ── Query ────────────────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Query the RAG pipeline",
    description="Submit a research question and receive an AI-generated answer with source citations.",
)
async def query_documents(request: Request, body: QueryRequest):
    """Run a query against ingested documents via the RAG pipeline."""
    pipeline = _get_pipeline(request)

    logger.info(
        "Query received: '%s' (doc_id=%s, top_k=%d)",
        body.question,
        body.doc_id,
        body.top_k,
    )

    try:
        response: QueryResponse = pipeline.query(body)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the answer: {exc}",
        ) from exc


@router.post(
    "/query/stream",
    summary="Query the RAG pipeline with SSE streaming",
    description="Submit a question and receive real-time Server-Sent Events (SSE) streaming tokens, sources, and timing.",
)
async def query_documents_stream(request: Request, body: QueryRequest):
    """Run a streaming query against ingested documents via the RAG pipeline."""
    pipeline = _get_pipeline(request)

    logger.info(
        "Streaming query received: '%s' (session_id=%s, top_k=%d)",
        body.question,
        body.session_id,
        body.top_k,
    )

    try:
        return StreamingResponse(
            pipeline.query_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Streaming query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start streaming query: {exc}",
        ) from exc


# ── Conversational Sessions ──────────────────────────────────────────────────


@router.post(
    "/sessions",
    summary="Create a new conversation session",
    description="Create a unique session identifier for tracking multi-turn research conversations.",
)
async def create_session(request: Request):
    """Create a new conversational session."""
    pipeline = _get_pipeline(request)
    session_id = pipeline.create_session()
    return {"session_id": session_id, "message": "Session created successfully."}


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List active conversation sessions",
    description="Retrieve metadata for all currently tracked conversational sessions.",
)
async def list_sessions(request: Request):
    """List all active conversational sessions."""
    pipeline = _get_pipeline(request)
    sessions = pipeline.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get conversation session history",
    description="Retrieve all user and assistant messages for a specific session.",
)
async def get_session(request: Request, session_id: str):
    """Retrieve full message history for a session."""
    pipeline = _get_pipeline(request)
    return pipeline.get_session(session_id)


@router.delete(
    "/sessions/{session_id}",
    summary="Delete conversation session",
    description="Permanently delete a conversation session and its message history.",
)
async def delete_session(request: Request, session_id: str):
    """Delete a conversation session."""
    pipeline = _get_pipeline(request)
    success = pipeline.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return {"session_id": session_id, "message": "Session cleared and deleted successfully."}


# ── Health ───────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current system status, version, and document/chunk counts.",
)
async def health_check(request: Request):
    """Return application health information."""
    pipeline = _get_pipeline(request)
    return pipeline.get_health()


# ── Documents ────────────────────────────────────────────────────────────────


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List ingested documents",
    description="Retrieve metadata for all documents currently stored in the vector database.",
)
async def list_documents(request: Request):
    """List all ingested documents."""
    pipeline = _get_pipeline(request)

    try:
        return pipeline.get_documents()
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document list: {exc}",
        ) from exc


@router.delete(
    "/documents/{doc_id}",
    summary="Delete a document",
    description="Remove a document and all its chunks from the vector database.",
)
async def delete_document(request: Request, doc_id: str):
    """Delete a document and its associated chunks."""
    pipeline = _get_pipeline(request)

    try:
        success = pipeline.delete_document(doc_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found.",
            )
        logger.info("Deleted document %s", doc_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {exc}",
        ) from exc

    return {"message": f"Document '{doc_id}' and its chunks have been deleted successfully."}
