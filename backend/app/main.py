"""ResearchLens FastAPI application entry point.

Creates the FastAPI application instance, configures middleware, mounts the
frontend static files, and wires up the API router.  The ``lifespan`` context
manager handles startup/shutdown logic (directory creation, LangSmith env
vars, RAG pipeline initialisation).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import Settings, get_settings
from app.services.rag_pipeline import RAGPipeline
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle handler."""
    settings: Settings = get_settings()

    # Ensure required directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    logger.info("Data directories ready (uploads=%s, chroma=%s)", settings.upload_dir, settings.chroma_persist_dir)

    # Configure LangSmith observability env vars
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint or "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_AGENT_ID"] = settings.langsmith_agent_id or "researchpilot"
        os.environ["LANGSMITH_AGENT_ENVIRONMENT"] = settings.langsmith_agent_environment or "local"
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langsmith_tracing else "false"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint or "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

        # Clear project vars when agent routing is active to prevent destination conflict
        if settings.langsmith_agent_id:
            os.environ.pop("LANGSMITH_PROJECT", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
        elif settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

        logger.info(
            "LangSmith tracing enabled: agent_id='%s', env='%s', endpoint='%s'",
            settings.langsmith_agent_id,
            settings.langsmith_agent_environment,
            settings.langsmith_endpoint,
        )
    else:
        logger.info("LangSmith tracing disabled (no API key configured)")

    # Ensure Google API key is in environment
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    # Initialize the RAG pipeline and attach to app state
    try:
        app.state.pipeline = RAGPipeline(settings)
        logger.info("RAG pipeline initialized successfully")
    except Exception as exc:
        logger.error("Failed to initialize RAG pipeline: %s", exc)
        raise

    logger.info("🚀 ResearchLens V2.0.0 (Hybrid RRF) started — listening on %s:%s", settings.host, settings.port)

    yield

    # Shutdown
    logger.info("ResearchLens shutting down — goodbye!")


# ── Application Factory ─────────────────────────────────────────────────────


app = FastAPI(
    title="ResearchLens API",
    description=(
        "A production-oriented RAG (Retrieval-Augmented Generation) pipeline "
        "for ingesting research papers and answering questions with cited sources. "
        "Powered by Google Gemini, LangChain, and ChromaDB."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Routes ───────────────────────────────────────────────────────────────

app.include_router(api_router)


# ── Exception Handlers ──────────────────────────────────────────────────────


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Return a JSON error for 404s on API paths; serve index.html otherwise."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"error": "Not Found", "detail": f"The path '{request.url.path}' does not exist."},
        )
    # For non-API paths, try to serve the frontend
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse(status_code=404, content={"error": "Not Found", "detail": "Resource not found."})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Catch-all handler for unhandled server errors."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred."},
    )


# ── Frontend Static Files ───────────────────────────────────────────────────

# Mount the frontend directory AFTER API routes so that API paths take priority.
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    logger.info("Frontend static files mounted from %s", _FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at %s — static files will not be served", _FRONTEND_DIR)
