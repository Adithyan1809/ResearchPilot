"""Embedding service wrapping Google Generative AI Embeddings.

Provides a thin, testable layer over ``langchain_google_genai`` so that the
rest of the application never touches the embedding provider directly.
"""

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Manage a ``GoogleGenerativeAIEmbeddings`` instance for the RAG pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the embedding model.

        Parameters
        ----------
        settings:
            Application settings providing the model name, API key, and
            embedding dimensions.
        """
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        logger.info(
            "EmbeddingService initialised (model=%s, dims=%d)",
            settings.embedding_model,
            settings.embedding_dimensions,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Return the underlying embeddings instance.

        This is useful when other LangChain components (e.g. Chroma) need
        the embeddings object directly.
        """
        return self._embeddings

    @traceable(
        name="query_embedding",
        run_type="embedding",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "embedding"],
    )
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Parameters
        ----------
        text:
            The query text to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        logger.debug("Embedding query (len=%d chars)", len(text))
        return self._embeddings.embed_query(text)

    @traceable(
        name="documents_embedding",
        run_type="embedding",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "embedding"],
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts.

        Parameters
        ----------
        texts:
            List of document strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        logger.debug("Embedding %d documents", len(texts))
        return self._embeddings.embed_documents(texts)
