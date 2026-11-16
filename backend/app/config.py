"""Application configuration using Pydantic Settings.

Loads configuration from environment variables and .env file.
All settings are centralized here for easy management.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are resolved in this order (highest priority first):
    1. Environment variables
    2. .env file values
    3. Default values defined here
    """

    # API Keys
    google_api_key: str = ""
    langsmith_api_key: str = ""

    # LangSmith Observability
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = ""
    langsmith_agent_id: str = "researchpilot"
    langsmith_agent_environment: str = "local"

    # Embedding Model
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = 3072

    # LLM Configuration
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # Chunking Strategy
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # "hybrid" | "dense" | "bm25"
    rrf_k: int = 60
    hybrid_dense_weight: float = 1.0
    hybrid_bm25_weight: float = 1.0
    hybrid_candidate_multiplier: int = 2

    # Reranker
    reranker_enabled: bool = True
    reranker_candidates_k: int = 15
    reranker_top_k: int = 3
    reranker_model: str = "gemini-3.5-flash-lite"

    # ChromaDB Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "research_papers"

    # File Upload
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings.

    Using ``@lru_cache`` ensures the .env file is read only once and the
    same ``Settings`` object is reused throughout the application lifetime.
    """
    return Settings()
