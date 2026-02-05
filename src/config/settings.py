from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "LocalRAG"
    VERSION: str = "1.0.0"

    # Vector DB Settings
    CHROMA_PERSIST_DIR: str = "./rag_db"
    COLLECTION_NAME: str = "user_docs"

    # Model Settings
    LLM_MODEL: str = "llama-3.2-1b"          # LiteLLM alias
    VISION_MODEL: str = "llava"              # For PDF images, diagrams
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Retrieval Settings
    RETRIEVAL_K: int = 5                     # R@5 sweet spot
    MAX_CONTEXT_DOCS: int = 4                # prevents long noisy queries

    # Chunking Settings (Advanced)
    CHUNK_SIZE: int = 500                    # Characters (not words)
    CHUNK_OVERLAP: int = 100                 # Character overlap between chunks
    CHUNK_STRATEGY: str = "recursive"        # Options: recursive, sentence, fixed
    
    # Hybrid Search Settings
    HYBRID_SEARCH_ENABLED: bool = True       # Enable BM25 + Vector hybrid search
    HYBRID_SEMANTIC_WEIGHT: float = 0.7      # Weight for semantic search (0-1)
    HYBRID_BM25_WEIGHT: float = 0.3          # Weight for BM25 keyword search (0-1)

    # Langfuse Settings
    LANGFUSE_SECRET_KEY: str = "sk-lf-79573c9a-e98b-416f-ae54-6d84e1c90a09"
    LANGFUSE_PUBLIC_KEY: str = "pk-lf-3b16fa78-c1c7-448b-b5ed-4af4fdc158aa"
    LANGFUSE_BASE_URL: str = "http://127.0.0.1:3000"
    LANGFUSE_ENABLED: bool = True

    # LiteLLM Gateway
    LITELLM_BASE_URL: str = "http://127.0.0.1:4000"

    # File Paths
    UPLOAD_DIR: Path = Path("./uploaded_docs")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
