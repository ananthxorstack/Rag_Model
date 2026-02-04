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

    # Chunking Settings (Optimized)
    CHUNK_SIZE: int = 420                    # best for nomic embeddings
    CHUNK_OVERLAP: int = 70                  # ideal for 20+ page docs

    # Langfuse Settings
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str = "http://127.0.0.1:3000"
    LANGFUSE_ENABLED: bool = True

    # LiteLLM Gateway
    LITELLM_BASE_URL: str = "http://127.0.0.1:4000"

    # File Paths
    UPLOAD_DIR: Path = Path("./uploaded_docs")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
