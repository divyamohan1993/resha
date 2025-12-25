import os
import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


def _generate_secure_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(48)


class Settings(BaseSettings):
    APP_NAME: str = "Resha"
    VERSION: str = "3.0.0"  # Enterprise-Grade Hybrid Multi-Model Architecture
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    
    # AI/ML Thresholds
    MIN_RESUME_LENGTH: int = 50
    THRESHOLD_SCORE: float = 0.70
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024  # 5MB
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH: str = os.path.join(BASE_DIR, "audit.db")
    
    # Database (Default to SQLite if not provided)
    DATABASE_URL: str = ""

    # Security - API_KEY is required and must be set in .env
    # If not set, a random one will be generated (not recommended for production)
    API_KEY: str = ""
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://localhost:3000"]
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    
    # Local LLM (Ollama) Configuration - For CPU-only inference
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_PREFERRED_MODEL: str = ""  # Empty = auto-detect best available
    LOCAL_LLM_ENABLED: bool = True    # Enable CPU-based local LLM
    
    # Development Mode - Enables detailed reasoning chain-of-thought display
    DEVELOPMENT_MODE: bool = False

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate secure API key if not provided
        if not self.API_KEY:
            self.API_KEY = _generate_secure_key()
            # Log warning that a random key was generated
            import logging
            logging.warning(
                f"API_KEY not set in .env - generated random key: {self.API_KEY[:20]}... "
                "Please set API_KEY in .env for production use."
            )


@lru_cache()
def get_settings():
    return Settings()


