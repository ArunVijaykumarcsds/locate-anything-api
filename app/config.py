"""
Application configuration — loaded from environment variables or .env file.
pydantic-settings handles all parsing; no .env file is required (defaults apply).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = "nvidia/LocateAnything-3B"
    """HuggingFace repo ID or absolute local path to the model directory."""

    DEVICE: str = "cuda"
    """Inference device: 'cuda' (recommended) or 'cpu' (testing only, very slow)."""

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
    )


# Instantiate once — safe because pydantic-settings never raises on missing .env
settings = Settings()
