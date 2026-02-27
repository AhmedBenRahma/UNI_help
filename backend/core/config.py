"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

    OPENAI_API_KEY: str
    ADMIN_KEY: str = Field(default="change-me-admin-key")
    UNIVERSITY_NAME: str = Field(default="Your University")
    MODEL_NAME: str = Field(default="gpt-4o-mini")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=100)
    MAX_RETRIEVED_DOCS: int = Field(default=5)
    CONFIDENCE_THRESHOLD: float = Field(default=0.75)
    VECTORSTORE_PATH: str = Field(default="data/vectorstore")
    DOCUMENTS_PATH: str = Field(default="data/documents")
    LOG_LEVEL: str = Field(default="INFO")
    CORS_ORIGINS: str = Field(default="*")

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def vectorstore_path(self) -> Path:
        """Return vectorstore path as Path object."""
        return Path(self.VECTORSTORE_PATH)

    @property
    def documents_path(self) -> Path:
        """Return documents path as Path object."""
        return Path(self.DOCUMENTS_PATH)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
