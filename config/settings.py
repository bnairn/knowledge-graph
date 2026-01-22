"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")

    # Google OAuth
    google_credentials_file: Path = Field(default=Path("credentials.json"))
    google_token_file: Path = Field(default=Path("token.json"))

    # LLM Provider (ollama or claude)
    llm_provider: str = Field(default="ollama")

    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_timeout: float = Field(default=300.0)  # 5 minutes for slow inference

    # Claude API (alternative)
    anthropic_api_key: str = Field(default="")
    claude_model: str = Field(default="claude-sonnet-4-20250514")
    claude_max_tokens: int = Field(default=4096)
    claude_requests_per_minute: int = Field(default=50)

    # Processing
    batch_size: int = Field(default=100)
    max_document_size_mb: int = Field(default=10)

    # Sync
    sync_state_file: Path = Field(default=Path(".sync_state.json"))

    # Email filtering - comma-separated list of sender patterns to skip
    skip_senders: str = Field(
        default="amazon,ups,fedex,walmart,target,costco,ebay,paypal,venmo,doordash,uber,lyft,grubhub,instacart,postmates,shipment,tracking,no-reply,noreply,do-not-reply,mailer-daemon"
    )

    def get_skip_senders_list(self) -> list[str]:
        """Get skip senders as a list."""
        return [s.strip().lower() for s in self.skip_senders.split(",") if s.strip()]

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Path | None = Field(default=None)

    @field_validator("log_file", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
