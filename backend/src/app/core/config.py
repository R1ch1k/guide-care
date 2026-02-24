from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    DATABASE_URL: str

    # Optional external services
    TRIAGE_API_URL: Optional[str] = None
    LOCAL_20B_API_URL: Optional[str] = None

    # Legacy remote LangGraph placeholder (can remain unused)
    LANGGRAPH_API_URL: Optional[str] = None
    LANGGRAPH_API_KEY: Optional[str] = None
    LANGGRAPH_WORKFLOW_ID: Optional[str] = None

    # Runtime knobs
    MODEL_HISTORY_MAX_MESSAGES: int = 20
    AI_TIMEOUT_SECONDS: float = 3.0
    AI_RETRIES: int = 1

    # Comma-separated list
    CORS_ORIGINS: Optional[str] = None


settings = Settings()
