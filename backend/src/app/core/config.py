from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    DATABASE_URL: str

    # LLM mode: "api" for OpenAI API, "local" for gpt-oss-20b via local server
    LLM_MODE: str = "api"

    # OpenAI API (used when LLM_MODE=api)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # Local model (used when LLM_MODE=local)
    LOCAL_MODEL_URL: str = "http://localhost:8080/v1"
    LOCAL_MODEL_NAME: str = "gpt-oss-20b"

    # Optional external services
    TRIAGE_API_URL: Optional[str] = None

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
