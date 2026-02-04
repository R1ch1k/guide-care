from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://guidecare:guidecare@localhost:5432/guidecare"
    LANGGRAPH_API_URL: Optional[str] = None
    LANGGRAPH_API_KEY: Optional[str] = None
    LANGGRAPH_WORKFLOW_ID: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
