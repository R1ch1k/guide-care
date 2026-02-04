from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    DATABASE_URL: str
    LANGGRAPH_API_URL: Optional[str] = None
    LANGGRAPH_API_KEY: Optional[str] = None
    LANGGRAPH_WORKFLOW_ID: Optional[str] = None

settings = Settings()
