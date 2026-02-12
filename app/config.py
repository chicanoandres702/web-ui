import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    Centralized application configuration.
    Reads from environment variables or .env file.
    """
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "17.1.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security / Auth
    API_SECRET_KEY: str = "change-me-in-production"
    SESSION_SECRET: str = "scholar-pro-super-secret-key"
    GOOGLE_CLIENT_ID: str | None = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str | None = "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"

    # LLM Configuration
    LLM_PROVIDER: str = "gemini"  # gemini, openai, ollama
    GEMINI_API_KEY: str | None = None
    GEMINI_SERVICE_ACCOUNT_FILE: str | None = None # Path to service account file
    OPENAI_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "gemini-flash-latest" 

    # Browser Settings
    HEADLESS: bool = False
    DISABLE_SECURITY: bool = True
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()