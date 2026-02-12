import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "17.1.1"
    
    GEMINI_API_KEY: str = "" 
    GOOGLE_CLIENT_ID: str = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str = "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"
    SESSION_SECRET: str = "scholar-pro-super-secret-key"
    
    LLM_PROVIDER: str = "ollama" 
    MODEL_NAME: str = "llama3" 
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()
