import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "16.0.0-unified"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Auth & Identity
    API_SECRET_KEY: str = "scholar-nexus-ultra-secret-2024"
    GOOGLE_CLIENT_ID: str = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com" 
    GOOGLE_SERVICE_ACCOUNT_JSON: str = "service_account.json"
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama" 
    MODEL_NAME: str = "llama3" 
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GEMINI_API_KEY: str = "" # Add your key if needed
    
    # Storage
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage")
    USER_DATA_DIR: str = os.path.join(os.getcwd(), "storage", "browser_data")
    KNOWLEDGE_BASE_DIR: str = os.path.join(os.getcwd(), "storage", "knowledge_bases")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def ensure_dirs(self):
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        os.makedirs(self.USER_DATA_DIR, exist_ok=True)
        os.makedirs(self.KNOWLEDGE_BASE_DIR, exist_ok=True)

@lru_cache()
def get_settings():
    s = Settings()
    s.ensure_dirs()
    return s
