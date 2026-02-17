import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from google.oauth2 import service_account
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str = None):
        p = settings.LLM_PROVIDER.lower()
        m = model_name or settings.MODEL_NAME
        
        # Priority for Gemini models
        if m and m.startswith("gemini-"):
            if settings.GEMINI_API_KEY:
                return ChatGoogleGenerativeAI(model=m, google_api_key=settings.GEMINI_API_KEY)
            
            # Use Service Account if available
            if os.path.exists(settings.GOOGLE_SERVICE_ACCOUNT_JSON):
                try:
                    creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_SERVICE_ACCOUNT_JSON)
                    return ChatGoogleGenerativeAI(model=m, credentials=creds)
                except Exception as e:
                    logger.error(f"Failed to load service account: {e}")
            
            return ChatGoogleGenerativeAI(model=m)

        if p == "ollama":
            return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=m)
        
        # Fallback
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY or None)

def get_llm(model_name=None):
    return LLMFactory.create_llm(model_name)
