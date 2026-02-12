import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str = None):
        p = settings.LLM_PROVIDER.lower()
        m = model_name or settings.MODEL_NAME
        
        # Priority for Gemini models
        if m and m.startswith("gemini-"):
            return ChatGoogleGenerativeAI(model=m, google_api_key=settings.GEMINI_API_KEY)

        if p == "ollama":
            return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=m)
        
        # Fallback
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)

def get_llm(model_name=None):
    return LLMFactory.create_llm(model_name)
