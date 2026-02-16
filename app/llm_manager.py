import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def create_llm(model_override: str = None, **kwargs):
    provider = settings.LLM_PROVIDER.lower()
    model = model_override or settings.MODEL_NAME

    if (model and "gemini" in model) or provider == "gemini":
        sa_file = settings.GEMINI_SERVICE_ACCOUNT_FILE or "service_account.json"
        if sa_file and os.path.exists(sa_file):
            try:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    sa_file
                )
                logger.info(f"Using Gemini with service account from {sa_file}")
                return ChatGoogleGenerativeAI(
                    model=model if model and "gemini" in model else "gemini-flash-latest",
                    credentials=credentials,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to load service account credentials: {e}")
                # Fallback to API key if service account fails
        
        if settings.GEMINI_API_KEY:
            logger.info("Using Gemini with API key")
            return ChatGoogleGenerativeAI(
                model=model if model and "gemini" in model else "gemini-flash-latest",
                google_api_key=settings.GEMINI_API_KEY,
                **kwargs
            )
        
        raise ValueError("Gemini provider selected, but no API key or service account file provided.")

    if provider == "ollama":
        logger.info(f"Using Ollama model {model}")
        return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=model, **kwargs)
        
    # Add other providers like OpenAI here if needed
    
    raise ValueError(f"Unsupported LLM provider: {provider}")
