import os
import logging
import google.auth
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("ScholarBackend")

def create_llm(model_override: str = None):
    provider = settings.LLM_PROVIDER.lower()
    model = model_override or settings.MODEL_NAME
    if (model and model.startswith("gemini")) or provider == "gemini":
        gemini_model = model if (model and model.startswith("gemini")) else "gemini-flash-latest"
        
        sa_file = "service_account.json"
        if os.path.exists(sa_file):
            try:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_file
                credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
                
                logger.info("Using service account for Gemini.")
                return ChatGoogleGenerativeAI(
                    model=gemini_model,
                    credentials=credentials,
                    temperature=0.1
                )
            except Exception as e:
                logger.error(f"Failed to use service account: {e}. Falling back.")

        # Fallback to API Key
        logger.info("Using API key for Gemini.")
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=model)
