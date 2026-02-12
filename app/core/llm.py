from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOllama 
from app.config import get_settings

settings = get_settings()

class LLMFactory:
    """
    Factory class to instantiate LLM clients based on configuration.
    """
    
    @staticmethod
    def create_llm():
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                # Fallback for dev/testing if key missing
                print("WARNING: GEMINI_API_KEY missing. LLM calls will fail.")
                return None
            return ChatGoogleGenerativeAI(
                model=settings.MODEL_NAME,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0
            )
            
        elif provider == "openai":
            if not settings.OPENAI_API_KEY:
                print("WARNING: OPENAI_API_KEY missing.")
                return None
            return ChatOpenAI(
                model_name=settings.MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
                temperature=0
            )
            
        # elif provider == "ollama":
        #     return ChatOllama(
        #         base_url=settings.OLLAMA_BASE_URL,
        #         model=settings.MODEL_NAME
        #     )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

def get_llm():
    """Dependency injection helper."""
    return LLMFactory.create_llm()