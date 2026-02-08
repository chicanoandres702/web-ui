import logging
import os
import time
from threading import Lock
from prometheus_client import Counter, Summary
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils import llm_provider

logger = logging.getLogger(__name__)

_llm_cache_lock = Lock()
_llm_cache: Dict[str, BaseChatModel] = {}

# Prometheus metrics
LLM_CACHE_HITS = Counter('llm_cache_hits', 'Number of LLM cache hits')
LLM_CACHE_MISSES = Counter('llm_cache_misses', 'Number of LLM cache misses')
LLM_INIT_TIME = Summary('llm_initialization_time', 'Time spent initializing LLM models')


_llm_cache_last_cleared: float = 0.0
LLM_CACHE_CLEAR_INTERVAL: float = 3600  # Clear cache every hour


def clear_llm_cache_if_needed():
    """
    Clears the LLM cache periodically based on time.
    """
    global _llm_cache_last_cleared, _llm_cache
    current_time = time.time()
    if current_time - _llm_cache_last_cleared > LLM_CACHE_CLEAR_INTERVAL:
        logger.info("Clearing LLM cache due to time interval.")
        _llm_cache = {}  # Clear the cache
        _llm_cache_last_cleared = current_time
        global _llm_cache_misses
        global _llm_cache_hits
        _llm_cache_hits = 0


def get_gemini_models() -> List[str]:
    """
    Returns a list of available Gemini models.
    """
    return ["gemini-flash-latest"]  # Add other Gemini models as needed


def get_llm_model(llm_settings: Dict[str, Any], websocket = None) -> Optional[BaseChatModel]:
    """
    Retrieves the language model based on the provider and settings.
    """
    provider = llm_settings.get("provider", "gemini")

    clear_llm_cache_if_needed()

    cache_key = f"{provider}-{llm_settings.get('model_name', 'gemini-flash-latest')}-{llm_settings.get('temperature', 0.0)}"
    
    if cache_key in _llm_cache:
        logger.info(f"Using cached LLM model for key: {cache_key}")
        return _llm_cache[cache_key]

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI


            # Check for OAuth credentials in session
            creds_data = websocket.session.get("google_creds") if websocket else None

            # Check for Service Account file
            service_account_path = "service_account.json"

            if os.path.exists(service_account_path):
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(service_account_path)
                if websocket:
                    websocket.send_json({"type": "log", "content": "Using Google Gemini with Service Account"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            elif creds_data:
                from google.oauth2.credentials import Credentials

                creds = Credentials(**creds_data)
                if websocket:
                    websocket.send_json({"type": "log", "content": "Using Google Gemini with OAuth Credentials"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            else:
                # Fallback to API Key
                if websocket:
                    websocket.send_json({"type": "log", "content": "Using Google Gemini with API Key"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    google_api_key=llm_settings.get("api_key", "")
                )
        except ImportError:
            if websocket:
                websocket.send_json({"type": "error", "content": "Please install langchain-google-genai to use Gemini."})
            return None
    else:
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=llm_settings.get("model_name", "gpt-4o"),
            temperature=float(llm_settings.get("temperature", 0.8)),
            base_url=llm_settings.get("base_url", ""),
            api_key=llm_settings.get("api_key", ""),
            project_id=llm_settings.get("google_project_id", "")
        )
    
    if llm:
        _llm_cache[cache_key] = llm
    
    return llm