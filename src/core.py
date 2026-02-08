import asyncio
import urllib.request
import json
import logging
import os
import importlib.util
import inspect
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/tags"
KB_DIR = "./tmp/knowledge_base"

async def check_ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Checks if Ollama is running at the specified URL."""
    try:
        def check_sync():
            try:
                with urllib.request.urlopen(ollama_url, timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False

        return await asyncio.to_thread(check_sync)
    except Exception as e:
        logger.warning(f"Error checking Ollama status: {e}")
        return False

def load_model_from_file(model_name: str):
    """Loads a Pydantic model from a Python file."""
    try:
        path = Path(f"./storage/{model_name}.py")
        if not path.exists():
            return None

        spec = importlib.util.spec_from_file_location(model_name, path)
        module = importlib.util.module_from_spec(spec)
        if not spec or not module:
            logger.error(f"Failed to load module from file: {path}")
            return None
        spec.loader.exec_module(module)

        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None