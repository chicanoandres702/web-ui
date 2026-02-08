import importlib.util
import inspect
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter
from src.utils.llm_manager import get_gemini_models
import urllib.request

router = APIRouter()
logger = logging.getLogger(__name__)

def load_model_from_file(model_name: str):
    try:
        path = Path(f"./storage/{model_name}.py")
        if not path.exists():
            return None
        
        spec = importlib.util.spec_from_file_location(model_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                if obj.__module__ == model_name:
                    return obj
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None

@router.get("/extraction_models")
async def get_extraction_models():
    models_dir = Path("./storage")
    if not models_dir.exists():
        return []
    files = [f.stem for f in models_dir.glob("*.py") if f.stem != "__init__"]
    return files

@router.get("/ollama_models")
async def get_ollama_models():
    try:
        url = "http://localhost:11434/api/tags"
        def fetch():
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode())
        
        data = await asyncio.to_thread(fetch)
        return [model['name'] for model in data.get('models', [])]
    except Exception:
        return []

@router.get("/api/gemini_models")
async def gemini_models_endpoint():
    """
    Endpoint to return available Gemini models.
    """
    try:
        models = get_gemini_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching Gemini models: {e}")
        return {"error": "Could not retrieve Gemini models"}, 500
