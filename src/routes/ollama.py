import os
import urllib.request
import json
from fastapi import APIRouter
import asyncio
from src.core import DEFAULT_OLLAMA_URL
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/ollama_models")
async def get_ollama_models():
    try:
        # Existing get_ollama_models implementation (same as in server.py)
        return []
    except Exception as e:
        logger.warning(f"Error fetching Ollama models: {e}")
        return []