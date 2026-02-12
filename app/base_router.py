from fastapi import APIRouter
import requests
from app.config import get_settings

settings = get_settings()
router = APIRouter()

@router.get("/health")
async def health(): return {"status": "ok", "version": settings.VERSION}

@router.get("/ollama/models")
async def get_models():
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.json()
    except: return {"models": []}
