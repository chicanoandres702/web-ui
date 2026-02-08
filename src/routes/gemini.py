from fastapi import APIRouter, Depends, HTTPException
from src.utils.gemini_utils import get_gemini_models
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/api/gemini_models")
async def get_gemini_models_route(models=Depends(get_gemini_models)):
    # Existing get_gemini_models implementation (mostly same as in server.py)
    return models