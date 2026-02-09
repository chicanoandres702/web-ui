
import asyncio
import logging
import os
from fastapi import FastAPI
import urllib.request
import json
from src.utils.utils import ensure_default_extraction_models, suppress_asyncio_cleanup_errors

logger = logging.getLogger(__name__)


async def startup_check_ollama():
    logger.info("🚀 Server starting up...")
    suppress_asyncio_cleanup_errors()
    os.makedirs("./tmp", exist_ok=True)
    
    logger.info("📦 Ensuring default extraction models are present...")
    await asyncio.to_thread(ensure_default_extraction_models)
    logger.info("✅ Extraction models ready.")

    logger.info("Checking for Ollama instance...")
    try:
        url = "http://localhost:11434/api/tags"
        def check():
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False
        
        is_running = await asyncio.to_thread(check)
        if is_running:
            logger.info("✅ Ollama is detected and running at http://localhost:11434")
        else:
            logger.warning("⚠️ Ollama is NOT detected at http://localhost:11434. (Ensure it's running for local LLMs)")
    except Exception as e:
        logger.warning(f"⚠️ Error checking Ollama status: {e}")

    # Check Google OAuth configuration
    if os.path.exists("client_secret.json"):
        logger.info("✅ Google OAuth credentials detected (client_secret.json).")
    elif not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        logger.warning("⚠️ Google OAuth credentials (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET) not found. 'Sign in with Google' will fail.")
    else:
        logger.info("✅ Google OAuth credentials detected.")

def mount_static_files(app: FastAPI, static_dir: str, docs_dir: str):
    """Mounts static file directories for the FastAPI application."""
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    app.mount("/docs", StaticFiles(directory=docs_dir), name="documentation")
    app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")
