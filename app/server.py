import os
import sys
import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.config import get_settings
from app.lifespan import lifespan
from app.auth import router as auth_router
from app.agent_router import router as agent_router
from app.base_router import router as base_router
from app.websocket import router as websocket_router

# Force Windows Loop FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

settings = get_settings()
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID)
os.environ.setdefault("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def create_app():
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)
    
    app.include_router(base_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/api/v1")
    app.include_router(auth_router)
    
    static_path = os.path.join(settings.BASE_DIR, "app", "static")
    os.makedirs(static_path, exist_ok=True)
    
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    return app
