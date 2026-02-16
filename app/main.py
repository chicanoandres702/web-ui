import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router as api_router
from app.auth import router as auth_router
from app.config import get_settings

# CRITICAL WINDOWS SUBPROCESS FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

settings = get_settings()

# Setup Env for Auth
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
if settings.GOOGLE_CLIENT_ID:
    os.environ.setdefault("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID)
if settings.GOOGLE_CLIENT_SECRET:
    os.environ.setdefault("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET)

app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix="/api/v1")
app.include_router(auth_router)

static_p = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_p): os.makedirs(static_p)
app.mount("/", StaticFiles(directory=static_p, html=True), name="static")