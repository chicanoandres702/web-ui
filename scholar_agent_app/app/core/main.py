import os
import sys
import asyncio
import nest_asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.features.agent.api import router as agent_router
from app.features.researcher.api import router as researcher_router
from app.features.auth.api import router as auth_router
from app.core.config import get_settings

nest_asyncio.apply()
# CRITICAL WINDOWS SUBPROCESS FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

# Middleware must be added before routers
app.add_middleware(SessionMiddleware, secret_key=settings.API_SECRET_KEY, https_only=False, max_age=3600)

app.include_router(agent_router, prefix="/api/v1")
app.include_router(researcher_router, prefix="/api/v1/researcher")
app.include_router(auth_router, prefix="/auth")

static_p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features", "navigator", "ui")
if not os.path.exists(static_p): os.makedirs(static_p)
app.mount("/", StaticFiles(directory=static_p, html=True), name="static")
