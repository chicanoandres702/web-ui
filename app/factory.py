import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.config import get_settings
from app import main, auth, agent_router
from app.knowledge_base import KnowledgeBase

from app.lifespan import lifespan_factory
from app import files, models
from fastapi import APIRouter
from .config import load_config


def initialize_knowledge_base() -> KnowledgeBase:
    """Initializes the KnowledgeBase with a default name."""
    return KnowledgeBase(name="default_knowledge")

def initialize_app_state(app: FastAPI):
    """Initializes application state."""
    app.state.knowledge_base = initialize_knowledge_base()

def create_middleware(app: FastAPI) -> None:
    """Creates and configures the middleware."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)

def register_routers(app: FastAPI) -> None:
    """Includes all the routers for the application."""
    app.include_router(main.router) # type: ignore
    app.include_router(auth.router)
    app.include_router(agent_router.router)
    # app.include_router(files.FilesRouter().get_router())
    # app.include_router(models.ModelsRouter().get_router())
    # app.include_router(create_knowledge_base_router())


def create_knowledge_base_router() -> APIRouter:
    """Creates the router for knowledge base operations."""
    from app.knowledge_base import create_knowledge_base_router as get_kb_router
    return get_kb_router()

def mount_static_files(app: FastAPI) -> None:
    """Mounts the static files directory."""    
    static_dir = os.getenv("STATIC_DIR", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


def configure_static_files(app: FastAPI) -> None:
    """Mounts the static files to the application."""
    STATIC_DIR = os.getenv("STATIC_DIR", "static")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def create_app(args):
    """
    Creates and configures the FastAPI application.
    """
    lifespan = lifespan_factory(args)
    app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    
    load_config()    
    create_middleware(app)
    initialize_app_state(app)    
    register_routers(app)
    mount_static_files(app)
    
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if not request.url.scheme == "https":
            response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; script-src 'self' 'unsafe-inline' 'unsafe-eval'"
            response.headers["Strict-Transport-Security"] = "max-age=0"  # Don't enable HSTS on HTTP
        return response

    return app
