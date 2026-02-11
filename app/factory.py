import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from config.settings import settings
from src.routes import main, auth, files, models, websocket
from src.knowledge_base_api import create_knowledge_base_router
from src.agent.browser_use.components.knowledge_base import KnowledgeBase
from app.lifespan import lifespan_factory

STATIC_DIR = os.getenv("STATIC_DIR", "static")  # or ./public

def initialize_knowledge_base():
    """Initializes the KnowledgeBase with a default name."""
    return KnowledgeBase(name="default_knowledge")

def configure_middleware(app: FastAPI) -> None:
    """Configures the middleware for the FastAPI application."""

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

def create_app(args):
    """
    Creates and configures the FastAPI application.
    """
    lifespan = lifespan_factory(args)
    app: FastAPI = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    configure_middleware(app)
    # Initialize KnowledgeBase as a dependency
    app.state.knowledge_base = initialize_knowledge_base()

    app.include_router(main.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(websocket.router)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    app.include_router(create_knowledge_base_router())

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if not request.url.scheme == "https":
            response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' 'unsafe-eval'"
            response.headers["Strict-Transport-Security"] = "max-age=0"  # Don't enable HSTS on HTTP
        return response

    return app
