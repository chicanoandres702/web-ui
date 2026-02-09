
import argparse
import logging
import os
import asyncio
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import sys
from dotenv import load_dotenv

# Ensure src and src/browser are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Internal Components
def get_cookie_manager():
    try:
        from src.browser.cookie_manager import CookieManager
        return CookieManager
    except ImportError:
        # Fallback to a dummy or local implementation if the specific utility is missing
        return None

from src.config import SECRET_KEY
from src.routes import main, auth, files, models, websocket
from src.knowledge_base_api import create_knowledge_base_router
from src.server_setup import startup_check_ollama
from src.agent.browser_use.components.knowledge_base import KnowledgeBase
from starlette.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed


STATIC_DIR = os.getenv("STATIC_DIR", "static")  # or ./public
DOCS_DIR = os.getenv("DOCS_DIR", "docs")

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
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


async def run_in_threadpool(func, *args, **kwargs):
    """Asynchronously run a synchronous function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

def initialize_knowledge_base():
    """Initializes the KnowledgeBase with a default name."""
    return KnowledgeBase(name="default_knowledge")

def create_app():
    """
    Creates and configures the FastAPI application.
    """
    app: FastAPI = FastAPI()
    configure_middleware(app)

    # Initialize KnowledgeBase as a dependency
    # app.state.knowledge_base = initialize_knowledge_base()

    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


    app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="documentation")
    app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")
    app.include_router(main.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(websocket.router)
    app.include_router(create_knowledge_base_router())
    return app


app: FastAPI = create_app()

async def initialize_services(app: FastAPI):
    """Initializes external services and managers."""
    CookieManagerClass = get_cookie_manager()
    if CookieManagerClass:
        app.state.cookie_manager = CookieManagerClass(None)  # Pass None for agent initially
        await app.state.cookie_manager.load_cookies()
    await startup_check_ollama()

@app.on_event("startup")
async def startup_event():
    """FastAPI startup handler."""
    await initialize_services(app)

@app.on_event("shutdown")
async def shutdown_event():
    # Get the current browser context
    # from src.agent.browser_use.browser_use_agent import BrowserUseAgent
    # browser_use_agent = BrowserUseAgent()
    # browser_context = browser_use_agent.config.browser_context

    #This is commented out because I'm unsure how to get the browser context here.  Please help me with the next prompt

    # Save cookies from the browser context
    # if browser_context:
    #     cookies = await browser_context.cookies()
    #     await app.state.cookie_manager.save_cookies(cookies)
    #     logger.info("Cookies saved on shutdown.")
    # else:
    #     logger.warning("Browser context not available on shutdown. Cookies not saved.")
    logger.info("Application shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web UI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting FastAPI server at http://{args.host}:{args.port}")

    # Start Uvicorn server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
