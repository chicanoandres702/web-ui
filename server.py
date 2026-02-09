import argparse
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi import FastAPI

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SECRET_KEY
from src.routes import main, auth, files, models, websocket
from src.server_setup import startup_check_ollama

from starlette.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_app():
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI()
    configure_middleware(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="documentation")
    app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")
    app.include_router(main.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(websocket.router)
    return app

app = create_app()



@app.on_event("startup")
async def startup_event():
    await startup_check_ollama()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser Agent Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting FastAPI server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
