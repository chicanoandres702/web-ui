import argparse
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import Depends
from dotenv import load_dotenv
from fastapi import FastAPI

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SECRET_KEY
from src.routes import main, auth, files, models, websocket
from src.server_setup import startup_check_ollama
from starlette.middleware import Middleware
from src.knowledge_base_api import create_knowledge_base_router  # Import the function
from src.agent.browser_use.components.knowledge_base import KnowledgeBase  # Import KnowledgeBase
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


def initialize_knowledge_base():
    """Initializes the KnowledgeBase with a default name."""
    return KnowledgeBase(name="default_knowledge")

def create_app():
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI()
    configure_middleware(app)
    
    # Initialize KnowledgeBase as a dependency
    app.state.knowledge_base = initialize_knowledge_base()

    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


    app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="documentation")
    app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")
    app.include_router(main.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(websocket.router)
    app.include_router(create_knowledge_base_router())
    
    @app.post("/knowledge/export")
    async def export_knowledge(filepath: str, knowledge_base: KnowledgeBase = Depends(lambda: app.state.knowledge_base)):
        """Exports the knowledge base to a JSON file."""
        if knowledge_base.export_knowledge(filepath):
            return {"message": f"Knowledge base successfully exported to '{filepath}'."}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to export knowledge base to '{filepath}'.")

    @app.get("/knowledge")
    async def get_all_knowledge(knowledge_base: KnowledgeBase = Depends(lambda: app.state.knowledge_base)):
         """Retrieves all knowledge from the knowledge base."""
         return knowledge_base.storage



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
