import argparse
import logging
import os
import asyncio
import sys
import inspect

from fastapi import Depends
import uvicorn
from fastapi import FastAPI, HTTPException, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Internal Components
def get_cookie_manager():
    try:
        from src.agent.browser_use.components.cookie_manager import CookieManager # type: ignore
        return CookieManager # type: ignore
    except ImportError:
        # Fallback to a dummy implementation if the specific utility is missing
        return None

from fastapi import Request
from src.config import SECRET_KEY
from src.routes import main, auth, files, models, websocket
from src.knowledge_base_api import create_knowledge_base_router
from src.agent.browser_use.components.knowledge_base import KnowledgeBase
from starlette.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import HTMLResponse

from src.server_setup import startup_check_ollama
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

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)



async def run_in_threadpool(func, *args, **kwargs):
    """Asynchronously run a synchronous function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)


def initialize_knowledge_base():
    """Initializes the KnowledgeBase with a default name."""
    return KnowledgeBase(name="default_knowledge")


def get_environment_banner_html():
    """
    Returns HTML for a banner indicating the environment (HTTP/HTTPS).
    """
    banner_style = """
    <style>
        .env-banner {
            background-color: #FFA500; /* Orange */
            color: white;
            text-align: center;
            padding: 5px;
            font-weight: bold;
        }
    </style>
    """
    if args and args.https:
        return ""


    banner_html = f"""
    {banner_style}
    <div class="env-banner">WARNING: Running in HTTP mode. HTTPS is highly recommended for security.</div>
    """
    return banner_html

def create_app():
    """
    Creates and configures the FastAPI application.
    """
    app: FastAPI = FastAPI(docs_url=None, redoc_url=None)
    configure_middleware(app)
    # Initialize KnowledgeBase as a dependency
    app.state.knowledge_base = initialize_knowledge_base()

    app.include_router(main.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(models.router)
    app.include_router(websocket.router)
    app.include_router(auth.router)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    app.include_router(create_knowledge_base_router())

    #app.include_router(router)
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if not request.url.scheme == "https":
            response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' 'unsafe-eval'"
            response.headers["Strict-Transport-Security"] = "max-age=0"  # Don't enable HSTS on HTTP
        return response

    return app


app: FastAPI = create_app()

def configure_middleware(app: FastAPI) -> None:
    """Configures the middleware for the FastAPI application."""
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],) # Allows all headers
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

async def initialize_services(app: FastAPI):
    """Initializes external services and managers."""
    CookieManagerClass = get_cookie_manager()
    if CookieManagerClass:
        app.state.cookie_manager = CookieManagerClass()  # Fixed: Expected 0 positional arguments
        
        # Check if load_cookies exists and determine if it needs a context
        if hasattr(app.state.cookie_manager, 'load_cookies'):
            try:
                sig = inspect.signature(app.state.cookie_manager.load_cookies)
                # Check if browser_context is a required parameter (no default value)
                params = sig.parameters
                if 'browser_context' in params and params['browser_context'].default is inspect.Parameter.empty:
                    # If required but we are at global startup (no browser yet), we skip or pass None
                    logger.info("CookieManager.load_cookies requires an active browser_context; skipping global load.")
                else:
                    await app.state.cookie_manager.load_cookies(browser_context=None)
                    # Call with no args if signature allows, or if browser_context is optional
            except Exception as e:

                logger.error(f"Failed to initialize cookies: {e}")
    #Override the static files to inject a warning banner
    original_static_files = app.router.routes[5] #Hardcoded here
    class SecureStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope: dict) -> Response:
            response = await super().get_response(path, scope)
            request = Request(scope)
            if request.url.scheme != "https" and response.media_type == "text/html":
                banner_html = get_environment_banner_html()
                body = response.body
                if isinstance(body, memoryview):
                    response_content = body.tobytes().decode("utf-8")
                else:
                    response_content = body.decode("utf-8")
                response.body = (banner_html + response_content).encode("utf-8")
                response.headers["Content-Length"] = str(len(response.body))
            return response

    if not args.https:
        app.mount("/", SecureStaticFiles(directory=STATIC_DIR, html=True), name="static")

    await startup_check_ollama()

@app.on_event("startup")
async def startup_event():
    """FastAPI startup handler."""
    await initialize_services(app)

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown handler."""
    logger.info("Application shutdown")
    #Get the current browser context
    if hasattr(app.state, 'cookie_manager') and  app.state.cookie_manager:
        # save_cookies requires a browser_context. Since this is global shutdown, 
        # we check if a context was registered or skip if none active.
            browser_context = getattr(app.state, 'browser_context', None)
            if browser_context:
                await app.state.cookie_manager.save_cookies(browser_context)

                logger.info("Cookies saved on shutdown.")
            else:
                logger.warning("Browser context not available on shutdown. Cookies not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web UI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS with automatically created certificates.")
    args = parser.parse_args()

    if args.https:
        try:
            import certipy
        except ImportError:
            logger.error("HTTPS enabled but 'certipy' is not installed. Please install it or run without --https.")
            sys.exit(1)
        # https://github.com/jdavis/certipy
        certificate = certipy.CertificateAuthority(
            path="./certs",
            common_name="localhost",
        )

        key_path = "private.key"
        cert_path = "certificate.crt"

        if not os.path.exists(os.path.join("./certs", key_path)) or not os.path.exists(os.path.join("./certs", cert_path)):
            certificate.generate_certificate(
                hosts=["localhost"],
            )

        # Check if certificate files exist
        cert_file = os.path.join("./certs", cert_path)
        key_file = os.path.join("./certs", key_path)
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            sys.exit(1)
        
        print(f"Starting FastAPI server at https://{args.host}:{args.port}")

        # Start Uvicorn server with HTTPS configuration
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=key_file,
            ssl_certfile=cert_file
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    else:
        print(f"Starting FastAPI server at http://{args.host}:{args.port}")

        # Start Uvicorn server
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    print(f"Starting FastAPI server at http://{args.host}:{args.port}")

    # Start Uvicorn server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
