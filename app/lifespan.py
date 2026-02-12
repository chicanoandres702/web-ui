import logging
import inspect
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles # type: ignore


logger = logging.getLogger(__name__)

def get_cookie_manager():
    try:
        from app.agent.browser_use.components.cookie_manager import CookieManager # type: ignore
        return CookieManager
    except ImportError:
        # Fallback to a dummy implementation if the specific utility is missing 
        return None

async def initialize_services(app: FastAPI, args):
    """Initializes external services and managers."""
    CookieManagerClass = get_cookie_manager()
    if CookieManagerClass:
        app.state.cookie_manager = CookieManagerClass()
        
        if hasattr(app.state.cookie_manager, 'load_cookies'):
            try:
                sig = inspect.signature(app.state.cookie_manager.load_cookies)
                params = sig.parameters
                if 'browser_context' in params and params['browser_context'].default is inspect.Parameter.empty:
                    logger.info("CookieManager.load_cookies requires an active browser_context; skipping global load.") # type: ignore
                else:
                    await app.state.cookie_manager.load_cookies(browser_context=None)
            except Exception as e:
                logger.error(f"Failed to initialize cookies: {e}")

    STATIC_DIR = app.extra.get("STATIC_DIR", "static")

    os.makedirs(STATIC_DIR, exist_ok=True)

    class SecureStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope: dict) -> Response:
            from app.environment_banner import get_environment_banner_html
            response = await super().get_response(path, scope)
            request = Request(scope)
            if request.url.scheme != "https" and response.media_type == "text/html":
                banner_html = get_environment_banner_html(args)
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



async def shutdown_event(app: FastAPI):
    """FastAPI shutdown handler."""
    logger.info("Application shutdown")
    if hasattr(app.state, 'cookie_manager') and app.state.cookie_manager:
        browser_context = getattr(app.state, 'browser_context', None)

        if browser_context:
            await app.state.cookie_manager.save_cookies(browser_context)
            logger.info("Cookies saved on shutdown.")
        else:
            logger.warning("Browser context not available on shutdown. Cookies not saved.")
def lifespan_factory(args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.extra["STATIC_DIR"] = "static"
        await initialize_services(app, args)
        yield
        await shutdown_event(app)
    return lifespan
