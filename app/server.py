import uvicorn
import asyncio
import logging

logger = logging.getLogger(__name__)

def start_uvicorn(app, host, port, log_level="info"):
    """
    Starts the Uvicorn server with the given configuration.
    """
    try:
        print(f"Starting FastAPI server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level=log_level) # type: ignore
    except Exception as e:
        logger.error(f"Uvicorn server failed to start: {e}")
        raise

def start_uvicorn_with_https(app, host, port, cert_file, key_file, log_level="info"):
     """
     Starts the Uvicorn server with HTTPS configuration.
     """
     try:
        print(f"Starting FastAPI server at https://{host}:{port}")
        config: uvicorn.Config = uvicorn.Config(app, host=host, port=port, log_level=log_level, ssl_keyfile=key_file, ssl_certfile=cert_file)
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
     except Exception as e:
         logger.error(f"Uvicorn HTTPS server failed to start: {e}")
         raise

if __name__ == "__main__":
    # This is just for testing
    # You'd replace this with your actual app import
    # For example:  from main import app
    pass