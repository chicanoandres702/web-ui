import argparse
import logging
import os

import asyncio

import sys


import uvicorn

from fastapi import FastAPI, WebSocket, Request
from certificate_helper import create_certificate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from fastapi.staticfiles import StaticFiles


if BASE_DIR not in sys.path:

    sys.path.append(BASE_DIR)

from dotenv import load_dotenv

from app.factory import create_app


load_dotenv()

async def websocket_endpoint(websocket: WebSocket, request: Request):

    await websocket.accept()


    client_host = request.client.host if request.client else "unknown"

    logger.info(f"Client connected from: {client_host}")

    while True:

        data: str = await websocket.receive_text()

        logger.info(f"Received message: {data}")


logger = logging.getLogger(__name__)

def configure_static_files(app: FastAPI, static_dir: str = "static"):

    """Mounts the static files directory at the root path."""

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":

    app = create_app(args = argparse.Namespace(
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", 8000)),
            https=os.getenv("HTTPS", "False").lower() == "true"
        )


        )

    parser = argparse.ArgumentParser(description="Web UI Server")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")

    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    parser.add_argument("--https", action="store_true", help="Enable HTTPS with automatically created certificates.")
    args = parser.parse_args ()

    configure_static_files(app)
    app.websocket("/ws")(websocket_endpoint) # type: ignore

    # Configure logging level based on args if needed

    logging.basicConfig(level=logging.INFO)

    if args.https:

        cert_path = "./certs"

        key_path = os.path.join(cert_path, "private.key")

        cert_file = os.path.join(cert_path, "certificate.crt")
        

        # Use the utility function to create certificates

        if not os.path.exists(key_path) or not os.path.exists(cert_file):
            create_certificate(cert_path)



        from app.server import start_uvicorn_with_https

        start_uvicorn_with_https( # type: ignore
            app,
            host=args.host,
            port=args.port,
            cert_file=cert_file,

            key_file=key_path,
        )

    else:


        from app.server import start_uvicorn

        start_uvicorn(app, host=args.host, port=args.port, log_level="info")
