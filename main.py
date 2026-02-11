import argparse
import logging
import os
import asyncio
import sys
import uvicorn
from app.factory import create_app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web UI Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS with automatically created certificates.")
    args = parser.parse_args()

    app = create_app(args)

    if args.https:
        try:
            import certipy
        except ImportError:
            logger.error("HTTPS enabled but 'certipy' is not installed. Please install it or run without --https.")
            sys.exit(1)
        
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

        cert_file = os.path.join("./certs", cert_path)
        key_file = os.path.join("./certs", key_path)
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            sys.exit(1)
        
        print(f"Starting FastAPI server at https://{args.host}:{args.port}")

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
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
