import os
import secrets

# Allow OAuth over HTTP for localhost
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Generate a secure token
WEBSOCKET_TOKEN = os.getenv("WEBSOCKET_TOKEN", secrets.token_urlsafe(32))
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

RATE_LIMIT_SECONDS = 60  # seconds
MAX_REQUESTS_PER_MINUTE = 5
