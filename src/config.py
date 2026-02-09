import os

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

SECRET_KEY = os.environ.get("SECRET_KEY", "your_default_secret_key") # Replace with a strong secret key in production

RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", 60))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 20))