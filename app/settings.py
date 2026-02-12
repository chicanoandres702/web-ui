import os

class Settings:
   secret_key: str = os.getenv("SECRET_KEY", "your-secret-key")
   google_client_id: str = os.getenv("GOOGLE_CLIENT_ID")