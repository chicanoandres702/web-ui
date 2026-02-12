import os
from dotenv import load_dotenv


def load_config():
    load_dotenv()

def get_config(key: str, default: str = None) -> str:
    return os.getenv(key, default)