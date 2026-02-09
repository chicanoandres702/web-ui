import logging

logger = logging.getLogger(__name__)

class APIAuthenticationError(Exception):
    """Custom exception for API authentication failures."""
    pass

def handle_api_error(e: Exception, message: str = "LLM API call failed"):
    """
    Centralized error handling for API calls.
    """
    logger.error(f"❌ {message}: {e}")
    # Re-raise as a custom exception for specific handling if needed
    raise APIAuthenticationError(message) from e

async def check_api_key(api_key: str, provider: str):
    """
    Validates the api key.
    """
    if not api_key:
        raise APIAuthenticationError(f"❌ Missing API Key for {provider}")

def create_error_message(error: str):
    """
    Creates a formatted error message.
    """
    return f"❌ {error}"