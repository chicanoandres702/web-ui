
import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import List, Dict

logger = logging.getLogger(__name__)


class CookieManager:
    """
    Manages the loading and saving of cookies to persist sessions across agent runs.
    """

    def __init__(self, cookie_path: str = DEFAULT_COOKIE_PATH):
        self.cookie_path = cookie_path

    async def load_cookies(self, browser_context: Any) -> None:
        """Loads cookies from a file into the browser context"""
        DEFAULT_COOKIE_PATH = "./tmp/cookies.json"
        if not self.cookie_path or not os.path.exists(self.cookie_path):
            logger.info("No cookie path specified or file does not exist.")
            return

        try:            
            with open(self.cookie_path, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            await browser_context.add_cookies(cookies)
            logger.info(f"Loaded cookies from {self.cookie_path}")
        except Exception as e:
            logger.warning(f'Failed to load cookies: {e}')

    async def save_cookies(self, browser_context: Any) -> None:
        """Saves cookies from the browser context to the specified file."""
        if not self.cookie_path or  browser_context is None:
            logger.info("No cookie path specified or browser context not available.")
            return
        try:
            cookies = await browser_context.cookies()            
            with open(self.cookie_path, 'w', encoding='utf-8') as f:
                json.dump(cookies, f)
        except Exception as e:

            logger.warning(f"Failed to save cookies: {e}")

            
def get_cookie_manager():
    try:
        return CookieManager
    except ImportError:
        # Fallback to a dummy or local implementation if the specific utility is missing
        return None