
import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Dict

logger = logging.getLogger(__name__)

COOKIE_PATH = "./tmp/cookies.json"

class CookieManager:
    """
    Manages the loading and saving of cookies to persist sessions across agent runs.
    """
    async def load_cookies(self, browser_context) -> None:

        if not self.config.cookie_path or not os.path.exists(self.config.cookie_path):
            logger.info("No cookie path specified or file does not exist.")
            return

        try:
            with open(self.config.cookie_path, 'r') as f:
                cookies = json.load(f)

            if not self.agent.config.browser_context:
                logger.warning("Browser context not available. Cannot load cookies.")
                return

            await self.agent.config.browser_context.add_cookies(cookies)
            logger.info(f"Loaded cookies from {self.config.cookie_path}")

        except Exception as e:
            logger.warning(f"Failed to load cookies: {e}")
    async def save_cookies(self, browser_context) -> None:
        """Saves cookies from the browser context to the specified file."""
        if not COOKIE_PATH or not browser_context:
            logger.info("No cookie path specified or browser context not available.")
            return

        try:
            cookies = await browser_context.cookies()
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(COOKIE_PATH), exist_ok=True)

            with open(COOKIE_PATH, 'w') as f:
                json.dump(cookies, f)
            logger.info(f"Saved cookies to {COOKIE_PATH}")

        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")

def get_cookie_manager():
    try:
        return CookieManager
    except ImportError:
        # Fallback to a dummy or local implementation if the specific utility is missing
        return None