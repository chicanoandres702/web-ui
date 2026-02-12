import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

class CookieHandler:
    """Handles cookie loading and saving for the agent."""

    def __init__(self, agent: Any):
        """Initializes the CookieHandler with the agent instance."""
        self.agent = agent

    async def load_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Loads cookies from a file into the browser context."""
        if cookie_path and os.path.exists(cookie_path):
            if browser_context:
                try:
                    with open(cookie_path, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                    await browser_context.add_cookies(cookies)
                    logger.info(f"Loaded cookies from {cookie_path}")
                except Exception as e:
                    logger.error(f"Failed to load cookies: {e}") # type: ignore

    async def save_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Saves cookies from the browser context to a file."""
        if browser_context and cookie_path:
            try:
                cookies = await browser_context.get_cookies()
                with open(cookie_path, 'w', encoding='utf-8') as f:
                    json.dump(cookies, f)
                logger.info(f"Saved cookies to {cookie_path}")
            except Exception as e:
                logger.error(f"Failed to save cookies: {e}") # type: ignore