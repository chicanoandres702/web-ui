from __future__ import annotations
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BrowserActionHandler:
    """Handles browser-specific actions.

    This class decouples browser interactions from the agent's core logic,
    providing a clear abstraction for performing actions in the browser.
    """

    def __init__(self, browser_context: Any, controller: Any, heuristics: Any):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics

    async def save_cookies(self, cookie_path: str) -> None:
        """Saves cookies from the browser context to a file."""
        try:
            cookies = await self.browser_context.get_cookies()
            # Serialize the cookies to a JSON string before writing to the file
            cookies_json = json.dumps(cookies, indent=2)  # Pretty print the JSON

            # Write the JSON string to the file
            with open(cookie_path, "w", encoding="utf-8") as f:
                f.write(cookies_json)
            logger.info(f"Saved cookies to {cookie_path}")
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")

    def detect_loop(self) -> bool:
        """Delegates loop detection to heuristics."""
        return self.heuristics.detect_loop()

    async def detect_progress(self) -> bool:
        """Delegates progress detection to heuristics."""
        return await self.heuristics.detect_progress()

    async def check_blocking_elements(self) -> None:
        """Delegates blocking element checks to heuristics."""
        await self.heuristics.check_blocking_elements()

    async def check_navigation_recovery(self) -> None:
        """Delegates navigation recovery to heuristics."""
        await self.heuristics.check_navigation_recovery()

    async def check_login_status(self) -> None:
        """Delegates login status checks to heuristics."""
        await self.heuristics.check_login_status()

    def manage_model_switching(self) -> None:
        """Delegates model switching logic to heuristics."""
        self.heuristics.manage_model_switching()

    def suggest_alternative_strategy(self) -> None:
        """Delegates strategy hints to heuristics."""
        self.heuristics.suggest_alternative_strategy()

    async def check_and_add_subtasks(self) -> None:
        """Delegates subtask analysis to heuristics."""
        await self.heuristics.check_and_add_subtasks()
