from browser_use.browser.browser import Browser
import asyncio
import logging
from typing import Any, Dict, Optional
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from playwright.async_api import BrowserContext
from playwright.async_api import async_playwright
from src.browser.custom_context import CustomBrowserContext
from src.agent.browser_use.components.cookie_manager import CookieManager

logger = logging.getLogger(__name__)

class CustomBrowser(Browser):
    def __init__(self, config: BrowserConfig, user_data_dir: str = None):
        super().__init__(config=config)
        self.user_data_dir = user_data_dir

    async def new_context(self, config: BrowserContextConfig = None) -> BrowserContext:
        """
        Override new_context to support persistent context if user_data_dir is set.
        """
        if self.user_data_dir:
            if not self.playwright:
                self.playwright = await async_playwright().start()

            args = [
                "--disable-blink-features=AutomationControlled",
            ]
            if self.config.extra_browser_args:
                args.extend(self.config.extra_browser_args)
            
            # Map BrowserContextConfig to launch_persistent_context args
            viewport = None
            if config:
                viewport = {"width": config.window_width, "height": config.window_height}

            context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                headless=self.config.headless,
                args=args,
                viewport=viewport,
            )
            
            # Wrap the playwright context in our CustomBrowserContext
            browser_context = CustomBrowserContext(browser=self, config=config)
            browser_context.context = context
            return browser_context
            
        return await super().new_context(config)
    
class AgentCleanupHandler:
    """
    Handles the cleanup process for the agent, ensuring resources are released
    and final states (like cookies) are persisted.
    """
    def __init__(self, agent: Any):
        self.agent = agent

    async def handle_cleanup(self) -> None:
        """
        Performs cleanup operations: saving cookies, closing planner, 
        and executing done callbacks.
        """
        logger.info("Starting agent cleanup...")
        
        try:
            # 1. Save Cookies via CookieHandler/Manager
            if hasattr(self.agent, 'cookie_handler') and self.agent.browser_context:
                cookie_path = getattr(self.agent, 'cookie_path', "./tmp/cookies.json")
                await self.agent.cookie_handler.save_cookies(self.agent.browser_context, cookie_path)
            
            # 2. Stop Planner Task
            if hasattr(self.agent, 'planner_task') and self.agent.planner_task:
                self.agent.planner_task.cancel()
                try:
                    await self.agent.planner_task
                except asyncio.CancelledError:
                    pass

            # 3. Execute Done Callbacks
            if hasattr(self.agent, '_execute_done_callback'):
                await self.agent._execute_done_callback()

            # 4. Close Browser Factory resources
            if hasattr(self.agent, 'browser_factory'):
                await self.agent.browser_factory.close_browser()

        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")
        finally:
            logger.info("Cleanup completed.")


class CookieHandler:
    """
    Refactored CookieHandler to modularize cookie operations within the agent.
    """
    def __init__(self, agent: Any):
        self.agent = agent
        self.manager = CookieManager()

    async def load_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Loads cookies into the browser context using the CookieManager."""
        self.manager.cookie_path = cookie_path
        # browser-use BrowserContext stores the Playwright context in
        # .context attribute
        playwright_context = getattr(browser_context, 'context', browser_context)
        await self.manager.load_cookies(playwright_context)

    async def save_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Saves cookies from the browser context using the CookieManager."""
        self.manager.cookie_path = cookie_path
        playwright_context = getattr(browser_context, 'context', browser_context)
        await self.manager.save_cookies(playwright_context)