from browser_use.browser.browser import Browser
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from playwright.async_api import BrowserContext
from playwright.async_api import async_playwright
from src.browser.custom_context import CustomBrowserContext

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