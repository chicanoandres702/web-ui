import logging
import os
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.browser.custom_browser import CustomBrowser

logger = logging.getLogger(__name__)


def create_browser(config: dict) -> CustomBrowser:
    """
    Creates a CustomBrowser instance based on the provided configuration.
    """
    headless = config.get("headless", False)
    browser_binary_path = config.get("browser_binary_path")
    if browser_binary_path == "":
        browser_binary_path = None

    use_own_browser = config.get("use_own_browser", False)
    enable_persistent_session = config.get("enable_persistent_session", False)
    browser_user_data = config.get("browser_user_data_dir")

    extra_browser_args = []

    # Add stealth arguments to avoid detection by CAPTCHAs
    extra_browser_args.append("--disable-blink-features=AutomationControlled")

    if use_own_browser:
        # If using own browser but path not explicitly set in config, try env var or default
        if not browser_binary_path:
            browser_binary_path = os.getenv("BROWSER_PATH", None)

        if not browser_user_data:
            browser_user_data = os.getenv("BROWSER_USER_DATA", None)

    # Automatic persistence logic: Use default ./browser_session if enabled and no path provided
    if enable_persistent_session and not browser_user_data:
        browser_user_data = os.path.abspath("./browser_session")
        os.makedirs(browser_user_data, exist_ok=True)

    # NOTE: We do NOT add --user-data-dir to extra_browser_args because we use launch_persistent_context
    # which requires user_data_dir as a positional argument, not a flag.

    disable_security = config.get("disable_security", False)
    if disable_security:
        extra_browser_args.extend(
            [
                "--disable-web-security",
                "--disable-site-isolation-trials",
                "--disable-features=IsolateOrigins,site-per-process",
            ]
        )

    window_w = int(config.get("window_w", 1280))
    window_h = int(config.get("window_h", 1100))

    wss_url = config.get("wss_url")
    cdp_url = config.get("cdp_url")

    browser_config = BrowserConfig(
        headless=headless,
        browser_binary_path=browser_binary_path,
        extra_browser_args=extra_browser_args,
        wss_url=wss_url,
        cdp_url=cdp_url,
        new_context_config=BrowserContextConfig(
            window_width=window_w,
            window_height=window_h,
        ),
    )

    # Pass user_data_dir to CustomBrowser so it can use launch_persistent_context
    return CustomBrowser(config=browser_config, user_data_dir=browser_user_data)


async def create_context(browser: CustomBrowser, config: dict):
    """
    Creates a new browser context.
    """
    window_w = int(config.get("window_w", 1280))
    window_h = int(config.get("window_h", 1100))
    save_recording_path = config.get("save_recording_path")
    save_trace_path = config.get("save_trace_path")
    save_downloads_path = config.get("save_downloads_path", "./tmp/downloads")

    context_config = BrowserContextConfig(
        window_width=window_w,
        window_height=window_h,
        save_recording_path=save_recording_path,
        trace_path=save_trace_path,
        save_downloads_path=save_downloads_path,
    )

    return await browser.new_context(config=context_config)


class BrowserFactory:
    """
    Manages browser instances with persistent context support.
    """
    def __init__(self, user_data_dir: str = "./browser_session"):
        # Ensure the session directory exists
        self.user_data_dir = os.path.abspath(user_data_dir)
        if not os.path.exists(self.user_data_dir):
            os.makedirs(self.user_data_dir)
        self.playwright = None
        self.browser_context = None

    async def setup_persistent_browser(self, headless: bool = False, viewport: Dict[str, int] = None, extra_args: list = None):
        """
        Launches a browser with a persistent context to save cookies/logins.
        """
        if not self.playwright:
            self.playwright = await async_playwright().start()

        if viewport is None:
            viewport = {'width': 1280, 'height': 720}

        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox"
        ]
        if extra_args:
            args.extend(extra_args)

        # Using launch_persistent_context saves all cookies, local storage, etc.
        self.browser_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=headless,
            args=args,
            viewport=viewport
        )
        return self.browser_context

    async def close(self):
        if self.browser_context:
            await self.browser_context.close()
        if self.playwright:
            await self.playwright.stop()
