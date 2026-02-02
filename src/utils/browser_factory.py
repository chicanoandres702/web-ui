import logging
import os
from typing import Optional

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
    extra_browser_args = []

    # Add stealth arguments to avoid detection by CAPTCHAs
    extra_browser_args.append("--disable-blink-features=AutomationControlled")

    if use_own_browser:
        # If using own browser but path not explicitly set in config, try env var or default
        if not browser_binary_path:
            browser_binary_path = os.getenv("BROWSER_PATH", None)

        browser_user_data = config.get("browser_user_data_dir")
        if not browser_user_data:
            browser_user_data = os.getenv("BROWSER_USER_DATA", None)

        if browser_user_data:
            extra_browser_args.append(f"--user-data-dir={browser_user_data}")

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

    return CustomBrowser(config=browser_config)


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
