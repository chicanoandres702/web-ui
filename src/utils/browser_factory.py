import logging
import os
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from src.browser.custom_browser import CustomBrowser
from src.utils import config as app_config

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
    browser_user_data = config.get("browser_user_data_dir")
    enable_persistent_session = config.get("enable_persistent_session", True)

    extra_browser_args = []

    # Add stealth arguments to avoid detection by CAPTCHAs
    extra_browser_args.append("--disable-blink-features=AutomationControlled")

    # Add user-defined extra args (from config or env)
    user_extra_args = config.get("extra_browser_args") or os.getenv("EXTRA_BROWSER_ARGS", "")
    if user_extra_args:
        extra_browser_args.extend(user_extra_args.split())

    # Add Chrome Profile argument if specified
    profile_name = config.get("chrome_profile_name") or os.getenv("CHROME_PROFILE_NAME", "")
    if profile_name:
        extra_browser_args.append(f"--profile-directory={profile_name}")

    if use_own_browser:
        # If using own browser but path not explicitly set in config, try env var or default
        if not browser_binary_path:
            browser_binary_path = os.getenv("BROWSER_PATH", None)

        if not browser_user_data:
            browser_user_data = os.getenv("BROWSER_USER_DATA", None)

    # Handle persistence logic
    if enable_persistent_session:
        if not browser_user_data or not browser_user_data.strip():
            browser_user_data = os.path.abspath(app_config.DEFAULT_BROWSER_SESSION_DIR)
        
        # Append profile name to path to allow parallel execution with different profiles
        if profile_name:
            safe_profile = "".join(c for c in profile_name if c.isalnum() or c in (' ', '_', '-')).strip()
            if safe_profile:
                browser_user_data = os.path.join(browser_user_data, safe_profile)
        os.makedirs(browser_user_data, exist_ok=True)
    else:
        browser_user_data = None

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
    Handles persistent sessions manually to ensure only one browser instance is launched.
    """
    window_w = int(config.get("window_w", 1280))
    window_h = int(config.get("window_h", 1100))
    save_recording_path = config.get("save_recording_path")
    save_trace_path = config.get("save_trace_path")
    save_downloads_path = config.get("save_downloads_path", "./tmp/downloads")

    # Check if we should launch a persistent context (Local only)
    # We bypass browser.new_context() here to prevent double-launching (one persistent, one ephemeral)
    if hasattr(browser, 'user_data_dir') and browser.user_data_dir and not browser.config.wss_url and not browser.config.cdp_url:
        logger.info(f"Launching persistent browser context from: {browser.user_data_dir}")
        
        if not browser.playwright:
            browser.playwright = await async_playwright().start()

        browser_config = browser.config
        
        # Prepare arguments for launch_persistent_context
        args = {
            "user_data_dir": browser.user_data_dir,
            "headless": browser_config.headless,
            "args": browser_config.extra_browser_args,
            "viewport": {"width": window_w, "height": window_h},
            "device_scale_factor": 1,
            "downloads_path": save_downloads_path,
        }

        if browser_config.browser_binary_path:
            args["executable_path"] = browser_config.browser_binary_path

        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)
            args["record_video_dir"] = save_recording_path
            args["record_video_size"] = {"width": window_w, "height": window_h}
            
        if hasattr(browser_config, 'proxy') and browser_config.proxy:
            args["proxy"] = browser_config.proxy

        # Launch the persistent context directly
        persistent_context = await browser.playwright.chromium.launch_persistent_context(**args)
        
        # Enable tracing if requested
        if save_trace_path:
            os.makedirs(save_trace_path, exist_ok=True)
            await persistent_context.tracing.start(screenshots=True, snapshots=True, sources=True)
            
        # Wrap in BrowserContext to satisfy Agent interface
        context_config = BrowserContextConfig(
            window_width=window_w,
            window_height=window_h,
            save_recording_path=save_recording_path,
            trace_path=save_trace_path,
            save_downloads_path=save_downloads_path,
        )
        
        browser_context = BrowserContext(browser=browser, config=context_config)
        browser_context.context = persistent_context

        # Polyfill for missing methods if BrowserContext wrapper is incomplete or version mismatch
        if not hasattr(browser_context, 'get_current_page'):
            async def _get_current_page():
                if persistent_context.pages: return persistent_context.pages[-1]
                return await persistent_context.new_page()
            browser_context.get_current_page = _get_current_page
        
        return browser_context

    # Fallback to standard ephemeral context (or remote connection)
    context_config = BrowserContextConfig(
        window_width=window_w,
        window_height=window_h,
        save_recording_path=save_recording_path,
        trace_path=save_trace_path,
        save_downloads_path=save_downloads_path,
    )

    return await browser.new_context(config=context_config)
