import logging
import json
from browser_use.browser.context import BrowserContext
from src.utils.browser_scripts import (
    COMMON_CLOSE_SELECTORS,
    JS_REMOVE_OVERLAYS,
    JS_SCROLL_SLOW,
    JS_REMOVE_ADS,
    JS_GET_SCROLL_INFO,
    JS_HANDLE_VIGNETTE,
    JS_CLOSE_NEWSLETTER,
    JS_HANDLE_RATE_EXPERIENCE,
    JS_HANDLE_NOTIFICATIONS_PROMPT,
    JS_HANDLE_INSTALL_APP,
    JS_HANDLE_AGE_GATE,
    JS_SCROLL_TO_TEXT,
    JS_SCROLL_TO_SELECTOR
)
from src.utils.utils import retry_async
from src.agent.browser_use.dom_purification import purify_page
from src.agent.browser_use.smart_focus import wait_for_text
from src.controller.helpers import attempt_close_cookie_banner

logger = logging.getLogger(__name__)

class NavigationActionsMixin:
    def _register_navigation_actions(self):
        @self.registry.action("Evaluate a URL for safety before navigating")
        async def check_url_safety(url: str):
            status = self.nav_controller.evaluate_url(url)
            if status == "SAFE":
                return f"URL '{url}' is considered SAFE."
            elif status == "SUSPICIOUS":
                return f"URL '{url}' is flagged as SUSPICIOUS (potential distraction). Proceed with caution."
            return f"URL '{url}' is EXTERNAL. Verify relevance to task."

        @self.registry.action("Clear all blocking elements (popups, ads, cookie banners) to restore view")
        async def clear_view(browser: BrowserContext):
            page = await browser.get_current_page()
            results = []
            
            # 0. Google Vignette (High priority)
            try:
                if await page.evaluate(JS_HANDLE_VIGNETTE):
                    results.append("Dismissed Google Vignette")
                    await page.wait_for_timeout(500)
            except Exception: pass

            # 1. Cookie Banners
            try:
                if await attempt_close_cookie_banner(page):
                    results.append("Closed cookie banner")
                    await page.wait_for_timeout(500)
            except Exception: pass
            
            # 1.5 Notification Prompts
            try:
                if await page.evaluate(JS_HANDLE_NOTIFICATIONS_PROMPT):
                    results.append("Dismissed notification prompt")
                    await page.wait_for_timeout(500)
            except Exception: pass
            
            # 1.6 App Banners
            try:
                if await page.evaluate(JS_HANDLE_INSTALL_APP):
                    results.append("Dismissed 'Install App' banner")
                    await page.wait_for_timeout(500)
            except Exception: pass
            
            # 1.7 Age Gates
            try:
                if await page.evaluate(JS_HANDLE_AGE_GATE):
                    results.append("Dismissed age gate")
                    await page.wait_for_timeout(500)
            except Exception: pass

            # 2. Common Close Buttons (Popups/Ads)
            try:
                clicked_close = False
                for selector in COMMON_CLOSE_SELECTORS:
                    try:
                        loc = page.locator(selector).first
                        if await loc.is_visible(timeout=200):
                            await loc.click(timeout=500)
                            clicked_close = True
                            await page.wait_for_timeout(500)
                    except Exception: continue
                if clicked_close:
                    results.append("Clicked close button(s)")
            except Exception: pass
            
            # 3. Purify Page (Advanced Cleaning)
            try:
                purify_result = await purify_page(page)
                if purify_result.get("removed", 0) > 0:
                    results.append(f"Purified {purify_result['removed']} elements")
                if purify_result.get("success"):
                    results.append("Anchored content")
            except Exception: pass

            # 4. Ads (Legacy check)
            try:
                ads_removed = await page.evaluate(JS_REMOVE_ADS)
                if ads_removed > 0:
                    results.append(f"Removed {ads_removed} ads")
            except Exception: pass

            # 5. Overlays/Popups (Legacy check)
            try:
                # Try Escape first
                await page.keyboard.press("Escape")
                overlays_removed = await page.evaluate(JS_REMOVE_OVERLAYS)
                if overlays_removed > 0:
                    results.append(f"Removed {overlays_removed} overlays")
            except Exception: pass
            
            if not results:
                return "No blocking elements found or removed."
            return " | ".join(results) + "."

        @self.registry.action("Close difficult popups, modals, or overlays that block the screen")
        async def close_difficult_popup(browser: BrowserContext):
            page = await browser.get_current_page()
            
            # Strategy 1: Press Escape
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass

            # Strategy 2: Click common close buttons
            clicked = False
            for selector in COMMON_CLOSE_SELECTORS:
                try:
                    # Short timeout to check visibility
                    loc = page.locator(selector).first
                    if await loc.is_visible(timeout=200):
                        await loc.click(timeout=500)
                        clicked = True
                        logger.info(f"Clicked popup close button: {selector}")
                        await page.wait_for_timeout(500)
                        break
                except Exception:
                    continue
            
            if clicked:
                return "Clicked a close button."

            # Strategy 3: Remove aggressive overlays via JS
            try:
                removed = await page.evaluate(JS_REMOVE_OVERLAYS)
                if removed > 0:
                    return f"Removed {removed} full-screen overlay(s) via JavaScript."
            except Exception as e:
                logger.warning(f"Failed to remove overlays: {e}")

            return "Attempted to close popup using Escape key and common selectors."

        @self.registry.action("Remove ads and tracking elements from the page")
        async def remove_ads(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                count = await page.evaluate(JS_REMOVE_ADS)
                return f"Removed {count} ad/tracking elements."
            except Exception as e:
                return f"Error removing ads: {e}"

        @self.registry.action("Attempt to close cookie consent banners")
        async def close_cookie_banner(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await attempt_close_cookie_banner(page):
                    await page.wait_for_timeout(1000)  # Wait for animation/removal
                    return "Clicked a cookie consent button."
                return "No common cookie banner buttons found."
            except Exception as e:
                return f"Error closing cookie banner: {e}"

        @self.registry.action("Dismiss Google Vignette ads and restore scrolling")
        async def dismiss_google_vignette(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_HANDLE_VIGNETTE):
                    return "Dismissed Google Vignette and restored scrolling."
                return "No Google Vignette found."
            except Exception as e:
                return f"Error dismissing vignette: {e}"

        @self.registry.action("Attempt to close newsletter/subscription modals")
        async def close_newsletter_modal(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_CLOSE_NEWSLETTER):
                    return "Closed a newsletter/subscription modal."
                return "No obvious newsletter modal found."
            except Exception as e:
                return f"Error closing newsletter: {e}"

        @self.registry.action("Dismiss 'Rate this experience' or feedback popups")
        async def dismiss_rating_popup(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_HANDLE_RATE_EXPERIENCE):
                    return "Dismissed a rating/feedback popup."
                return "No rating popup found."
            except Exception as e:
                return f"Error dismissing rating popup: {e}"

        @self.registry.action("Dismiss 'Allow Notifications' HTML prompts")
        async def dismiss_notification_prompt(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_HANDLE_NOTIFICATIONS_PROMPT):
                    return "Dismissed a notification prompt."
                return "No notification prompt found."
            except Exception as e:
                return f"Error dismissing notification prompt: {e}"

        @self.registry.action("Dismiss 'Install App' or 'Open in App' banners")
        async def dismiss_app_banner(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_HANDLE_INSTALL_APP):
                    return "Dismissed an 'Install App' banner."
                return "No app banner found."
            except Exception as e:
                return f"Error dismissing app banner: {e}"

        @self.registry.action("Dismiss age verification gates")
        async def dismiss_age_gate(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                if await page.evaluate(JS_HANDLE_AGE_GATE):
                    return "Clicked an age verification button."
                return "No age gate found."
            except Exception as e:
                return f"Error dismissing age gate: {e}"

        @self.registry.action("Scroll down the page. Options: 'page' (default), 'half', 'quarter', or 'full' (scans whole page & resets).")
        async def scroll_down(browser: BrowserContext, amount: str = "page"):
            page = await browser.get_current_page()
            
            if amount == "full":
                async def _do_full_scroll():
                    # Scroll to bottom slowly to trigger lazy loading
                    await page.evaluate(JS_SCROLL_SLOW)
                    # Wait a moment for final loads
                    await page.wait_for_timeout(500)
                    # Scroll back to top
                    await page.evaluate("window.scrollTo(0, 0)")
                    # Extract text content to ensure the agent "reads" it
                    return await page.evaluate("document.body.innerText")

                try:
                    content = await retry_async(_do_full_scroll, logger=logger, error_message="Full scroll failed")
                    preview = content[:2000]
                    if len(content) > 2000:
                        preview += "\n...[content truncated]..."
                    return f"Scanned entire page (scrolled down and back up). You are now at the top. Content preview:\n{preview}"
                except Exception as e:
                    return f"Failed to scan full page: {e}"
            else:
                # Granular scroll control
                if amount == "half":
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.5)")
                elif amount == "quarter":
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.25)")
                else:
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.75)")
                
                info = await page.evaluate(JS_GET_SCROLL_INFO)
                status = "at the bottom" if info['isAtBottom'] else f"at {info['percent']}%"
                return f"Scrolled down. Now {status} of the page."

        @self.registry.action("Wait for network traffic to settle (Network Idle)")
        async def wait_for_network_idle(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
                return "Network is idle."
            except Exception:
                return "Timeout waiting for network idle (5s)."

        @self.registry.action("Wait for specific text to appear on the page")
        async def wait_for_content(browser: BrowserContext, text: str, timeout: int = 10):
            page = await browser.get_current_page()
            found = await wait_for_text(page, text, timeout=timeout * 1000)
            if found:
                return f"Text '{text}' appeared on the page."
            return f"Timeout: Text '{text}' did not appear within {timeout} seconds."

        @self.registry.action("Wait for an element matching a selector to appear")
        async def wait_for_element_by_selector(browser: BrowserContext, selector: str, timeout: int = 10):
            page = await browser.get_current_page()
            try:
                await page.wait_for_selector(selector, state="visible", timeout=timeout * 1000)
                return f"Element '{selector}' appeared."
            except Exception:
                return f"Timeout: Element '{selector}' did not appear within {timeout} seconds."

        @self.registry.action("List all iframes on the current page")
        async def list_iframes(browser: BrowserContext):
            page = await browser.get_current_page()
            frames = page.frames
            frame_data = []
            for i, frame in enumerate(frames):
                try:
                    is_main = (frame == page.main_frame)
                    frame_data.append({
                        "index": i,
                        "name": frame.name,
                        "url": frame.url,
                        "is_main_frame": is_main
                    })
                except Exception:
                    continue
            return f"Found {len(frames)} frames:\n{json.dumps(frame_data, indent=2)}"

        @self.registry.action("Wait for the URL to change (navigation to complete)")
        async def wait_for_url_change(browser: BrowserContext, expected_url: str = None, timeout: int = 10):
            page = await browser.get_current_page()
            current_url = page.url
            try:
                if expected_url:
                    await page.wait_for_url(expected_url, timeout=timeout * 1000)
                    return f"URL changed to match '{expected_url}'."
                else:
                    await page.wait_for_function(
                        "(oldUrl) => window.location.href !== oldUrl",
                        arg=current_url,
                        timeout=timeout * 1000
                    )
                    return f"URL changed from '{current_url}' to '{page.url}'."
            except Exception:
                return f"Timeout: URL did not change within {timeout} seconds."

        @self.registry.action("Navigate back in history")
        async def go_back(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.go_back()
            return f"Navigated back to {page.url}"

        @self.registry.action("Navigate forward in history")
        async def go_forward(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.go_forward()
            return f"Navigated forward to {page.url}"

        @self.registry.action("Scroll up the page")
        async def scroll_up(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.evaluate("window.scrollBy(0, -window.innerHeight * 0.8)")
            return "Scrolled up one screen."

        @self.registry.action("Check current scroll position and page length")
        async def check_scroll_status(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                info = await page.evaluate(JS_GET_SCROLL_INFO)
                return f"Scroll Status: {info['percent']}% viewed. At bottom: {info['isAtBottom']}."
            except Exception as e:
                return f"Error checking scroll status: {e}"

        @self.registry.action("Scroll down until specific text is found on the page")
        async def scroll_to_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                found = await page.evaluate(JS_SCROLL_TO_TEXT, text)
                if found:
                    return f"Scrolled and found text: '{text}'."
                else:
                    return f"Text '{text}' not found after scrolling."
            except Exception as e:
                return f"Error scrolling to text: {e}"

        @self.registry.action("Scroll down until a specific element selector is visible")
        async def scroll_to_element(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                found = await page.evaluate(JS_SCROLL_TO_SELECTOR, selector)
                if found:
                    return f"Scrolled and found element: '{selector}'."
                else:
                    return f"Element '{selector}' not found after scrolling."
            except Exception as e:
                return f"Error scrolling to element: {e}"
