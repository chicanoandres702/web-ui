import logging
import os
import json
from browser_use.browser.context import BrowserContext
from src.utils.browser_scripts import JS_HIGHLIGHT_ELEMENTS, JS_REMOVE_HIGHLIGHTS

logger = logging.getLogger(__name__)

class DebuggingActionsMixin:
    def _register_debugging_actions(self):
        @self.registry.action("Take a full page screenshot and save to file")
        async def take_full_page_screenshot(browser: BrowserContext, filename: str):
            page = await browser.get_current_page()
            safe_dir = os.path.abspath("./tmp/downloads")
            os.makedirs(safe_dir, exist_ok=True)
            
            base_name = os.path.basename(filename)
            if not base_name.lower().endswith('.png'):
                base_name += '.png'
            
            filepath = os.path.join(safe_dir, base_name)
            
            try:
                await page.screenshot(path=filepath, full_page=True)
                return f"Full page screenshot saved to {filepath}"
            except Exception as e:
                return f"Error taking screenshot: {e}"

        @self.registry.action("Take a screenshot of a specific element and save to file")
        async def take_element_screenshot(browser: BrowserContext, selector: str, filename: str):
            page = await browser.get_current_page()
            safe_dir = os.path.abspath("./tmp/downloads")
            os.makedirs(safe_dir, exist_ok=True)
            
            base_name = os.path.basename(filename)
            if not base_name.lower().endswith('.png'): base_name += '.png'
            filepath = os.path.join(safe_dir, base_name)
            try:
                await page.locator(selector).first.screenshot(path=filepath)
                return f"Element screenshot saved to {filepath}"
            except Exception as e:
                return f"Error taking element screenshot: {e}"

        @self.registry.action("Highlight specific elements on the page for debugging")
        async def highlight_element(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                count = await page.evaluate(JS_HIGHLIGHT_ELEMENTS, selector)
                if count > 0:
                    return f"Highlighted {count} element(s) matching '{selector}'."
                return f"No elements found matching '{selector}'."
            except Exception as e:
                return f"Error highlighting element: {e}"

        @self.registry.action("Remove all highlighting from the page")
        async def remove_highlights(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                await page.evaluate(JS_REMOVE_HIGHLIGHTS)
                return "Removed debugging highlights."
            except Exception as e:
                return f"Error removing highlights: {e}"

        @self.registry.action("Execute arbitrary JavaScript on the current page")
        async def execute_javascript(browser: BrowserContext, code: str):
            page = await browser.get_current_page()
            try:
                result = await page.evaluate(code)
                return f"JavaScript executed successfully. Result: {json.dumps(result, default=str)}"
            except Exception as e:
                return f"Error executing JavaScript: {e}"
