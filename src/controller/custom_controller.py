from browser_use.controller.service import Controller
from browser_use.browser.context import BrowserContext
import logging
import json
import os
import aiohttp
from src.utils.utils import save_text_to_file as utils_save_text, retry_async
from src.utils.browser_scripts import (
    COMMON_CLOSE_SELECTORS,
    JS_REMOVE_OVERLAYS,
    JS_SCROLL_SLOW,
    JS_EXTRACT_LINKS,
    JS_GET_STRUCTURE,
    JS_HIGHLIGHT_ELEMENTS,
    JS_REMOVE_HIGHLIGHTS,
    JS_GET_COMPUTED_STYLE,
    JS_GET_DROPDOWN_OPTIONS,
    JS_ENABLE_LOG_CAPTURE,
    JS_FIND_TEXT_ELEMENTS,
    JS_GET_LOCAL_STORAGE,
    JS_WAIT_FOR_DOM_STABILITY,
    JS_REMOVE_ADS,
    JS_GET_SCROLL_INFO,
    JS_ASSESS_SECTION,
    JS_SCROLL_TO_TEXT,
    JS_CHECK_TEXT_ELEMENT_STATUS,
    JS_CLOSE_COOKIE_BANNERS
)
from src.utils.knowledge_base import search_kb_files, load_kb_content, list_kb_files
from src.utils.memory_utils import reset_mem0

logger = logging.getLogger(__name__)

class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [], **kwargs):
        super().__init__(exclude_actions=exclude_actions, **kwargs)
        self.memory_file = None
        self.fast_mode = False
        self.require_confirmation = False
        self.kb_dir = None
        self.clipboard_content = ""
        self._register_custom_actions()

    def set_memory_file(self, memory_file: str):
        self.memory_file = memory_file

    def set_knowledge_base_dir(self, kb_dir: str):
        self.kb_dir = kb_dir
        
    async def close_mcp_client(self):
        # Placeholder for MCP client cleanup if implemented in registry
        pass

    def _register_custom_actions(self):
        self._register_navigation_actions()
        self._register_interaction_actions()
        self._register_extraction_actions()
        self._register_debugging_actions()
        self._register_system_actions()

    async def _smart_scan(self, page, script, args=None, check_empty=True):
        """Executes a script, and if results are empty, waits for DOM stability and retries."""
        # Execute first attempt
        result = await page.evaluate(script, args) if args else await page.evaluate(script)
        
        should_retry = False
        if check_empty:
            if isinstance(result, list) and len(result) == 0:
                should_retry = True
            elif isinstance(result, dict) and not result:
                should_retry = True
                
        if should_retry:
            logger.info("Smart Scan: Empty result detected. Waiting for DOM stability...")
            await page.evaluate(JS_WAIT_FOR_DOM_STABILITY)
            result = await page.evaluate(script, args) if args else await page.evaluate(script)
            
        return result

    def _register_navigation_actions(self):
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
                clicked = await page.evaluate(JS_CLOSE_COOKIE_BANNERS)
                return "Clicked a cookie consent button." if clicked else "No common cookie banner buttons found."
            except Exception as e:
                return f"Error closing cookie banner: {e}"

        @self.registry.action("Scroll down the page. Optional: amount='full' scans the whole page.")
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
                    return f"Scanned entire page (scrolled down and back up). Content preview:\n{preview}"
                except Exception as e:
                    return f"Failed to scan full page: {e}"
            else:
                # Simple scroll down
                await page.evaluate("window.scrollBy(0, window.innerHeight * 0.7)")
                # Get scroll info after scrolling to inform the agent
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

    def _register_extraction_actions(self):
        @self.registry.action("Extract all hyperlinks from the current page")
        async def extract_page_links(browser: BrowserContext):
            page = await browser.get_current_page()
            
            async def _extract():
                return await self._smart_scan(page, JS_EXTRACT_LINKS)

            try:
                links = await retry_async(
                    _extract,
                    logger=logger,
                    error_message="Link extraction failed"
                )
                
                # Deduplicate based on href
                unique_links = {}
                for link in links:
                    unique_links[link['href']] = link['text']
                
                formatted_links = [{"text": txt, "href": href} for href, txt in unique_links.items()]
                
                return f"Found {len(formatted_links)} unique links. Here are the top 50: {json.dumps(formatted_links[:50], indent=2)}"
            except Exception as e:
                return f"Failed to extract links after retries: {e}"

        @self.registry.action("Get page heading structure (H1-H6)")
        async def get_page_structure(browser: BrowserContext):
            page = await browser.get_current_page()
            structure = await self._smart_scan(page, JS_GET_STRUCTURE)
            return f"Page Structure: {json.dumps(structure, indent=2)}"

        @self.registry.action("Get full HTML and attributes of a specific element")
        async def inspect_element_details(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                element = page.locator(selector).first
                if not await element.count():
                    return f"Element not found: {selector}"
                
                html = await element.evaluate("el => el.outerHTML")
                box = await element.bounding_box()
                return f"HTML:\n{html}\n\nBounding Box: {box}"
            except Exception as e:
                return f"Error inspecting element: {e}"

        @self.registry.action("Get all cookies from the current context")
        async def get_cookies(browser: BrowserContext):
            page = await browser.get_current_page()
            cookies = await page.context.cookies()
            return json.dumps(cookies, indent=2)

        @self.registry.action("Set cookies for the current context")
        async def set_cookies(browser: BrowserContext, cookies_json: str):
            page = await browser.get_current_page()
            try:
                cookies = json.loads(cookies_json)
                if isinstance(cookies, dict):
                    cookies = [cookies]
                await page.context.add_cookies(cookies)
                return "Cookies set successfully."
            except Exception as e:
                return f"Error setting cookies: {e}"

        @self.registry.action("Get all local storage items from the current page")
        async def get_local_storage(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_GET_LOCAL_STORAGE)
                return f"Local Storage:\n{json.dumps(data, indent=2)}"
            except Exception as e:
                return f"Error getting local storage: {e}"

        @self.registry.action("Enable console log capture on the current page")
        async def enable_console_log_capture(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.evaluate(JS_ENABLE_LOG_CAPTURE)
            return "Console log capture enabled. Use 'get_console_logs' to retrieve them."

        @self.registry.action("Get captured console logs")
        async def get_console_logs(browser: BrowserContext):
            page = await browser.get_current_page()
            logs = await page.evaluate("() => window._captured_logs || []")
            if not logs:
                return "No logs captured (did you run 'enable_console_log_capture'?)."
            return json.dumps(logs, indent=2)

        @self.registry.action("Get computed style of an element to check visibility")
        async def get_element_computed_style(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                style = await page.evaluate(JS_GET_COMPUTED_STYLE, selector)
                
                if not style:
                    return f"Element not found: {selector}"
                
                return f"Computed Style for '{selector}':\n{json.dumps(style, indent=2)}"
            except Exception as e:
                return f"Error getting computed style: {e}"

        @self.registry.action("List all elements containing specific text")
        async def get_elements_by_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                elements = await self._smart_scan(page, JS_FIND_TEXT_ELEMENTS, args=text)
                return f"Found {len(elements)} visible elements containing '{text}':\n{json.dumps(elements, indent=2)}"
            except Exception as e:
                return f"Error finding elements by text: {e}"

        @self.registry.action("Get options from a dropdown/select element")
        async def get_dropdown_options(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                options = await self._smart_scan(page, JS_GET_DROPDOWN_OPTIONS, args=selector)
                
                if options is None:
                    return f"Element not found: {selector}"
                
                return f"Dropdown Options for '{selector}':\n{json.dumps(options, indent=2)}"
            except Exception as e:
                return f"Error getting options: {e}"

        @self.registry.action("Get HTML content of a specific iframe")
        async def get_iframe_content(browser: BrowserContext, frame_index: int):
            page = await browser.get_current_page()
            frames = page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return f"Frame index {frame_index} out of bounds."
            try:
                content = await frames[frame_index].content()
                return f"Content of iframe {frame_index} (truncated):\n{content[:4000]}"
            except Exception as e:
                return f"Error getting content: {e}"

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

        @self.registry.action("Assess the current page section to determine next step (Scroll vs Submit)")
        async def assess_page_section(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_ASSESS_SECTION)
                
                recommendation = []
                if data['unfilled_inputs'] > 0:
                    recommendation.append(f"Fill {data['unfilled_inputs']} visible inputs.")
                
                if data['actions']:
                    recommendation.append(f"Consider clicking: {', '.join(data['actions'])}.")
                
                if not data['at_bottom']:
                    recommendation.append("Scroll down to see more content.")
                else:
                    recommendation.append("You are at the bottom of the page.")
                
                status = f"Page Status: {data['progress']}% viewed."
                return f"{status}\nAnalysis: {' '.join(recommendation)}"
            except Exception as e:
                return f"Error assessing page: {e}"

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

        @self.registry.action("Check if an element containing specific text appears selected/active")
        async def check_element_state_by_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                results = await self._smart_scan(page, JS_CHECK_TEXT_ELEMENT_STATUS, args=text)
                if not results:
                    return f"No visible elements found containing text: '{text}'"
                
                # Filter for likely selected
                selected = [r for r in results if r['isLikelySelected']]
                if selected:
                    return f"YES. Found {len(selected)} element(s) with text '{text}' that appear selected/active. Details: {json.dumps(selected, indent=2)}"
                else:
                    return f"NO. Found elements with text '{text}', but none appear selected. They might be unselected or use a non-standard state indication."
            except Exception as e:
                return f"Error checking element state: {e}"

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

        @self.registry.action("Highlight specific elements on the page for debugging")
        async def highlight_elements(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                count = await page.evaluate(JS_HIGHLIGHT_ELEMENTS, selector)
                return f"Highlighted {count} element(s) matching '{selector}'."
            except Exception as e:
                return f"Error highlighting elements: {e}"

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

    def _register_interaction_actions(self):
        @self.registry.action("Simulate hovering over an element")
        async def hover_element(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                await page.hover(selector)
                return f"Hovered over element: {selector}"
            except Exception as e:
                return f"Error hovering over element: {e}"

        @self.registry.action("Simulate pressing a specific key combination")
        async def press_key(browser: BrowserContext, key: str):
            page = await browser.get_current_page()
            try:
                await page.keyboard.press(key)
                return f"Pressed key: {key}"
            except Exception as e:
                return f"Error pressing key: {e}"

        @self.registry.action("Upload a file to a specific selector")
        async def upload_file_to_element(browser: BrowserContext, selector: str, filename: str):
            page = await browser.get_current_page()
            safe_dir = os.path.abspath("./tmp/downloads")
            filepath = os.path.join(safe_dir, os.path.basename(filename))
            
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            try:
                await page.set_input_files(selector, filepath)
                return f"Uploaded {filename} to {selector}"
            except Exception as e:
                return f"Error uploading file: {e}"

        @self.registry.action("Select an option from a dropdown element")
        async def select_dropdown_option(browser: BrowserContext, selector: str, value: str):
            page = await browser.get_current_page()
            try:
                await page.select_option(selector, value)
                return f"Selected option '{value}' in {selector}"
            except Exception as e:
                return f"Error selecting option: {e}"

        @self.registry.action("Drag and drop an element to a target location")
        async def drag_and_drop(browser: BrowserContext, source_selector: str, target_selector: str):
            page = await browser.get_current_page()
            try:
                await page.drag_and_drop(source_selector, target_selector)
                return f"Dragged {source_selector} to {target_selector}"
            except Exception as e:
                return f"Error dragging and dropping: {e}"

        @self.registry.action("Execute JavaScript in a specific iframe")
        async def execute_js_in_iframe(browser: BrowserContext, frame_index: int, code: str):
            page = await browser.get_current_page()
            frames = page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return f"Frame index {frame_index} out of bounds."
            try:
                result = await frames[frame_index].evaluate(code)
                return f"Result: {json.dumps(result, default=str)}"
            except Exception as e:
                return f"Error executing JS: {e}"

        @self.registry.action("Copy text to internal clipboard")
        async def copy_to_clipboard(text: str):
            self.clipboard_content = text
            return "Text copied to internal clipboard."

        @self.registry.action("Paste text from internal clipboard to an element")
        async def paste_from_clipboard(browser: BrowserContext, selector: str):
            if not self.clipboard_content:
                return "Clipboard is empty."
            page = await browser.get_current_page()
            try:
                await page.fill(selector, self.clipboard_content)
                return f"Pasted content to {selector}"
            except Exception as e:
                return f"Error pasting content: {e}"

        @self.registry.action("Read internal clipboard content")
        async def read_clipboard():
            return self.clipboard_content if self.clipboard_content else "Clipboard is empty."

        @self.registry.action("Attempt to solve simple CAPTCHAs (checkboxes)")
        async def solve_captcha(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                # Google reCAPTCHA
                recaptcha_frame = page.frame_locator("iframe[src*='google.com/recaptcha'][src*='anchor']")
                if await recaptcha_frame.locator(".recaptcha-checkbox-border").count() > 0:
                    await recaptcha_frame.locator(".recaptcha-checkbox-border").click()
                    await page.wait_for_timeout(2000)
                    return "Clicked Google reCAPTCHA checkbox. Please verify if solved."
                
                # Cloudflare Turnstile
                cf_frame = page.frame_locator("iframe[src*='challenges.cloudflare.com']")
                if await cf_frame.locator("input[type='checkbox']").count() > 0:
                    await cf_frame.locator("input[type='checkbox']").click()
                    await page.wait_for_timeout(2000)
                    return "Clicked Cloudflare Turnstile checkbox."
                
                # Generic "I am human" buttons
                human_btns = page.locator("button:has-text('I am human'), button:has-text('Verify you are human')")
                if await human_btns.count() > 0:
                    await human_btns.first.click()
                    return "Clicked generic human verification button."

                return "No simple CAPTCHA found. If a complex CAPTCHA is present, please ask the user for help."
            except Exception as e:
                return f"Error attempting to solve CAPTCHA: {e}"

    def _register_system_actions(self):
        @self.registry.action("Save text content to a file")
        async def save_text_to_file(filename: str, content: str, append: bool = False):
            safe_dir = os.path.abspath("./tmp/downloads")
            os.makedirs(safe_dir, exist_ok=True)
            filepath = os.path.join(safe_dir, os.path.basename(filename))
            mode = "a" if append else "w"
            utils_save_text(filepath, content, mode)
            return f"Saved content to {filepath}"

        @self.registry.action("Clear browser cookies and local storage")
        async def clear_browser_data(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                context = page.context
                await context.clear_cookies()
                await page.evaluate("try { window.localStorage.clear(); window.sessionStorage.clear(); } catch (e) {}")
                return "Cleared cookies and local/session storage."
            except Exception as e:
                return f"Error clearing browser data: {e}"

        @self.registry.action("Clear agent memory (Mem0)")
        async def clear_agent_memory():
            reset_mem0()
            return "Agent memory (Mem0) cleared."

        @self.registry.action("Download a file from a URL")
        async def download_file(browser: BrowserContext, url: str, filename: str):
            path = browser.config.save_downloads_path
            if not path:
                return "Error: No download path configured."
            
            os.makedirs(path, exist_ok=True)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            filepath = os.path.join(path, safe_filename)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(filepath, 'wb') as f:
                                f.write(content)
                            return f"Successfully downloaded {url} to {filepath}"
                        else:
                            return f"Failed to download. Status code: {response.status}"
            except Exception as e:
                return f"Error downloading file: {e}"

        @self.registry.action("Search the knowledge base for information")
        async def search_knowledge_base(query: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            matches = search_kb_files(self.kb_dir, query)
            if not matches:
                return f"No matches found in knowledge base for query: '{query}'"
            
            # If few matches, return content of first few
            result = f"Found {len(matches)} matches:\n"
            for filename in matches[:3]: # Limit to top 3
                content = load_kb_content(self.kb_dir, filename)
                preview = content[:500] + "..." if len(content) > 500 else content
                result += f"\n--- File: {filename} ---\n{preview}\n"
            
            if len(matches) > 3:
                result += f"\n...and {len(matches) - 3} more files: {', '.join(matches[3:])}"
                
            return result

        @self.registry.action("Read the full content of a specific knowledge base file")
        async def read_knowledge_base_file(filename: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            content = load_kb_content(self.kb_dir, filename)
            if not content:
                return f"File '{filename}' not found or empty."
            
            return f"--- File: {filename} ---\n{content}"

        @self.registry.action("List all available knowledge base files")
        async def list_knowledge_base_files():
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            files = list_kb_files(self.kb_dir)
            if not files:
                return "No knowledge base files found."
            
            return f"Available Knowledge Base Files:\n" + "\n".join(files)