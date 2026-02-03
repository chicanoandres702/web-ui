from browser_use.controller.service import Controller
from browser_use.browser.context import BrowserContext
import logging
import json
import os
from src.utils.browser_scripts import JS_INJECT_HUD
from src.controller.navigation_controller import NavigationController
from src.controller.actions.navigation_actions import NavigationActionsMixin
from src.controller.actions.interaction_actions import InteractionActionsMixin
from src.controller.actions.extraction_actions import ExtractionActionsMixin
from src.controller.actions.debugging_actions import DebuggingActionsMixin
from src.controller.actions.system_actions import SystemActionsMixin

logger = logging.getLogger(__name__)

class CustomController(
    Controller,
    NavigationActionsMixin,
    InteractionActionsMixin,
    ExtractionActionsMixin,
    DebuggingActionsMixin,
    SystemActionsMixin
):
    def __init__(self, exclude_actions: list[str] = [], **kwargs):
        super().__init__(exclude_actions=exclude_actions, **kwargs)
        self.memory_file = None
        self.fast_mode = False
        self.require_confirmation = False
        self.kb_dir = None
        self.clipboard_content = ""
        self.nav_controller = NavigationController()
        self.webui_manager = None
        self.callbacks = {}
        self.disable_hud = False
        self._register_custom_actions()

    def set_memory_file(self, memory_file: str):
        self.memory_file = memory_file

    def set_knowledge_base_dir(self, kb_dir: str):
        self.kb_dir = kb_dir

    def set_webui_manager(self, manager):
        self.webui_manager = manager
        
    def set_callbacks(self, callbacks: dict):
        self.callbacks = callbacks

    async def close_mcp_client(self):
        # Placeholder for MCP client cleanup if implemented in registry
        pass

    async def execute_action_by_name(self, action_name: str, params: dict, browser_context: BrowserContext):
        """Executes a registered action by name, handling registry lookup."""
        if action_name not in self.registry:
            return f"Action '{action_name}' not found."
        
        action = self.registry[action_name]
        try:
            return await action.function(browser_context, **params)
        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return f"Error executing {action_name}: {e}"

    def _register_custom_actions(self):
        self._register_navigation_actions()
        self._register_interaction_actions()
        self._register_extraction_actions()
        self._register_debugging_actions()
        self._register_system_actions()

    async def _agent_control_handler(self, action: str):
        """Handles agent control actions from the browser (HUD)."""
        if self.webui_manager and self.webui_manager.bu_agent:
            if action == 'pause':
                if self.webui_manager.bu_agent.state.paused:
                    self.webui_manager.bu_agent.resume()
                    logger.info("Agent resumed via HUD")
                else:
                    self.webui_manager.bu_agent.pause()
                    logger.info("Agent paused via HUD")
            elif action == 'stop':
                self.webui_manager.bu_agent.stop()
                logger.info("Agent stopped via HUD")

    async def _expose_agent_control(self, browser: BrowserContext):
        """Exposes the py_agent_control function to the browser context."""
        if getattr(browser, "_agent_control_exposed", False):
            return

        try:
            # Handle wrapper objects that don't expose expose_function directly
            # Try to get the underlying Playwright context
            context = getattr(browser, "context", None)
            if context and hasattr(context, "expose_function"):
                await context.expose_function("py_agent_control", self._agent_control_handler)
            else:
                # Fallback: Expose on the current page
                page = await browser.get_current_page()
                await page.expose_function("py_agent_control", self._agent_control_handler)
            
            browser._agent_control_exposed = True
        except Exception as e:
            if "already" in str(e).lower():
                browser._agent_control_exposed = True
            else:
                logger.warning(f"Failed to expose agent control: {e}")

    async def _update_hud_impl(self, browser: BrowserContext, data: dict):
        """Helper to update HUD to avoid duplication."""
        if self.disable_hud:
            return "HUD disabled."
        page = await browser.get_current_page()
        await self._expose_agent_control(browser)
        try:
            await page.evaluate(JS_INJECT_HUD, data)
            return "Updated HUD overlay."
        except Exception as e:
            return f"Error updating HUD: {e}"

    async def refresh_hud(self, browser: BrowserContext, last_action: str = None):
        """Refreshes the HUD overlay using current WebUI state."""
        if self.disable_hud or not self.webui_manager:
            return
        try:
            page = await browser.get_current_page()
            await self._expose_agent_control(browser)
            plan = getattr(self.webui_manager, "bu_plan", [])
            goal = getattr(self.webui_manager, "current_goal", "No goal set")
            hud_data = {
                "plan": plan,
                "goal": goal,
                "last_action": last_action or "Status Update"
            }
            await self._update_hud_impl(browser, hud_data)
        except Exception as e:
            logger.debug(f"Failed to refresh HUD: {e}")

        @self.registry.action("Handle the next browser dialog (alert/confirm/prompt)")
        async def handle_next_dialog(browser: BrowserContext, action: str = "accept", prompt_text: str = None):
            page = await browser.get_current_page()
            
            async def handle_dialog(dialog):
                if action == "accept":
                    await dialog.accept(prompt_text)
                else:
                    await dialog.dismiss()
            
            page.once("dialog", handle_dialog)
            return f"Set up handler to {action} the next dialog."

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

        @self.registry.action("Read the full text content of the current page")
        async def read_page_content(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                content = await page.evaluate("document.body.innerText")
                return f"Page Content ({len(content)} chars):\n{content[:5000]}..." # Truncate for log, but agent gets it
            except Exception as e:
                return f"Error reading page content: {e}"

        @self.registry.action("Save the current page text content to a file for review")
        async def save_page_content(browser: BrowserContext, filename: str = "page_content.txt"):
            page = await browser.get_current_page()
            try:
                content = await page.evaluate("document.body.innerText")
                title = await page.title()
                url = page.url
                full_text = f"URL: {url}\nTitle: {title}\n\n{content}"
                
                safe_dir = os.path.abspath("./tmp/downloads")
                os.makedirs(safe_dir, exist_ok=True)
                filepath = os.path.join(safe_dir, os.path.basename(filename))
                
                utils_save_text(filepath, full_text, "w")
                return f"Saved page content to {filepath}"
            except Exception as e:
                return f"Error saving page content: {e}"

    def _register_system_actions(self):
        @self.registry.action("Update the HUD (Heads Up Display) overlay on the page")
        async def update_hud(browser: BrowserContext, data_json: str):
            try:
                data = json.loads(data_json)
                return await self._update_hud_impl(browser, data)
            except Exception as e:
                return f"Error parsing HUD data: {e}"

        @self.registry.action("Add a new step to the agent's execution plan")
        async def add_plan_step(browser: BrowserContext, step_description: str):
            if self.webui_manager and hasattr(self.webui_manager, "bu_plan"):
                new_step = {"step": step_description, "status": "pending"}
                self.webui_manager.bu_plan.append(new_step)
                self.webui_manager.bu_plan_updated = True
                # Also update the overlay immediately if possible
                try:
                    hud_data = {
                        "plan": self.webui_manager.bu_plan,
                        "goal": getattr(self.webui_manager, "current_goal", "Processing..."),
                        "last_action": f"Added step: {step_description}"
                    }
                    await self._update_hud_impl(browser, hud_data)
                except: pass
                return f"Added step to plan: '{step_description}'"
            return "Plan manager not available."

        @self.registry.action("Mark a specific step in the plan as completed")
        async def mark_step_complete(browser: BrowserContext, step_index: int):
            if self.webui_manager and hasattr(self.webui_manager, "bu_plan"):
                idx = step_index - 1 # Convert 1-based to 0-based
                if 0 <= idx < len(self.webui_manager.bu_plan):
                    self.webui_manager.bu_plan[idx]["status"] = "completed"
                    self.webui_manager.bu_plan_updated = True
                    try:
                        hud_data = {
                            "plan": self.webui_manager.bu_plan,
                            "goal": getattr(self.webui_manager, "current_goal", "Processing..."),
                            "last_action": f"Completed step {step_index}"
                        }
                        await self._update_hud_impl(browser, hud_data)
                    except: pass
                    return f"Marked step {step_index} as completed."
                return f"Step index {step_index} out of bounds."
            return "Plan manager not available."

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

        @self.registry.action("Save specific navigation knowledge for a website")
        async def save_site_knowledge(browser: BrowserContext, domain: str, knowledge: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            # Use the shared memory manager for consistency
            manager = get_memory_manager(self.kb_dir)
            
            if manager.save_site_knowledge(domain, knowledge):
                 return f"Saved knowledge for {domain}."
            return f"Failed to save knowledge for {domain}."

        @self.registry.action("Save general knowledge/study notes to the knowledge base")
        async def save_to_knowledge_base(filename: str, content: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            os.makedirs(self.kb_dir, exist_ok=True)
            
            if not (filename.endswith('.md') or filename.endswith('.txt')):
                filename += '.md'
            
            safe_filename = sanitize_filename(filename)
            filepath = os.path.join(self.kb_dir, safe_filename)
            utils_save_text(filepath, content, "w")
            return f"Saved knowledge to {filepath}"

        @self.registry.action("Retrieve stored knowledge for a website")
        async def get_site_knowledge(browser: BrowserContext, domain: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            manager = get_memory_manager(self.kb_dir)
            content = manager.get_site_knowledge(domain)
            
            if content:
                return f"Stored Knowledge for {domain}:\n{content}"
            return f"No specific knowledge found for {domain}."

        @self.registry.action("Download a file from a URL")
        async def download_file(browser: BrowserContext, url: str, filename: str):
            path = browser.config.save_downloads_path
            if not path:
                return "Error: No download path configured."
            
            os.makedirs(path, exist_ok=True)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            filepath = os.path.join(path, safe_filename)
            
            page = await browser.get_current_page()
            
            # Strategy 1: Fetch via Page (Uses Browser Cookies/Session)
            try:
                js_script = """
                    async (url) => {
                        const response = await fetch(url);
                        if (!response.ok) throw new Error(response.statusText);
                        const blob = await response.blob();
                        return new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        });
                    }
                """
                data_url = await page.evaluate(js_script, url)
                # data_url is like "data:application/pdf;base64,JVBERi0xLjQK..."
                if "," in data_url:
                    _, encoded = data_url.split(",", 1)
                else:
                    encoded = data_url
                
                data = base64.b64decode(encoded)
                with open(filepath, 'wb') as f:
                    f.write(data)
                return f"Successfully downloaded {url} to {filepath}"
            except Exception as e_page:
                logger.warning(f"Page fetch failed for {url}: {e_page}. Trying fallback.")
            
            # Strategy 2: Fallback to aiohttp (Manual Cookies)
            try:
                cookies = await page.context.cookies()
                cookie_dict = {c['name']: c['value'] for c in cookies}
                
                async with aiohttp.ClientSession(cookies=cookie_dict) as session:
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