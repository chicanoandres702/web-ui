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