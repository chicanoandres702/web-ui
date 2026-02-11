from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent

logger = logging.getLogger(__name__)


class PreStepHandler:
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def _process_control_queue(self) -> None:
        """Processes commands from the control queue."""
        await self.agent.control_queue_handler.process_control_queue()

    async def handle_pre_step(self) -> bool:
        """
        Handles pre-step checks: pause, stop, model switching, failures.

        Also manages cookie persistence across sessions

        Returns True if the agent should stop/break the loop.
        """
        # Process control queue
        await self._process_control_queue()

        # Manage Model Switching (Cost Saver / Smart Retry)
        self.agent.heuristics.manage_model_switching()

        # Suggest alternative strategies on failure
        if hasattr(self.agent, "_suggest_alternative_strategy"):
            self.agent._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self._manage_tabs()

        # Custom Memory Injection
        # if self.use_custom_memory and self.memory_manager:
        await self.agent.site_knowledge_injector.inject_site_knowledge()

        # Check if we should stop due to too many failures
        # Note: self.heuristics is a direct attribute of BrowserUseAgent

        # if await self.heuristics.check_max_failures():
        #    return True

        # Check if MFA is required and wait for approval

        if self.agent.heuristics.is_waiting_for_mfa:
            logger.info(
                "MFA required, waiting for approval from another device, after setting pre step..."
            )
            if getattr(self.agent, "send_agent_message_callback", None):
                await self.agent.send_agent_message_callback(  # type: ignore
                    {"type": "agent_status", "status": "Waiting for MFA 🔐"}
                )

            while self.agent.heuristics.is_waiting_for_mfa and not self.agent.state.stopped:
                await asyncio.sleep(15)  # Check every second
            logger.info("MFA approved or agent stopped.")
            self.agent.heuristics.is_waiting_for_mfa = False  # Reset flag

            # Verify there is still a valid page and browser context
            if not self.agent.browser_context or not hasattr(
                self.agent.browser_context, "get_current_page"
            ):
                logger.warning(
                    "Browser context is no longer valid after MFA, stopping agent."
                )
                self.agent.state.stopped = True
                return True

            return True

        if self.agent.state.stopped:
            logger.info("Agent stopped")
            return True

        while self.agent.state.paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self.agent.state.stopped:  # Allow stopping while paused
                return True
        # Load cookies at the beginning of each session
        if self.agent.cookie_path and os.path.exists(self.agent.cookie_path):
            if self.agent.browser_context:
                try:
                    with open(self.agent.cookie_path, "r", encoding="utf-8") as f:
                        cookies = json.load(f)

                        # browser-use BrowserContext objects store the Playwright context in .context
                        playwright_context = getattr(
                            self.agent.browser_context, "context", None
                        )
                        if playwright_context and hasattr(
                            playwright_context, "add_cookies"
                        ):
                            await playwright_context.add_cookies(cookies)
                        else:
                            logger.warning(
                                "Could not access Playwright context to add cookies."
                            )

                        logger.info(f"Loaded cookies from {self.agent.cookie_path}")
                except Exception as e:
                    logger.error(f"Failed to load cookies: {e}")
        return False

    async def _manage_tabs(self):
        """Closes extraneous tabs to keep the agent focused."""
        nav_controller = getattr(self.agent.controller, "nav_controller", None)
        if nav_controller and self.agent.browser_context:
            await nav_controller.manage_tabs(self.agent.browser_context)

    async def _inject_site_knowledge(self):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        await self.agent.site_knowledge_injector.inject_site_knowledge()
