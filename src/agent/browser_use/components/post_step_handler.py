from __future__ import annotations
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent
    from src.agent.browser_use.components.completion_check_handler import (
        CompletionCheckHandler,
    )

logger = logging.getLogger(__name__)


class PostStepHandler:
    """Handles post-step logic: callbacks, persistence, completion check."""

    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def handle_post_step(self, step: int, max_steps: int) -> bool:
        await self.agent._execute_step_callbacks(step)

        # Persistence: Auto-save history after each step
        if self.agent.save_history_path and hasattr(
            self.agent.cleanup_handler, "save_history_async"
        ):
            try:
                await self.agent.cleanup_handler.save_history_async(
                    self.agent.save_history_path
                )
            except Exception as e:
                logger.warning(f"Failed to auto-save history: {e}")

        await self.agent.heuristics.check_completion_heuristics()

        # Check for loops
        self.agent.heuristics.detect_loop()

        # Check for progress indicators
        await self.agent.heuristics.detect_progress()

        # Check for blocking elements
        await self.agent.heuristics.check_blocking_elements()

        # Check navigation recovery
        await self.agent.heuristics.check_navigation_recovery()

        # Check for complex page structure and add subtasks
        await self.agent.heuristics.check_and_add_subtasks()

        # completion_check_handler = CompletionCheckHandler(self)
        # await completion_check_handler.check_completion()
        completion_check_handler: CompletionCheckHandler = self.agent.completion_check_handler
        await completion_check_handler.check_completion()

        # Check login status (New)
        await self.agent.heuristics.check_login_status()

        if self.agent.cookie_path:
            await self.agent.action_handler.save_cookies(self.agent.cookie_path)

    async def _handle_shutdown_cookie_persistence(self) -> None:
        """Handles saving cookies on shutdown if a browser context exists."""
        await self.agent.action_handler.save_cookies(self.cookie_path)

        # --- User Interaction Dialog ---
        if await self._handle_user_interaction():
            return True  # Stop requested

        if self.agent.state.history.is_done():
            if not await self.agent._validate_output():
                return False  # Validation failed, continue loop

            await self.agent.cleanup_handler.cleanup()
            return True  # Task done, break loop

        return False

    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        """
        handler = self.agent.user_interaction_handler
        return await handler.handle_user_interaction()
