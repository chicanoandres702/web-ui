from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING, Union


from browser_use.agent.gif import create_history_gif
from browser_use.agent.views import AgentHistory, ActionResult, BrowserStateHistory
from src.utils.io_manager import IOManager

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent
    from src.agent.browser_use.user_interaction_handler import UserInteractionHandler


logger = logging.getLogger(__name__)


class AgentCleanupHandler:
    """Handles cleanup tasks for the agent."""

    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def _manage_tabs(self):
        """Closes extraneous tabs to keep the agent focused."""
        nav_controller = getattr(self.agent.controller, "nav_controller", None)
        if nav_controller:
            await nav_controller.manage_tabs(self.agent.browser_context)

    async def _inject_site_knowledge(self):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        await self.agent.site_knowledge_injector.inject_site_knowledge()

    async def cleanup(self):
        """Performs cleanup of tasks and resources."""
        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel()
            try:
                await self.agent.planner_task
            except asyncio.CancelledError:
                pass

        if not self.agent.inhibit_close:
            # MessageManager does not have a close() method in browser-use.
            # If there are specific cleanup tasks for message_manager, they should be handled here.
            # Otherwise, we rely on the parent Agent class or garbage collection.
            pass

        if self.agent.settings and getattr(self.agent.settings, "generate_gif", False):
            output_path = "agent_history.gif"
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(
                task=self.agent.task,
                history=self.agent.state.history,
                output_path=output_path,
            )

        await self.agent._execute_done_callback()

    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = (
            self.agent.state.history.history[-1]
            if self.agent.state.history.history
            else None
        )
        intel = "No specific new information from this step."
        if last_step_info and last_step_info.model_output:
            # Use getattr to safely access 'thought' as it might not be in the schema
            thought = getattr(last_step_info.model_output, "thought", None)
            if thought:
                intel = thought

        next_action_desc = "Agent is considering its next move."
        if (
            last_step_info
            and last_step_info.model_output
            and last_step_info.model_output.action
        ):
            action_strings = []
            actions = (
                last_step_info.model_output.action
                if isinstance(last_step_info.model_output.action, list)
                else [last_step_info.model_output.action]
            )
            for action_model in actions:
                if hasattr(action_model, "model_dump"):
                    action_strings.append(str(action_model.model_dump()))
                else:
                    action_strings.append(str(action_model))
            next_action_desc = ", ".join(action_strings)
        if not next_action_desc:
            next_action_desc = "Agent is considering its next move."

        handler: UserInteractionHandler = self.agent.user_interaction_handler
        user_decision = await handler.request_user_decision(intel, next_action_desc)

        if user_decision is False:  # type: ignore # User chose No/Abort
            self.agent.state.stopped = True
            return True  # Break the loop
        elif isinstance(user_decision, str):  # User provided a custom task
            self.agent.task = user_decision  # Update the main task for the agent to re-evaluate

            self.agent.heuristics.inject_message(
                f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan."
            )

            return False  # Continue the loop, agent will now work on the new task

        return False

    async def _handle_max_steps_reached(self):
        """Handles the case where the agent reaches maximum steps without completion."""
        error_message = "Failed to complete task in maximum steps"
        self.agent.state.history.history.append(
            AgentHistory(
                model_output=None,
                result=[ActionResult(error=error_message, include_in_memory=True)],
                state=BrowserStateHistory(
                    url="",
                    title="",
                    tabs=[],
                    interacted_element=[],
                    screenshot=None,
                ),
                metadata=None,
            )
        )

        logger.info(f"❌ {error_message}")

    async def save_history_async(self, path: str):
        """Saves the agent history to a file asynchronously."""
        try:
            await IOManager.write_file(
                path, self.agent.state.history.model_dump_json(indent=2)
            )
        except Exception as e:
            logger.error(f"Failed to save history asynchronously: {e}")

    async def _handle_shutdown(self) -> None:
        """Handle cleanup after run."""

        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel()  # type: ignore
            logger.info("🚫 Cleanup inhibited: Browser will remain open.")  # type: ignore
        else:
            try:
                await self.agent.close()
            except Exception as e:
                # Log as debug to avoid noise during forced shutdowns/interrupts
                logger.debug(f"Error during agent cleanup (likely benign): {e}")

        if self.agent.settings and getattr(self.agent.settings, "generate_gif", False):
            output_path: str = "agent_history.gif"
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(
                task=self.agent.task,
                history=self.agent.state.history,
                output_path=output_path,
            )

        # Execute Done Callback
        await self.agent._execute_done_callback()
