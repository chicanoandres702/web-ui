import logging
from typing import Any

from browser_use.agent.gif import create_history_gif

logger = logging.getLogger(__name__)

class AgentCleanupHandler:
    """Handles agent cleanup tasks."""

    def __init__(self, agent: Any):
        """Initializes the AgentCleanupHandler with the agent instance."""
        self.agent = agent

    async def handle_cleanup(self) -> None:
        """Handles cleanup after run, including closing the browser and generating a GIF."""
        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel()
            logger.info("🚫 Cleanup inhibited: Browser will remain open.")
        else:
            if not getattr(self.agent, 'inhibit_close', False):
                try:
                    await self.agent.close()
                except Exception as e:
                    logger.debug(f"Error during agent cleanup (likely benign): {e}")

        if self.agent.settings and getattr(self.agent.settings, 'generate_gif', False):
            output_path: str = 'agent_history.gif'
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            try:
                create_history_gif(task=self.agent.task, history=self.agent.state.history, output_path=output_path)
            except Exception as e:
                logger.error(f"Failed to generate history GIF: {e}")