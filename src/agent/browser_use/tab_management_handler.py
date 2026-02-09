import logging
from typing import Any

logger = logging.getLogger(__name__)

class TabManagementHandler:
    """
    Handles tab management for the agent.
    This class is responsible for closing extraneous tabs to keep the agent focused.
    """

    def __init__(self, agent: Any):
        """Initializes the TabManagementHandler with the agent instance."""
        self.agent = agent

    async def manage_tabs(self, browser_context: Any) -> None:
        """Closes extraneous tabs to keep the agent focused."""
        nav_controller = getattr(self.agent.controller, 'nav_controller', None)
        if nav_controller:
            await nav_controller.manage_tabs(browser_context)