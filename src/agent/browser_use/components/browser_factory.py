from __future__ import annotations
from typing import Any


class BrowserFactory:
    """
    Handles browser creation and context management
    """

    def __init__(self, agent: Any):
        self.agent = agent

    async def create_browser(self):
        # logic from websocket/run_browser_agent and browser_use_agent here for modularity.
        pass
