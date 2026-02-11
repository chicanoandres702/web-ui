from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent


logger = logging.getLogger(__name__)


class CompletionCheckHandler:
    """Handles the logic for checking if a task is completed and validating the output."""

    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def check_completion(self) -> bool:
        """
        Checks if the agent has marked the task as done and performs validation.
        """
        if self.agent.state.history.is_done():
            if self.agent.confirmer_llm:
                if not await self.agent._validate_output():
                    return False  # Validation failed, continue loop

            await self.agent.log_completion()
            return True
        return False
