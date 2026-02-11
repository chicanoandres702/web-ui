from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent


class ComponentInitializer:
    """Initializes and configures agent components."""

    def __init__(
        self,
        agent: BrowserUseAgent,
        agent_kwargs: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ):
        self.agent = agent
        self._setup_system_prompt(system_prompt)

    def _setup_system_prompt(self, system_prompt: Optional[str]):
        if system_prompt:
            self.agent.message_manager.system_prompt = SystemMessage(
                content=system_prompt
            )
