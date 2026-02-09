import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ComponentInitializer:
    """
    Initializes agent components and applies system prompt override.
    This class centralizes the initialization logic for various agent components,
    making the BrowserUseAgent class cleaner and more focused.
    """

    def __init__(self, agent: Any, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None) -> None:
        """
        Initializes the agent's components.

        Args:
            agent: The BrowserUseAgent instance.
            agent_kwargs: Keyword arguments passed to the agent.
            system_prompt: An optional system prompt to override the default.
        """
        self.agent = agent
        self._apply_system_prompt_override(system_prompt)

    def _apply_system_prompt_override(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt:
            self.agent.system_prompt = system_prompt  # type: ignore