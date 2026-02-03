import logging
from typing import Any, Optional
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.controller.custom_controller import CustomController
from src.utils.prompts import FULL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

async def create_agent(
    llm: Any,
    browser: Any,
    browser_context: Any,
    task: str,
    use_vision: bool = True,
    memory_file: Optional[str] = None,
    controller: Optional[CustomController] = None,
    system_prompt: str = FULL_SYSTEM_PROMPT,
    **kwargs
) -> BrowserUseAgent:
    """
    Factory to create a BrowserUseAgent with standard configuration.
    Centralizes agent creation logic to ensure consistency across tools and UI.
    """
    if controller is None:
        controller = CustomController()
        if memory_file:
            controller.set_memory_file(memory_file)

    return BrowserUseAgent(
        task=task,
        llm=llm,
        browser=browser,
        browser_context=browser_context,
        controller=controller,
        use_vision=use_vision,
        system_prompt=system_prompt,
        **kwargs
    )