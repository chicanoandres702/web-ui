import os
import logging
from typing import Dict, Any, Tuple, Optional

from src.utils.browser_factory import create_browser, create_context
from src.controller.custom_controller import CustomController
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils.prompts import FULL_SYSTEM_PROMPT, KNOWLEDGE_ONLY_INSTRUCTION

logger = logging.getLogger(__name__)


async def prepare_directories(browser_settings: Dict[str, Any], task_id: str) -> Tuple[str, str]:
    """Prepares directories for history and recordings."""
    save_history_path = browser_settings.get("save_agent_history_path", "./tmp/agent_history")
    os.makedirs(save_history_path, exist_ok=True)

    history_dir = os.path.join(save_history_path, task_id)
    os.makedirs(history_dir, exist_ok=True)

    history_file = os.path.join(history_dir, f"{task_id}.json")

    save_recording_path = browser_settings.get("save_recording_path")
    gif_path = ""
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
        gif_path = os.path.join(save_recording_path, f"{task_id}.gif")

    return history_file, gif_path


async def configure_controller(
        webui_manager,
        agent_settings: Dict[str, Any],
        memory_file: Optional[str],
        extraction_model: Optional[str],
        callbacks: Dict[str, Any]
):
    """Configures the CustomController with settings and callbacks."""
    controller = CustomController()

    if agent_settings.get("use_memory") and memory_file:
        controller.set_memory_file(memory_file)
        # Configure KB directory based on memory file location
        kb_dir = os.path.dirname(os.path.abspath(memory_file))
        controller.set_knowledge_base_dir(kb_dir)
        
    # Set controller attributes based on settings
    controller.fast_mode = agent_settings.get("fast_mode", False)
    controller.require_confirmation = agent_settings.get("require_confirmation", False)

    # Store callbacks on the controller if needed, or just on the manager/agent later
    # For CustomController, we assume it might use them if we attach them,
    # or we rely on the agent construction to use them.
    
    webui_manager.bu_controller = controller
    return controller


async def initialize_browser_infrastructure(webui_manager, browser_settings: Dict[str, Any]):
    """Initializes browser and context."""
    if webui_manager.bu_browser:
        await webui_manager.bu_browser.close()

    webui_manager.bu_browser = create_browser(browser_settings)
    webui_manager.bu_browser_context = await create_context(webui_manager.bu_browser, browser_settings)


async def construct_agent(
        webui_manager,
        task: str,
        agent_settings: Dict[str, Any],
        llms: tuple,
        history_file: str,
        gif_path: str,
        callbacks: Dict[str, Any]
):
    """Constructs the BrowserUseAgent."""
    main_llm, planner_llm, confirmer_llm, smart_retry_llm, cheap_llm = llms

    system_prompt = FULL_SYSTEM_PROMPT
    if agent_settings.get("override_system_prompt"):
        system_prompt = agent_settings["override_system_prompt"]
    elif agent_settings.get("extend_system_prompt"):
        system_prompt += f"\n\n{agent_settings['extend_system_prompt']}"

    if agent_settings.get("restrict_to_knowledge_base"):
        system_prompt += f"\n\n{KNOWLEDGE_ONLY_INSTRUCTION}"

    agent_kwargs = {
        "task": task,
        "llm": main_llm,
        "browser": webui_manager.bu_browser,
        "browser_context": webui_manager.bu_browser_context,
        "controller": webui_manager.bu_controller,
        "use_vision": agent_settings.get("use_vision", True),
        "use_memory": agent_settings.get("use_memory", False),
        "save_conversation_path": os.path.join(os.path.dirname(history_file), "conversation.json"),
        "save_history_path": history_file,
        "system_prompt": system_prompt,
        "confirmer_llm": confirmer_llm,
        "confirmer_strictness": agent_settings.get("confirmer_strictness"),
        "use_vision_for_confirmer": agent_settings.get("confirmer_use_vision", True),
        "smart_retry_llm": smart_retry_llm,
        "cheap_llm": cheap_llm,
        "enable_cost_saver": agent_settings.get("enable_cost_saver", False),
        "auto_save_on_stuck": agent_settings.get("auto_save_on_stuck", False),
        "source": "webui",
        "max_actions_per_step": int(agent_settings.get("max_actions", 10)),
    }

    webui_manager.bu_agent = BrowserUseAgent(**agent_kwargs)

    # Attach callbacks to the agent instance so they can be triggered during execution
    if callbacks:
        for name, func in callbacks.items():
            setattr(webui_manager.bu_agent, name, func)