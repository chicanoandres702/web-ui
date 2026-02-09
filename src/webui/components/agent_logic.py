import os
import logging
from typing import List, Dict, Any
from src.utils.browser_factory import create_browser, create_context
from src.controller.custom_controller import CustomController
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils.utils import ensure_default_extraction_models
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

async def prepare_directories(browser_settings, task_id):
    """Ensures necessary directories exist and returns paths."""
    # Base directories
    dirs = [
        "./tmp/downloads",
        "./tmp/agent_history",
        "./tmp/recordings",
        "./tmp/traces",
        "./tmp/memory"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    ensure_default_extraction_models()
    
    # Task specific paths
    save_history_path = browser_settings.get("save_agent_history_path")
    if not save_history_path:
        save_history_path = "./tmp/agent_history"
    os.makedirs(save_history_path, exist_ok=True)
    
    # Create task directory
    task_dir = os.path.join(save_history_path, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    history_file = os.path.join(task_dir, f"{task_id}.json")
    
    # Recording path
    save_recording_path = browser_settings.get("save_recording_path")
    if not save_recording_path:
        save_recording_path = "./tmp/recordings"
    os.makedirs(save_recording_path, exist_ok=True)
    gif_path = os.path.join(save_recording_path, f"{task_id}.gif")
    
    return history_file, gif_path

async def initialize_browser_infrastructure(webui_manager, browser_settings):
    """Initializes the browser and context."""
    if webui_manager.bu_browser is None:
        webui_manager.bu_browser = create_browser(browser_settings)
    
    if webui_manager.bu_browser_context is None:
        webui_manager.bu_browser_context = await create_context(webui_manager.bu_browser, browser_settings)
        
    return webui_manager.bu_browser, webui_manager.bu_browser_context

async def configure_controller(webui_manager, agent_settings, memory_file=None, extraction_model=None, callbacks=None):
    """Configures the controller with actions."""
    controller = CustomController()
    
    if agent_settings.get("disable_hud", False):
        controller.disable_hud = True

    if memory_file:
        controller.set_memory_file(memory_file)
        
    if callbacks:
        controller.set_callbacks(callbacks)

    controller.set_webui_manager(webui_manager)
    
    webui_manager.bu_controller = controller
    return controller

async def construct_agent(webui_manager, task, agent_settings, llms, history_file, gif_path, callbacks=None, initial_actions=None, current_step_index=None):
    """Constructs the BrowserUseAgent."""
    main_llm, planner_llm, confirmer_llm, priority_llms = llms
    
    # Extract settings
    use_vision = agent_settings.get("use_vision", True)
    max_input_tokens = agent_settings.get("max_input_tokens", 128000)
    confirmer_strictness = agent_settings.get("confirmer_strictness", 7)
    enable_cost_saver = agent_settings.get("enable_cost_saver", False)    
    enable_smart_retry = agent_settings.get("enable_smart_retry", False)
    auto_save_on_stuck = agent_settings.get("auto_save_on_stuck", True)
    use_memory = agent_settings.get("use_memory", False)
    override_system_prompt = agent_settings.get("override_system_prompt", None)
    max_consecutive_failures = agent_settings.get("max_consecutive_failures", 1000)
    extend_system_prompt = agent_settings.get("extend_system_prompt", None)
    
    final_system_prompt = override_system_prompt
    if extend_system_prompt:
        if final_system_prompt:
            final_system_prompt += f"\n\n{extend_system_prompt}"
        else:
            final_system_prompt = extend_system_prompt

    agent = BrowserUseAgent(
        task=task,
        llm=main_llm,
        browser_context=webui_manager.bu_browser_context,
        controller=webui_manager.bu_controller,
        use_vision=use_vision,
        max_input_tokens=max_input_tokens,
        generate_gif=gif_path,
        
        # Custom BrowserUseAgent parameters
        confirmer_llm=confirmer_llm,
        confirmer_strictness=confirmer_strictness,
        model_priority_list=priority_llms,
        enable_cost_saver=enable_cost_saver,
        enable_smart_retry=enable_smart_retry,
        auto_save_on_stuck=auto_save_on_stuck,
        use_memory=use_memory,
        system_prompt=final_system_prompt,
        save_history_path=history_file,
        initial_actions=initial_actions,
        current_step_index=current_step_index,
        max_consecutive_failures=max_consecutive_failures,
    )
    
    if callbacks:
        if "step_callback" in callbacks:
            agent.step_callback = callbacks["step_callback"]
        if "done_callback" in callbacks:
            agent.done_callback = callbacks["done_callback"]
            
    webui_manager.bu_agent = agent
    return agent
