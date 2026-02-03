import json
import os

import gradio as gr
from gradio.components import Component
from typing import Any, Dict, Optional
from src.webui.webui_manager import WebuiManager
from src.utils import config
import logging
from functools import partial
from src.webui.components.shared import create_llm_settings_ui, safe_execution, create_template_manager_ui, setup_template_manager_logic, update_mcp_server
from src.utils.prompts import DEFAULT_PLANNER_PROMPT

logger = logging.getLogger(__name__)


def create_agent_settings_tab(webui_manager: WebuiManager):
    """
    Creates an agent settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Column():
            override_system_prompt = gr.Textbox(label="Override system prompt", lines=4, interactive=True)
            extend_system_prompt = gr.Textbox(label="Extend system prompt", lines=4, interactive=True)

    with gr.Group():
        mcp_json_file = gr.File(label="MCP server json", interactive=True, file_types=[".json"])
        mcp_server_config = gr.Textbox(label="MCP server", lines=6, interactive=True, visible=False)

    # --- Main LLM ---
    llm_comps, _ = create_llm_settings_ui(
        prefix="llm",
        label="LLM",
        default_provider="ollama",
        include_vision=True,
        default_vision=False,
        include_ctx=True
    )

    # --- Confirmer LLM ---
    confirmer_comps, _ = create_llm_settings_ui(
        prefix="confirmer_llm",
        label="Confirmer LLM",
        default_provider="ollama",
        include_vision=True,
        default_vision=False,
        include_ctx=True,
        include_strictness=True,
        default_strictness=7
    )

    with gr.Group():
        with gr.Row():
            gr.Markdown("### 🧠 Smart Retry Settings")
        with gr.Row():
            enable_smart_retry = gr.Checkbox(
                label="Enable Smart Retry",
                value=True,
                info="Automatically switch to a stronger model after 2 consecutive failures.",
                interactive=True
            )
        
        smart_retry_comps, smart_retry_container = create_llm_settings_ui(
            prefix="smart_retry_llm",
            label="Retry",
            default_provider="ollama",
            visible=False
        )

    enable_smart_retry.change(lambda x: gr.update(visible=x), inputs=[enable_smart_retry], outputs=[smart_retry_container])

    with gr.Group():
        with gr.Row():
            gr.Markdown("### 💰 Cost Saver Settings")
        with gr.Row():
            enable_cost_saver = gr.Checkbox(
                label="Enable Cost Saver",
                value=False,
                info="Use a cheaper model for simple tasks, switching to the main model only on failure.",
                interactive=True
            )
        
        cheap_comps, cheap_container = create_llm_settings_ui(
            prefix="cheap_llm",
            label="Cheap",
            default_provider="ollama",
            visible=False
        )

    enable_cost_saver.change(lambda x: gr.update(visible=x), inputs=[enable_cost_saver], outputs=[cheap_container])

    # --- Planner LLM ---
    planner_comps, _ = create_llm_settings_ui(
        prefix="planner_llm",
        label="Planner LLM",
        default_provider="ollama",
        include_vision=True,
        default_vision=False,
        include_ctx=True
    )

    with gr.Row():
        enable_hierarchical_planning = gr.Checkbox(
            label="Enable Hierarchical Planning",
            value=True,
            info="Uses the Planner LLM (or Main LLM) to break down tasks into a step-by-step plan before execution.",
            interactive=True
        )
        
        enable_auto_pause = gr.Checkbox(
            label="Auto-Pause on Plan Failure",
            value=True,
            info="Automatically pause the agent if a plan step is marked as failed, allowing user intervention.",
            interactive=True
        )
        
        enable_kb_auto_save = gr.Checkbox(
            label="Auto-save Generated Knowledge",
            value=True,
            info="Automatically save useful knowledge extracted from the session to the knowledge base without asking.",
            interactive=True
        )

    # --- Planning Templates ---
    planner_template_comps = create_template_manager_ui("📝 Planning Strategy Templates", default_value=DEFAULT_PLANNER_PROMPT)
    (planner_template_dropdown, planner_refresh_btn, planner_load_btn, 
     planner_new_name, planner_save_btn, planner_system_prompt) = planner_template_comps
    
    # Wire up logic for templates
    setup_template_manager_logic("./tmp/planning_templates", *planner_template_comps)

    with gr.Row():
        task_complexity = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Task Complexity Preset",
            info="Automatically adjusts Max Steps and Actions (1=Simple, 10=Complex)",
            interactive=True
        )

    with gr.Row():
        max_steps = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Max Run Steps",
            info="Maximum number of steps the agent will take",
            interactive=True
        )
        max_actions = gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            step=1,
            label="Max Number of Actions",
            info="Maximum number of actions the agent will take per step",
            interactive=True
        )

        fast_mode = gr.Checkbox(
            label="Fast Mode",
            value=True,
            info="Reduces delays for faster execution.",
            interactive=True
        )

        enable_memory = gr.Checkbox(
            label="Enable Memory",
            value=False,
            info="Enable the agent's memory capabilities to retain context across steps.",
            interactive=True
        )

        auto_save_on_stuck = gr.Checkbox(
            label="Auto-Save on Stuck",
            value=True,
            info="Automatically save page content to a file if the agent fails repeatedly.",
            interactive=True
        )

        require_confirmation = gr.Checkbox(
            label="Require Confirmation (Human in the Loop)",
            value=False,
            info="Pause and ask for permission before executing critical actions (click, type, navigate).",
            interactive=True
        )

        disable_hud = gr.Checkbox(
            label="Disable HUD",
            value=False,
            info="Disable the in-browser Heads Up Display overlay.",
            interactive=True
        )
        
        restrict_to_knowledge_base = gr.Checkbox(
            label="Restrict to Knowledge Base",
            value=False,
            info="If enabled, the agent will be instructed to use ONLY provided knowledge base files and NOT search the internet for information.",
            interactive=True
        )

    with gr.Row():
        max_input_tokens = gr.Number(
            label="Max Input Tokens",
            value=128000,
            precision=0,
            interactive=True
        )
        tool_calling_method = gr.Dropdown(
            label="Tool Calling Method",
            value="auto",
            interactive=True,
            allow_custom_value=True,
            choices=['function_calling', 'json_mode', 'raw', 'auto', 'tools', "None"],
            visible=True
        )
    
    # Update tab_components with explicitly created components
    tab_components.update(dict(
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        enable_hierarchical_planning=enable_hierarchical_planning,
        enable_auto_pause=enable_auto_pause,
        enable_kb_auto_save=enable_kb_auto_save,
        planner_system_prompt=planner_system_prompt,
        enable_smart_retry=enable_smart_retry,
        enable_cost_saver=enable_cost_saver,
        max_steps=max_steps,
        task_complexity=task_complexity,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        fast_mode=fast_mode,
        enable_memory=enable_memory,
        auto_save_on_stuck=auto_save_on_stuck,
        require_confirmation=require_confirmation,
        disable_hud=disable_hud,
        restrict_to_knowledge_base=restrict_to_knowledge_base,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    
    # Add generated components to tab_components
    tab_components.update(llm_comps)
    tab_components.update(confirmer_comps)
    tab_components.update(smart_retry_comps)
    tab_components.update(cheap_comps)
    tab_components.update(planner_comps)

    webui_manager.add_components("agent_settings", tab_components)

    def update_complexity(complexity):
        steps = int(complexity * 20)
        actions = int(complexity * 2)
        return gr.update(value=steps), gr.update(value=actions)

    task_complexity.change(fn=update_complexity, inputs=[task_complexity], outputs=[max_steps, max_actions])

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        async for update in safe_execution(update_mcp_server, mcp_file, webui_manager):
            yield update

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )
    return tab_components
