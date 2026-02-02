import json
import os

import gradio as gr
from gradio.components import Component
from typing import Any, Dict, Optional
from src.webui.webui_manager import WebuiManager
from src.utils import config
import logging
from functools import partial

logger = logging.getLogger(__name__)


def update_model_dropdown(llm_provider):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use predefined models for the selected provider
    if llm_provider in config.model_names:
        return gr.Dropdown(choices=config.model_names[llm_provider], value=config.model_names[llm_provider][0],
                           interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server.
    """
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    if not mcp_file:
        return None, gr.update(visible=False)

    if not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)


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

    with gr.Group():
        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="LLM Provider",
                value=os.getenv("DEFAULT_LLM", "ollama"),
                info="Select LLM provider for LLM",
                interactive=True
            )
            llm_model_name = gr.Dropdown(
                label="LLM Model Name",
                choices=config.model_names[os.getenv("DEFAULT_LLM", "ollama")],
                value=config.model_names[os.getenv("DEFAULT_LLM", "ollama")][0],
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            use_vision = gr.Checkbox(
                label="Use Vision",
                value=True,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=True,
                interactive=True
            )

        with gr.Row():
            llm_base_url = gr.Textbox(
                label="Base URL",
                value="http://137.131.201.189:11434",
                info="API endpoint URL (if required)"
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            confirmer_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Confirmer LLM Provider",
                info="Select LLM provider for Confirmer (Optional)",
                value=None,
                interactive=True
            )
            confirmer_llm_model_name = gr.Dropdown(
                label="Confirmer LLM Model Name",
                choices=config.model_names["ollama"],
                value=config.model_names["ollama"][0],
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            confirmer_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="Confirmer LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            confirmer_use_vision = gr.Checkbox(
                label="Use Vision(Confirmer LLM)",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            confirmer_ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )
            
            confirmer_strictness = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Confirmer Strictness",
                info="Control how strictly the confirmer validates actions (1=Lenient, 10=Strict)",
                interactive=True
            )

        with gr.Row():
            confirmer_llm_base_url = gr.Textbox(
                label="Base URL",
                value="http://137.131.201.189:11434",
                info="API endpoint URL (if required)"
            )
            confirmer_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            gr.Markdown("### 🧠 Smart Retry Settings")
        with gr.Row():
            enable_smart_retry = gr.Checkbox(
                label="Enable Smart Retry",
                value=False,
                info="Automatically switch to a stronger model after 2 consecutive failures.",
                interactive=True
            )
        
        with gr.Row(visible=False) as smart_retry_row:
            smart_retry_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Retry Provider",
                value=None,
                interactive=True
            )
            smart_retry_llm_model_name = gr.Dropdown(
                label="Retry Model",
                choices=config.model_names["ollama"],
                value=config.model_names["ollama"][0],
                interactive=True,
                allow_custom_value=True
            )
            smart_retry_llm_temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.6, step=0.1, label="Temperature", interactive=True
            )
            smart_retry_llm_base_url = gr.Textbox(
                label="Base URL",
                value="http://137.131.201.189:11434",
                info="API endpoint URL (if required)"
            )
            smart_retry_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    enable_smart_retry.change(lambda x: gr.update(visible=x), inputs=[enable_smart_retry], outputs=[smart_retry_row])

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
        
        with gr.Row(visible=False) as cost_saver_row:
            cheap_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Cheap Provider",
                value=None,
                interactive=True
            )
            cheap_llm_model_name = gr.Dropdown(
                label="Cheap Model",
                choices=config.model_names["ollama"],
                value=config.model_names["ollama"][0],
                interactive=True,
                allow_custom_value=True
            )
            cheap_llm_temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.6, step=0.1, label="Temperature", interactive=True
            )
            cheap_llm_base_url = gr.Textbox(
                label="Base URL",
                value="http://137.131.201.189:11434",
                info="API endpoint URL (if required)"
            )
            cheap_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    enable_cost_saver.change(lambda x: gr.update(visible=x), inputs=[enable_cost_saver], outputs=[cost_saver_row])

    with gr.Group():
        with gr.Row():
            planner_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Planner LLM Provider",
                info="Select LLM provider for Planner (Optional)",
                value=None,
                interactive=True
            )
            planner_llm_model_name = gr.Dropdown(
                label="Planner LLM Model Name",
                choices=config.model_names["ollama"],
                value=config.model_names["ollama"][0],
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            planner_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="Planner LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            planner_use_vision = gr.Checkbox(
                label="Use Vision(Planner LLM)",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            planner_ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            planner_llm_base_url = gr.Textbox(
                label="Base URL",
                value="http://137.131.201.189:11434",
                info="API endpoint URL (if required)"
            )
            planner_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

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
            value=False,
            info="Reduces delays and lowers strictness for faster execution.",
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
    tab_components.update(dict(
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        use_vision=use_vision,
        ollama_num_ctx=ollama_num_ctx,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        planner_llm_provider=planner_llm_provider,
        planner_llm_model_name=planner_llm_model_name,
        planner_llm_temperature=planner_llm_temperature,
        planner_use_vision=planner_use_vision,
        planner_ollama_num_ctx=planner_ollama_num_ctx,
        planner_llm_base_url=planner_llm_base_url,
        planner_llm_api_key=planner_llm_api_key,
        confirmer_llm_provider=confirmer_llm_provider,
        confirmer_llm_model_name=confirmer_llm_model_name,
        confirmer_llm_temperature=confirmer_llm_temperature,
        confirmer_use_vision=confirmer_use_vision,
        confirmer_ollama_num_ctx=confirmer_ollama_num_ctx,
        confirmer_strictness=confirmer_strictness,
        confirmer_llm_base_url=confirmer_llm_base_url,
        confirmer_llm_api_key=confirmer_llm_api_key,
        enable_smart_retry=enable_smart_retry,
        smart_retry_llm_provider=smart_retry_llm_provider,
        smart_retry_llm_model_name=smart_retry_llm_model_name,
        smart_retry_llm_temperature=smart_retry_llm_temperature,
        smart_retry_llm_base_url=smart_retry_llm_base_url,
        smart_retry_llm_api_key=smart_retry_llm_api_key,
        enable_cost_saver=enable_cost_saver,
        cheap_llm_provider=cheap_llm_provider,
        cheap_llm_model_name=cheap_llm_model_name,
        cheap_llm_temperature=cheap_llm_temperature,
        cheap_llm_base_url=cheap_llm_base_url,
        cheap_llm_api_key=cheap_llm_api_key,
        max_steps=max_steps,
        task_complexity=task_complexity,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        fast_mode=fast_mode,
        auto_save_on_stuck=auto_save_on_stuck,
        require_confirmation=require_confirmation,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    webui_manager.add_components("agent_settings", tab_components)

    llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=llm_provider,
        outputs=ollama_num_ctx
    )
    llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[llm_provider],
        outputs=[llm_model_name]
    )
    planner_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[planner_llm_provider],
        outputs=[planner_ollama_num_ctx]
    )
    planner_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[planner_llm_provider],
        outputs=[planner_llm_model_name]
    )
    confirmer_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[confirmer_llm_provider],
        outputs=[confirmer_ollama_num_ctx]
    )
    confirmer_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[confirmer_llm_provider],
        outputs=[confirmer_llm_model_name]
    )
    smart_retry_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[smart_retry_llm_provider],
        outputs=[smart_retry_llm_model_name]
    )
    smart_retry_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[smart_retry_llm_provider],
        outputs=[ollama_num_ctx] # Reusing main ollama ctx for simplicity or create new one if needed
    )
    cheap_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[cheap_llm_provider],
        outputs=[cheap_llm_model_name]
    )
    cheap_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[cheap_llm_provider],
        outputs=[ollama_num_ctx]
    )

    def update_complexity(complexity):
        steps = int(complexity * 20)
        actions = int(complexity * 2)
        return gr.update(value=steps), gr.update(value=actions)

    task_complexity.change(fn=update_complexity, inputs=[task_complexity], outputs=[max_steps, max_actions])

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        update_dict = await update_mcp_server(mcp_file, webui_manager)
        yield update_dict

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )
