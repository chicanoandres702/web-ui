import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr

from gradio.components import Component

from src.webui.components.shared import create_agent_control_buttons, safe_execution, get_valid_input_components, map_dict_to_gradio_outputs
from src.webui.webui_manager import WebuiManager

from src.utils.prompt_library import get_all_prompts, save_custom_prompt, delete_custom_prompt, get_prompt_by_name
from src.webui.components.browser_use_agent_handlers import (
    handle_submit, handle_retry, handle_stop, handle_pause_resume, handle_clear,
    handle_save_chat, handle_reload_memory, handle_update_kb_list, handle_load_kb_file,
    handle_refresh_history_files, handle_update_plan, handle_save_generated_kb,
    handle_resume_session
)
from src.utils.utils import ensure_default_extraction_models

logger = logging.getLogger(__name__)

def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """
    Create the run agent tab, defining UI, state, and handlers.
    """
    webui_manager.init_browser_use_agent()
    ensure_default_extraction_models()

    # --- Define UI Components ---
    tab_components = {}
    
    with gr.Row(equal_height=False):
        # Left Column: Chat & Controls
        with gr.Column(scale=3):
            agent_status = gr.Markdown(value="### Agent Status\nReady", label="Agent Status")
            plan_status = gr.Markdown(value="", visible=False, label="Current Plan")
            
            with gr.Accordion("📝 Edit Plan", open=False, visible=False) as plan_editor_accordion:
                plan_editor = gr.Code(language="json", label="Plan JSON", interactive=True)
                update_plan_btn = gr.Button("Update Plan", variant="secondary")

            with gr.Accordion("🧠 Knowledge Generation Proposal", open=False, visible=False) as kb_gen_accordion:
                kb_gen_title = gr.Textbox(label="Suggested Title")
                kb_gen_content = gr.Code(label="Suggested Content", language="markdown", interactive=True)
                with gr.Row():
                    kb_gen_save_btn = gr.Button("💾 Save to KB", variant="primary")
                    kb_gen_discard_btn = gr.Button("🗑️ Discard", variant="stop")

            chatbot = gr.Chatbot(
                lambda: webui_manager.bu_chat_history,  # Load history dynamically
                elem_id="browser_use_chatbot",
                label="Agent Interaction",
                type="messages",
                height=600,
                allow_tags=True,
            )
            user_input = gr.Textbox(
                label="Your Task or Response",
                placeholder="Enter your task here or provide assistance when asked.",
                lines=3,
                interactive=True,
                elem_id="user_input",
            )
            
            with gr.Accordion("📚 Prompt Library", open=False):
                with gr.Row():
                    prompt_selector = gr.Dropdown(
                        label="Select a Prompt", 
                        choices=[p[0] for p in get_all_prompts()],
                        interactive=True,
                        scale=3
                    )
                    refresh_prompts_btn = gr.Button("🔄", scale=0)
                    delete_prompt_btn = gr.Button("🗑️", scale=0, variant="stop")
                with gr.Accordion("➕ Add Custom Prompt", open=False):
                    with gr.Row():
                        new_prompt_name = gr.Textbox(label="Name", placeholder="My Custom Task", scale=3)
                        save_prompt_btn = gr.Button("💾 Save Current Task", scale=1)

            with gr.Row():
                stop_button, pause_resume_button, clear_button = create_agent_control_buttons("browser_use_agent")
                save_chat_button = gr.Button(
                    "💾 Save Chat", interactive=True, variant="secondary", scale=1
                )
                retry_button = gr.Button(
                    "🔄 Retry", interactive=True, variant="secondary", scale=1
                )
                run_button = gr.Button("▶️ Submit Task", variant="primary", scale=2)
            
            with gr.Accordion("🧠 Brain & Memory", open=False):
                with gr.Row():
                    brain_file = gr.Textbox(label="Brain File (System Prompt)", value="./brain.md", placeholder="Path to markdown file", scale=2)
                    memory_file = gr.Textbox(label="Memory File (Read/Write)", value="./tmp/memory/memory.txt", placeholder="Path to text file", scale=2)
                    reload_memory_btn = gr.Button("📂 View Memory", variant="secondary", scale=0)
                with gr.Row():
                    kb_files_dropdown = gr.Dropdown(label="Knowledge Base Files", choices=[], value=None, interactive=True, scale=3, allow_custom_value=True)
                    refresh_kb_btn = gr.Button("🔄 Scan Files", variant="secondary", scale=1)
                    load_kb_btn = gr.Button("📂 Load Content", variant="secondary", scale=1)
                memory_content = gr.TextArea(label="Memory Content", interactive=False, lines=10)
            
            with gr.Accordion("📊 Extraction Model", open=False):
                extraction_choices = []
                ext_path = "./tmp/extraction_models"
                if os.path.exists(ext_path):
                    extraction_choices = sorted([f.replace(".json", "") for f in os.listdir(ext_path) if f.endswith(".json")])

                extraction_model_dropdown = gr.Dropdown(
                    label="Select Extraction Model",
                    choices=extraction_choices,
                    value=None,
                    interactive=True,
                    allow_custom_value=True
                )
                refresh_extraction_btn = gr.Button("🔄 Refresh Models", variant="secondary", scale=0)
            
            with gr.Accordion("📂 Session Management", open=False):
                with gr.Row():
                    history_files_dropdown = gr.Dropdown(label="Saved Sessions", choices=[], interactive=True, scale=3, allow_custom_value=True)
                    refresh_history_btn = gr.Button("🔄 Scan", variant="secondary", scale=1)
                    resume_session_btn = gr.Button("📂 Resume Session", variant="secondary", scale=1)

        # Right Column: Browser View & Outputs
        with gr.Column(scale=2):
            browser_view = gr.HTML(
                value="<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",
                label="Browser Live View",
                elem_id="browser_view",
                visible=False,
                padding=False,
            )
            with gr.Group():
                gr.Markdown("### Task Outputs")
                recording_gif = gr.Image(
                    label="Task Recording GIF",
                    format="gif",
                    interactive=False,
                    type="filepath",
                )
                agent_history_file = gr.File(label="Agent History JSON", interactive=False)
                chat_log_file = gr.File(label="Chat Log JSON", interactive=False)

    # --- Store Components in Manager ---
    tab_components.update(
        dict(
            chatbot=chatbot,
            agent_status=agent_status,
            plan_status=plan_status,
            plan_editor=plan_editor,
            plan_editor_accordion=plan_editor_accordion,
            kb_gen_accordion=kb_gen_accordion,
            kb_gen_title=kb_gen_title,
            kb_gen_content=kb_gen_content,
            kb_gen_save_btn=kb_gen_save_btn,
            user_input=user_input,
            clear_button=clear_button,
            run_button=run_button,
            retry_button=retry_button,
            save_chat_button=save_chat_button,
            stop_button=stop_button,
            pause_resume_button=pause_resume_button,
            brain_file=brain_file,
            memory_file=memory_file,
            kb_files_dropdown=kb_files_dropdown,
            refresh_kb_btn=refresh_kb_btn,
            load_kb_btn=load_kb_btn,
            reload_memory_btn=reload_memory_btn,
            memory_content=memory_content,
            extraction_model_dropdown=extraction_model_dropdown,
            refresh_extraction_btn=refresh_extraction_btn,
            agent_history_file=agent_history_file,
            chat_log_file=chat_log_file,
            recording_gif=recording_gif,
            browser_view=browser_view,
            history_files_dropdown=history_files_dropdown,
            refresh_history_btn=refresh_history_btn,
            resume_session_btn=resume_session_btn,
        )
    )
    webui_manager.add_components(
        "browser_use_agent", tab_components
    )  # Use "browser_use_agent" as tab_name prefix

    all_managed_components = list(
        webui_manager.get_components()
    )  # Get all components known to manager
    
    # Filter out layout components that cannot be used as inputs
    input_components = get_valid_input_components(all_managed_components)
    
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(
            *args
    ) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_submit that yields its results."""
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_submit, webui_manager, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def retry_wrapper(
            *args
    ) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_retry."""
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_retry, webui_manager, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def stop_wrapper():
        """Wrapper for handle_stop."""
        await handle_stop(webui_manager)
        # No yield, side effect only

    async def pause_resume_wrapper():
        """Wrapper for handle_pause_resume."""
        await handle_pause_resume(webui_manager)
        # No yield, side effect only

    async def clear_wrapper() -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_clear."""
        async for update in safe_execution(handle_clear, webui_manager):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def save_chat_wrapper() -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_save_chat."""
        async for update in safe_execution(handle_save_chat, webui_manager):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def reload_memory_wrapper(path: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_reload_memory."""
        async for update in safe_execution(handle_reload_memory, webui_manager, path):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def update_kb_list_wrapper(path: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_update_kb_list."""
        async def _logic():
            update = await handle_update_kb_list(path)
            yield {kb_files_dropdown: update}
        async for update in safe_execution(_logic):
            yield map_dict_to_gradio_outputs(update, [kb_files_dropdown])

    async def load_kb_file_wrapper(path: str, filename: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_load_kb_file."""
        async for update in safe_execution(handle_load_kb_file, webui_manager, path, filename):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def resume_session_wrapper(history_file: str, *args) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_resume_session."""
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_resume_session, webui_manager, history_file, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def update_plan_wrapper(new_plan_json: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_update_plan."""
        async for update in safe_execution(handle_update_plan, webui_manager, new_plan_json):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def save_kb_wrapper(title: str, content: str, mem_path: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_save_generated_kb."""
        async for update in safe_execution(handle_save_generated_kb, webui_manager, title, content, mem_path):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def refresh_extraction_models_wrapper():
        models = []
        path = "./tmp/extraction_models"
        if os.path.exists(path):
             models = [f.replace(".json", "") for f in os.listdir(path) if f.endswith(".json")]
        return gr.update(choices=sorted(models))

    # --- Connect Event Handlers using the Wrappers --
    run_button.click(
        fn=submit_wrapper, inputs=input_components, outputs=run_tab_outputs, trigger_mode="multiple"
    )
    user_input.submit(
        fn=submit_wrapper, inputs=input_components, outputs=run_tab_outputs, trigger_mode="multiple"
    )
    retry_button.click(fn=retry_wrapper, inputs=input_components, outputs=run_tab_outputs)
    stop_button.click(fn=stop_wrapper, inputs=None, outputs=[])
    pause_resume_button.click(fn=pause_resume_wrapper, inputs=None, outputs=[])
    save_chat_button.click(fn=save_chat_wrapper, inputs=None, outputs=run_tab_outputs)
    clear_button.click(fn=clear_wrapper, inputs=None, outputs=run_tab_outputs)
    refresh_kb_btn.click(fn=update_kb_list_wrapper, inputs=[memory_file], outputs=[kb_files_dropdown])
    load_kb_btn.click(fn=load_kb_file_wrapper, inputs=[memory_file, kb_files_dropdown], outputs=run_tab_outputs)
    reload_memory_btn.click(fn=reload_memory_wrapper, inputs=[memory_file], outputs=run_tab_outputs)
    update_plan_btn.click(fn=update_plan_wrapper, inputs=[plan_editor], outputs=run_tab_outputs)
    kb_gen_save_btn.click(fn=save_kb_wrapper, inputs=[kb_gen_title, kb_gen_content, memory_file], outputs=run_tab_outputs)
    kb_gen_discard_btn.click(fn=lambda: {kb_gen_accordion: gr.update(visible=False, open=False)}, outputs=[kb_gen_accordion])
    refresh_extraction_btn.click(fn=refresh_extraction_models_wrapper, outputs=[extraction_model_dropdown])
    
    # Re-binding refresh_history_btn to use a function that looks up the path
    async def refresh_history_click_handler(*args):
        async def _logic():
            # Find the save path from components
            components_dict = dict(zip(input_components, args))
            # We need to find the component ID "browser_settings.save_agent_history_path"
            save_path_comp = webui_manager.id_to_component.get("browser_settings.save_agent_history_path")
            save_path = components_dict.get(save_path_comp) or "./tmp/agent_history"
            update = await handle_refresh_history_files(save_path)
            yield {history_files_dropdown: update}
        async for update in safe_execution(_logic):
            yield map_dict_to_gradio_outputs(update, [history_files_dropdown])

    refresh_history_btn.click(fn=refresh_history_click_handler, inputs=input_components, outputs=[history_files_dropdown])
    resume_session_btn.click(fn=resume_session_wrapper, inputs=[history_files_dropdown] + input_components, outputs=run_tab_outputs)

    def refresh_prompts_handler():
        return gr.update(choices=[p[0] for p in get_all_prompts()], value=None)

    def save_prompt_handler(name, content):
        if not name or not content:
            gr.Warning("Name and content are required.")
            return gr.update()
        if save_custom_prompt(name, content):
            gr.Info(f"Saved prompt: {name}")
            return refresh_prompts_handler()
        else:
            gr.Error("Failed to save prompt.")
            return gr.update()

    def delete_prompt_handler(prompt_name):
        if not prompt_name:
            return gr.update()
        
        if delete_custom_prompt(prompt_name):
            gr.Info(f"Deleted prompt: {prompt_name}")
            return refresh_prompts_handler()
        else:
            if not prompt_name.startswith("[Custom] "):
                gr.Warning("Cannot delete built-in prompts.")
            else:
                gr.Error("Failed to delete prompt.")
            return gr.update()

    def prompt_selection_handler(prompt_name):
        if not prompt_name:
            return gr.update()
        content = get_prompt_by_name(prompt_name)
        if content:
            return content
        return gr.update()

    refresh_prompts_btn.click(fn=refresh_prompts_handler, outputs=[prompt_selector])
    save_prompt_btn.click(fn=save_prompt_handler, inputs=[new_prompt_name, user_input], outputs=[prompt_selector])
    delete_prompt_btn.click(fn=delete_prompt_handler, inputs=[prompt_selector], outputs=[prompt_selector])

    prompt_selector.change(
        fn=prompt_selection_handler,
        inputs=[prompt_selector],
        outputs=[user_input]
    )