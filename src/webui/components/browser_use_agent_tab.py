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
    handle_resume_session, handle_delete_session,
    handle_move_step_up, handle_move_step_down, handle_delete_step_from_plan,
    handle_step_selection_change, handle_save_step_text, handle_add_step,
    handle_clear_completed_steps, handle_save_workflow, handle_load_workflow, handle_run_step_action, 
    handle_set_step_action, handle_duplicate_step, handle_streamline_plan, handle_generate_plan_from_prompt, 
    handle_refresh_workflows, handle_delete_workflow, handle_export_plan, handle_complete_current_step
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
                with gr.Row():
                    update_plan_btn = gr.Button("Update Plan", variant="secondary", scale=1)
                    plan_export_btn = gr.Button("📤 Export Plan", scale=1)
                plan_export_file = gr.File(label="Download Plan", visible=False, interactive=False)
                
                with gr.Group():
                    gr.Markdown("#### Generate Plan")
                    with gr.Row():
                        plan_prompt_input = gr.Textbox(label="Task Prompt", placeholder="Describe the task to generate a plan for...", scale=3)
                        plan_generate_btn = gr.Button("✨ Generate Plan", variant="secondary", scale=1)
                
                with gr.Group():
                    gr.Markdown("#### Manage Steps")
                    with gr.Row():
                        plan_step_selector = gr.Dropdown(label="Select Step to Move/Delete/Edit", choices=[], interactive=True, scale=3)
                        plan_move_up_btn = gr.Button("⬆️ Up", scale=1)
                        plan_move_down_btn = gr.Button("⬇️ Down", scale=1)
                        plan_duplicate_step_btn = gr.Button("📋 Duplicate", scale=1)
                        plan_delete_step_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        plan_run_action_btn = gr.Button("▶️ Run Action", variant="secondary", scale=1)
                        plan_complete_step_btn = gr.Button("✅ Complete Current", variant="secondary", scale=1)
                        plan_clear_completed_btn = gr.Button("🧹 Clear Done", variant="secondary", scale=1)
                    
                    with gr.Row():
                        plan_step_text = gr.Textbox(label="Edit Step Description", interactive=True, scale=3)
                        plan_add_step_btn = gr.Button("➕ Add Step", variant="secondary", scale=1)
                        plan_save_step_btn = gr.Button("💾 Update Step", variant="primary", scale=1)
                    
                    with gr.Row():
                        plan_action_name = gr.Dropdown(
                            label="Action Name", 
                            choices=["go_to_url", "click_element_by_text", "type_into_element_by_text", "scroll_down", "extract_page_links", "wait_for_content", "google_search"], 
                            allow_custom_value=True, 
                            interactive=True, 
                            scale=2
                        )
                        plan_action_params = gr.Textbox(label="Params (JSON)", placeholder='{"url": "..."}', interactive=True, scale=3)
                        plan_set_action_btn = gr.Button("⚙️ Set Action", variant="secondary", scale=1)
                    
                    plan_action_result = gr.Textbox(label="Last Action Result", interactive=False, lines=3)
                
                with gr.Group():
                    gr.Markdown("#### ⚡ Workflows & Optimization")
                    with gr.Row():
                        workflow_filter = gr.Textbox(label="Filter", placeholder="Search...", scale=1)
                        workflow_selector = gr.Dropdown(label="Load Workflow", choices=webui_manager.workflow_manager.list_workflows(), interactive=True, scale=2)
                        workflow_refresh_btn = gr.Button("🔄", scale=0)
                        workflow_load_btn = gr.Button("📂 Load", scale=1)
                        workflow_delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=0)
                        workflow_name_input = gr.Textbox(label="Save as Workflow", placeholder="folder/name (e.g. research/news_scraper)...", scale=2)
                        workflow_save_btn = gr.Button("💾 Save", scale=1)
                    plan_streamline_btn = gr.Button("✨ Streamline Plan (Optimize with AI)", variant="primary")

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
                    allow_custom_value=False
                )
                refresh_extraction_btn = gr.Button("🔄 Refresh Models", variant="secondary", scale=0)
            
            with gr.Accordion("📂 Session Management", open=False):
                with gr.Row():
                    history_filter = gr.Textbox(label="Filter Sessions", placeholder="Date or ID...", scale=2)
                    history_files_dropdown = gr.Dropdown(label="Saved Sessions", choices=[], interactive=True, scale=3, allow_custom_value=True)
                    refresh_history_btn = gr.Button("🔄 Scan", variant="secondary", scale=1)
                    resume_session_btn = gr.Button("📂 Resume Session", variant="secondary", scale=1)
                    delete_session_btn = gr.Button("🗑️ Delete", variant="stop", scale=0)

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
            plan_export_btn=plan_export_btn,
            plan_export_file=plan_export_file,
            plan_prompt_input=plan_prompt_input,
            plan_generate_btn=plan_generate_btn,
            plan_editor_accordion=plan_editor_accordion,
            plan_step_selector=plan_step_selector,
            plan_move_up_btn=plan_move_up_btn,
            plan_move_down_btn=plan_move_down_btn,
            plan_duplicate_step_btn=plan_duplicate_step_btn,
            plan_delete_step_btn=plan_delete_step_btn,
            plan_run_action_btn=plan_run_action_btn,
            plan_complete_step_btn=plan_complete_step_btn,
            plan_clear_completed_btn=plan_clear_completed_btn,
            plan_step_text=plan_step_text,
            plan_add_step_btn=plan_add_step_btn,
            plan_save_step_btn=plan_save_step_btn,
            plan_action_name=plan_action_name,
            plan_action_params=plan_action_params,
            plan_set_action_btn=plan_set_action_btn,
            plan_action_result=plan_action_result,
            workflow_filter=workflow_filter,
            workflow_selector=workflow_selector,
            workflow_refresh_btn=workflow_refresh_btn,
            workflow_load_btn=workflow_load_btn,
            workflow_delete_btn=workflow_delete_btn,
            workflow_name_input=workflow_name_input,
            workflow_save_btn=workflow_save_btn,
            plan_streamline_btn=plan_streamline_btn,
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
            history_filter=history_filter,
            history_files_dropdown=history_files_dropdown,
            refresh_history_btn=refresh_history_btn,
            resume_session_btn=resume_session_btn,
            delete_session_btn=delete_session_btn,
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

    async def delete_session_wrapper(history_file: str, *args) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_delete_session."""
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_delete_session, webui_manager, history_file, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def update_plan_wrapper(new_plan_json: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_update_plan."""
        async for update in safe_execution(handle_update_plan, webui_manager, new_plan_json):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def move_up_wrapper(selected_step: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_move_step_up, webui_manager, selected_step):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def move_down_wrapper(selected_step: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_move_step_down, webui_manager, selected_step):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def duplicate_step_wrapper(selected_step: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_duplicate_step, webui_manager, selected_step):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def delete_step_wrapper(selected_step: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_delete_step_from_plan, webui_manager, selected_step):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def step_selection_change_wrapper(selected_step: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_step_selection_change, webui_manager, selected_step):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def save_step_text_wrapper(selected_step: str, new_text: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_save_step_text, webui_manager, selected_step, new_text):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def add_step_wrapper(new_text: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_add_step, webui_manager, new_text):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def clear_completed_wrapper() -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_clear_completed_steps, webui_manager):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def run_step_action_wrapper(selected_step: str, *args) -> AsyncGenerator[List[Any], None]:
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_run_step_action, webui_manager, selected_step, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def complete_step_wrapper(*args) -> AsyncGenerator[List[Any], None]:
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_complete_current_step, webui_manager, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def set_step_action_wrapper(selected_step: str, action_name: str, params: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_set_step_action, webui_manager, selected_step, action_name, params):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def generate_plan_wrapper(prompt: str, *args) -> AsyncGenerator[List[Any], None]:
        components_dict = dict(zip(input_components, args))
        async for update in safe_execution(handle_generate_plan_from_prompt, webui_manager, prompt, components_dict):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def export_plan_wrapper() -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_export_plan, webui_manager):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def refresh_workflows_wrapper(filter_val: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_refresh_workflows, webui_manager, filter_val):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def delete_workflow_wrapper(name: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_delete_workflow, webui_manager, name):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def save_workflow_wrapper(name: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_save_workflow, webui_manager, name):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def load_workflow_wrapper(name: str) -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_load_workflow, webui_manager, name):
            yield map_dict_to_gradio_outputs(update, run_tab_outputs)

    async def streamline_plan_wrapper() -> AsyncGenerator[List[Any], None]:
        async for update in safe_execution(handle_streamline_plan, webui_manager):
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
    plan_export_btn.click(fn=export_plan_wrapper, inputs=None, outputs=run_tab_outputs)
    plan_generate_btn.click(fn=generate_plan_wrapper, inputs=[plan_prompt_input] + input_components, outputs=run_tab_outputs)
    plan_move_up_btn.click(fn=move_up_wrapper, inputs=[plan_step_selector], outputs=run_tab_outputs)
    plan_move_down_btn.click(fn=move_down_wrapper, inputs=[plan_step_selector], outputs=run_tab_outputs)
    plan_delete_step_btn.click(fn=delete_step_wrapper, inputs=[plan_step_selector], outputs=run_tab_outputs)
    plan_duplicate_step_btn.click(fn=duplicate_step_wrapper, inputs=[plan_step_selector], outputs=run_tab_outputs)
    plan_run_action_btn.click(fn=run_step_action_wrapper, inputs=[plan_step_selector] + input_components, outputs=run_tab_outputs)
    plan_complete_step_btn.click(fn=complete_step_wrapper, inputs=input_components, outputs=run_tab_outputs)
    plan_clear_completed_btn.click(fn=clear_completed_wrapper, inputs=None, outputs=run_tab_outputs)
    plan_step_selector.change(fn=step_selection_change_wrapper, inputs=[plan_step_selector], outputs=run_tab_outputs)
    plan_save_step_btn.click(fn=save_step_text_wrapper, inputs=[plan_step_selector, plan_step_text], outputs=run_tab_outputs)
    plan_add_step_btn.click(fn=add_step_wrapper, inputs=[plan_step_text], outputs=run_tab_outputs)
    plan_set_action_btn.click(fn=set_step_action_wrapper, inputs=[plan_step_selector, plan_action_name, plan_action_params], outputs=run_tab_outputs)
    workflow_refresh_btn.click(fn=refresh_workflows_wrapper, inputs=[workflow_filter], outputs=run_tab_outputs)
    workflow_filter.change(fn=refresh_workflows_wrapper, inputs=[workflow_filter], outputs=run_tab_outputs)
    workflow_delete_btn.click(fn=delete_workflow_wrapper, inputs=[workflow_selector], outputs=run_tab_outputs)
    workflow_save_btn.click(fn=save_workflow_wrapper, inputs=[workflow_name_input], outputs=run_tab_outputs)
    workflow_load_btn.click(fn=load_workflow_wrapper, inputs=[workflow_selector], outputs=run_tab_outputs)
    plan_streamline_btn.click(fn=streamline_plan_wrapper, inputs=None, outputs=run_tab_outputs)
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
            
            filter_val = components_dict.get(history_filter, "")
            update = await handle_refresh_history_files(save_path, filter_val)
            yield {history_files_dropdown: update}
        async for update in safe_execution(_logic):
            yield map_dict_to_gradio_outputs(update, [history_files_dropdown])

    refresh_history_btn.click(fn=refresh_history_click_handler, inputs=input_components, outputs=[history_files_dropdown])
    history_filter.submit(fn=refresh_history_click_handler, inputs=input_components, outputs=[history_files_dropdown])
    resume_session_btn.click(fn=resume_session_wrapper, inputs=[history_files_dropdown] + input_components, outputs=run_tab_outputs)
    delete_session_btn.click(fn=delete_session_wrapper, inputs=[history_files_dropdown] + input_components, outputs=run_tab_outputs)

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