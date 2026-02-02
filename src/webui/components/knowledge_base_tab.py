import gradio as gr
import os
from datetime import datetime
from src.webui.webui_manager import WebuiManager
from src.webui.components.knowledge_base_logic import (
    list_kb_files,
    search_kb_files,
    load_kb_content,
    save_kb_content,
    create_kb_file,
    delete_kb_file,
    rename_kb_file,
    import_kb_files
)
import logging

logger = logging.getLogger(__name__)

def create_knowledge_base_tab(webui_manager: WebuiManager):
    """
    Creates the Knowledge Base management tab.
    """
    with gr.Row():
        kb_directory = gr.Textbox(
            label="Knowledge Base Directory", 
            value="./tmp/memory",
            placeholder="Path to directory containing KB files",
            interactive=True
        )
        search_query = gr.Textbox(label="Search", placeholder="Search content...", scale=2)
        search_btn = gr.Button("🔍", scale=0)
        refresh_btn = gr.Button("🔄 Refresh List", scale=0)

    with gr.Row():
        with gr.Column(scale=1):
            file_list = gr.Dropdown(label="Files", choices=[], interactive=True, allow_custom_value=False)
            with gr.Group():
                new_filename = gr.Textbox(label="New File Name", placeholder="e.g. folder/topic.md")
                create_btn = gr.Button("➕ Create New", variant="secondary")
            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
            
            with gr.Group():
                rename_target = gr.Textbox(label="Rename Selected To", placeholder="e.g. new_folder/new_name")
                upload_files = gr.File(label="Bulk Import (Markdown)", file_count="multiple", file_types=[".md", ".txt"])
                import_btn = gr.Button("⬆️ Import Files", variant="secondary")
        
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Edit"):
                    file_content = gr.Code(label="File Content", language="markdown", interactive=True, lines=25)
                    manual_render_btn = gr.Button("▶️ Render Preview", variant="secondary", size="sm")
                with gr.Tab("Preview"):
                    file_preview = gr.Markdown(label="Preview")
            with gr.Row():
                save_btn = gr.Button("💾 Save Changes", variant="primary")
                status_msg = gr.Textbox(label="Status", interactive=False, show_label=False)

    # --- Handlers ---

    def list_files(dir_path):
        files = list_kb_files(dir_path)
        return gr.update(choices=files)

    def search_files(dir_path, query):
        matches = search_kb_files(dir_path, query)
        return gr.update(choices=matches)

    def load_file_content(dir_path, filename):
        content = load_kb_content(dir_path, filename)
        val = content if content else "File not found or empty."
        return val, val

    def save_file_content(dir_path, filename, content):
        success, msg = save_kb_content(dir_path, filename, content)
        return msg

    def create_new_file(dir_path, filename):
        success, msg, new_name = create_kb_file(dir_path, filename)
        if success:
            return msg, list_files(dir_path), gr.update(value=new_name)
        else:
            return msg, gr.update(), gr.update()

    def delete_selected_file(dir_path, filename):
        success, msg = delete_kb_file(dir_path, filename)
        if success:
            return msg, list_files(dir_path), ""
        else:
            return msg, gr.update(), ""

    def rename_selected_file(dir_path, filename, new_name):
        success, msg, renamed_name = rename_kb_file(dir_path, filename, new_name)
        if success:
            return msg, list_files(dir_path), gr.update(value=renamed_name)
        else:
            return msg, gr.update(), gr.update()

    def import_uploaded_files(dir_path, files):
        success, msg = import_kb_files(dir_path, files)
        return msg, list_files(dir_path)

    # --- Event Wiring ---
    
    # Refresh / Directory Change
    refresh_btn.click(list_files, inputs=[kb_directory], outputs=[file_list])
    kb_directory.change(list_files, inputs=[kb_directory], outputs=[file_list])
    search_btn.click(search_files, inputs=[kb_directory, search_query], outputs=[file_list])
    search_query.submit(search_files, inputs=[kb_directory, search_query], outputs=[file_list])
    
    # Load
    file_list.change(load_file_content, inputs=[kb_directory, file_list], outputs=[file_content, file_preview])
    
    # Sync Edit to Preview
    file_content.change(lambda x: x, inputs=[file_content], outputs=[file_preview])
    manual_render_btn.click(lambda x: x, inputs=[file_content], outputs=[file_preview])
    
    # Save
    save_btn.click(save_file_content, inputs=[kb_directory, file_list, file_content], outputs=[status_msg])
    
    # Create
    create_btn.click(create_new_file, inputs=[kb_directory, new_filename], outputs=[status_msg, file_list, file_list])
    
    # Delete
    delete_btn.click(delete_selected_file, inputs=[kb_directory, file_list], outputs=[status_msg, file_list, file_content])

    # Rename
    rename_btn.click(rename_selected_file, inputs=[kb_directory, file_list, rename_target], outputs=[status_msg, file_list, file_list])

    # Import
    import_btn.click(import_uploaded_files, inputs=[kb_directory, upload_files], outputs=[status_msg, file_list])

    # Register components
    webui_manager.add_components("knowledge_base", {
        "kb_directory": kb_directory,
        "file_list": file_list,
        "file_content": file_content,
        "search_query": search_query,
        "search_btn": search_btn,
        "rename_target": rename_target,
        "upload_files": upload_files
    })