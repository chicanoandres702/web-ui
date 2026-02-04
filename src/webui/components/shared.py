import json
import re
import inspect
import os
import shutil
import logging
import gradio as gr
from gradio.components import Component
from typing import Dict, Any, AsyncGenerator, List, Type
from pydantic import BaseModel, create_model, Field
from src.utils import config
from src.webui.llm_helper import initialize_llm
from browser_use.agent.views import AgentOutput, AgentHistoryList
from src.utils.utils import parse_agent_thought
from src.utils.prompts import KNOWLEDGE_EXTRACTION_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.io_manager import IOManager

logger = logging.getLogger(__name__)

def safe_int(value: Any, default: int) -> int:
    """Safely converts a value to int, returning default on failure."""
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def update_model_dropdown(llm_provider):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    if llm_provider in config.model_names:
        return gr.update(choices=config.model_names[llm_provider], value=config.model_names[llm_provider][0], interactive=True)
    return gr.update(choices=[], value="", interactive=True, allow_custom_value=True)

def create_llm_settings_ui(
    prefix: str,
    label: str,
    default_provider: str = None,
    default_base_url: str = "http://137.131.201.189:11434",
    include_vision: bool = False,
    default_vision: bool = False,
    include_ctx: bool = False,
    include_strictness: bool = False,
    default_strictness: int = 5,
    visible: bool = True,
    interactive: bool = True
):
    """
    Creates a standardized group of LLM settings components.
    Returns a dictionary of components and the container row/group.
    """
    components = {}
    
    with gr.Group(visible=visible) as container:
        with gr.Row():
            provider = gr.Dropdown(
                choices=[p for p, m in config.model_names.items()],
                label=f"{label} Provider",
                value=default_provider,
                interactive=interactive,
                elem_id=f"{prefix}_provider"
            )
            model_name = gr.Dropdown(
                label=f"{label} Model Name",
                choices=config.model_names.get(default_provider, []) if default_provider else [],
                value=config.model_names.get(default_provider, [""])[0] if default_provider else None,
                interactive=interactive,
                allow_custom_value=True,
                elem_id=f"{prefix}_model_name"
            )
        
        components[f"{prefix}_provider"] = provider
        components[f"{prefix}_model_name"] = model_name

        with gr.Row():
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.6, step=0.1,
                label=f"{label} Temperature",
                interactive=interactive,
                elem_id=f"{prefix}_temperature"
            )
            components[f"{prefix}_temperature"] = temperature

            if include_vision:
                vision = gr.Checkbox(
                    label=f"Use Vision ({label})",
                    value=default_vision,
                    interactive=interactive,
                    elem_id=f"{prefix}_use_vision"
                )
                components[f"{prefix}_use_vision"] = vision
            
            if include_ctx:
                ctx = gr.Slider(
                    minimum=2**8, maximum=2**16, value=16000, step=1,
                    label="Ollama Context Length",
                    visible=(default_provider == "ollama"),
                    interactive=interactive,
                    elem_id=f"{prefix}_ollama_num_ctx"
                )
                components[f"{prefix}_ollama_num_ctx"] = ctx

            if include_strictness:
                strictness = gr.Slider(
                    minimum=1, maximum=10, value=default_strictness, step=1,
                    label="Confirmer Strictness",
                    interactive=interactive,
                    elem_id=f"{prefix}_strictness"
                )
                components[f"{prefix}_strictness"] = strictness

        with gr.Row():
            base_url = gr.Textbox(
                label="Base URL",
                value=default_base_url,
                info="API endpoint URL (if required)",
                elem_id=f"{prefix}_base_url"
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)",
                elem_id=f"{prefix}_api_key"
            )
            components[f"{prefix}_base_url"] = base_url
            components[f"{prefix}_api_key"] = api_key

    # Bind default events
    provider.change(
        lambda p: update_model_dropdown(p),
        inputs=[provider],
        outputs=[model_name]
    )
    
    if include_ctx:
        provider.change(
            fn=lambda x: gr.update(visible=x == "ollama"),
            inputs=[provider],
            outputs=[ctx]
        )

    return components, container

def create_agent_control_buttons(prefix: str):
    """
    Creates standard agent control buttons: Stop, Pause/Resume, Clear.
    Returns the button components.
    """
    stop_btn = gr.Button("⏹️ Stop", interactive=False, variant="stop", scale=1, elem_id=f"{prefix}_stop_button")
    pause_btn = gr.Button("⏸️ Pause", interactive=False, variant="secondary", scale=1, elem_id=f"{prefix}_pause_button")
    clear_btn = gr.Button("🗑️ Clear", interactive=True, variant="secondary", scale=1, elem_id=f"{prefix}_clear_button")
    return stop_btn, pause_btn, clear_btn

def get_agent_settings_values(webui_manager, components: Dict[Component, Any]) -> Dict[str, Any]:
    """Retrieves common agent settings from UI components."""
    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"agent_settings.{key}")
        val = components.get(comp, default) if comp else default
        return val if val is not None else default

    settings = {}
    settings["override_system_prompt"] = get_setting("override_system_prompt") or None
    settings["extend_system_prompt"] = get_setting("extend_system_prompt") or None
    settings["llm_provider_name"] = get_setting("llm_provider", None)
    settings["llm_model_name"] = get_setting("llm_model_name", None)
    settings["llm_temperature"] = get_setting("llm_temperature", 0.6)
    settings["use_vision"] = get_setting("llm_use_vision", True)
    settings["ollama_num_ctx"] = get_setting("llm_ollama_num_ctx", 16000)
    settings["llm_base_url"] = get_setting("llm_base_url") or None
    settings["llm_api_key"] = get_setting("llm_api_key") or None
    settings["max_steps"] = safe_int(get_setting("max_steps"), 100)
    settings["max_actions"] = safe_int(get_setting("max_actions"), 10)
    settings["fast_mode"] = get_setting("fast_mode", False)
    settings["use_memory"] = get_setting("enable_memory", False)
    settings["auto_save_on_stuck"] = get_setting("auto_save_on_stuck", True)
    settings["require_confirmation"] = get_setting("require_confirmation", False)
    settings["disable_hud"] = get_setting("disable_hud", False)
    settings["restrict_to_knowledge_base"] = get_setting("restrict_to_knowledge_base", False)
    settings["max_input_tokens"] = safe_int(get_setting("max_input_tokens"), 128000)
    
    tool_calling_str = get_setting("tool_calling_method", "auto")
    settings["tool_calling_method"] = tool_calling_str if tool_calling_str != "None" else None

    mcp_server_config_comp = webui_manager.id_to_component.get("agent_settings.mcp_server_config")
    mcp_server_config_str = components.get(mcp_server_config_comp) if mcp_server_config_comp else None
    settings["mcp_server_config"] = json.loads(mcp_server_config_str) if mcp_server_config_str else None

    # Planner
    settings["planner_llm_provider_name"] = get_setting("planner_llm_provider") or None
    settings["planner_llm_model_name"] = get_setting("planner_llm_model_name")
    settings["planner_llm_temperature"] = get_setting("planner_llm_temperature", 0.6)
    settings["planner_ollama_num_ctx"] = safe_int(get_setting("planner_llm_ollama_num_ctx"), 16000)
    settings["planner_llm_base_url"] = get_setting("planner_llm_base_url") or None
    settings["planner_llm_api_key"] = get_setting("planner_llm_api_key") or None
    settings["planner_use_vision"] = get_setting("planner_llm_use_vision", False)
    settings["enable_hierarchical_planning"] = get_setting("enable_hierarchical_planning", False)
    settings["planner_system_prompt"] = get_setting("planner_system_prompt") or None
    settings["enable_auto_pause"] = get_setting("enable_auto_pause", False)
    settings["enable_kb_auto_save"] = get_setting("enable_kb_auto_save", False)
    settings["enable_auto_streamline"] = get_setting("enable_auto_streamline", False)

    # Confirmer
    settings["confirmer_llm_provider_name"] = get_setting("confirmer_llm_provider") or None
    settings["confirmer_llm_model_name"] = get_setting("confirmer_llm_model_name")
    settings["confirmer_llm_temperature"] = get_setting("confirmer_llm_temperature", 0.6)
    settings["confirmer_ollama_num_ctx"] = safe_int(get_setting("confirmer_llm_ollama_num_ctx"), 16000)
    settings["confirmer_llm_base_url"] = get_setting("confirmer_llm_base_url") or None
    settings["confirmer_llm_api_key"] = get_setting("confirmer_llm_api_key") or None
    settings["confirmer_strictness"] = safe_int(get_setting("confirmer_strictness"), 5)
    settings["confirmer_use_vision"] = get_setting("confirmer_llm_use_vision", False)

    # Smart Retry
    settings["enable_smart_retry"] = get_setting("enable_smart_retry", False)
    settings["smart_retry_llm_provider"] = get_setting("smart_retry_llm_provider")
    settings["smart_retry_llm_model_name"] = get_setting("smart_retry_llm_model_name")
    settings["smart_retry_llm_temperature"] = get_setting("smart_retry_llm_temperature", 0.6)
    settings["smart_retry_llm_base_url"] = get_setting("smart_retry_llm_base_url") or None
    settings["smart_retry_llm_api_key"] = get_setting("smart_retry_llm_api_key") or None

    # Cost Saver
    settings["enable_cost_saver"] = get_setting("enable_cost_saver", False)
    settings["cheap_llm_provider"] = get_setting("cheap_llm_provider")
    settings["cheap_llm_model_name"] = get_setting("cheap_llm_model_name")
    settings["cheap_llm_temperature"] = get_setting("cheap_llm_temperature", 0.6)
    settings["cheap_llm_base_url"] = get_setting("cheap_llm_base_url") or None
    settings["cheap_llm_api_key"] = get_setting("cheap_llm_api_key") or None

    return settings

def get_browser_settings_values(webui_manager, components: Dict[Component, Any]) -> Dict[str, Any]:
    """Retrieves browser settings from UI components."""
    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"browser_settings.{key}")
        val = components.get(comp, default) if comp else default
        return val if val is not None else default

    settings = {}
    settings["browser_binary_path"] = get_setting("browser_binary_path") or None
    settings["browser_user_data_dir"] = get_setting("browser_user_data_dir") or None
    settings["use_own_browser"] = get_setting("use_own_browser", False)
    settings["enable_persistent_session"] = get_setting("enable_persistent_session", True)
    settings["keep_browser_open"] = get_setting("keep_browser_open", False)
    settings["headless"] = get_setting("headless", False)
    settings["disable_security"] = get_setting("disable_security", False)
    settings["window_w"] = safe_int(get_setting("window_w"), 1280)
    settings["window_h"] = safe_int(get_setting("window_h"), 1100)
    settings["cdp_url"] = get_setting("cdp_url") or None
    settings["wss_url"] = get_setting("wss_url") or None
    settings["save_recording_path"] = get_setting("save_recording_path") or None
    settings["save_trace_path"] = get_setting("save_trace_path") or None
    settings["save_agent_history_path"] = get_setting("save_agent_history_path", "./tmp/agent_history")
    settings["save_download_path"] = get_setting("save_download_path", "./tmp/downloads")
    
    return settings

async def safe_execution(func, *args, **kwargs):
    """
    Safely executes an async function or async generator, catching exceptions and displaying them via gr.Error.
    """
    try:
        if inspect.isasyncgenfunction(func):
            async for item in func(*args, **kwargs):
                yield item
        else:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            yield result
    except Exception as e:
        logger.error(f"Unhandled error in {func.__name__}: {e}", exc_info=True)
        gr.Error(f"Error: {str(e)}")

def get_valid_input_components(components):
    """
    Filters a list/set of components to return only those valid as inputs for Gradio events.
    Excludes layout components and output-only components.
    """
    excluded_types = (
        gr.Accordion,
        gr.Markdown,
        gr.HTML,
        gr.Row,
        gr.Column,
        gr.Group,
        gr.Tab,
        gr.Tabs,
        gr.Label,
        gr.Button
    )
    return [c for c in components if not isinstance(c, excluded_types)]

def json_to_pydantic_model(json_data: Dict[str, Any]) -> Type[BaseModel]:
    """Converts a JSON definition to a Pydantic model class."""
    model_name = json_data.get("name", "DynamicModel")
    fields_def = {}
    
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "List[str]": List[str],
        "List[int]": List[int],
        "List[float]": List[float]
    }
    
    for field in json_data.get("fields", []):
        f_name = field["name"]
        f_type_str = field["type"]
        f_desc = field.get("description", "")
        
        # Default to str if type not found
        f_type = type_map.get(f_type_str, str)
        
        fields_def[f_name] = (f_type, Field(description=f_desc))
        
    return create_model(model_name, **fields_def)

def map_dict_to_gradio_outputs(updates: Dict[Component, Any], outputs: List[Component]) -> List[Any]:
    """
    Maps a dictionary of component updates to a list of output components.
    Returns a list of values corresponding to the outputs order.
    """
    return [updates.get(comp, gr.update()) for comp in outputs]

def create_template_manager_ui(label: str, default_value: str = ""):
    """
    Creates a reusable UI for managing text templates (Load/Save/Edit).
    """
    with gr.Group():
        gr.Markdown(f"### {label}")
        with gr.Row():
            dropdown = gr.Dropdown(label="Select Template", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("🔄", scale=0)
            load_btn = gr.Button("📂 Load", variant="secondary", scale=1)
        
        with gr.Row():
            new_name = gr.Textbox(label="New Template Name", placeholder="e.g. my_strategy", scale=3)
            save_btn = gr.Button("💾 Save", variant="primary", scale=1)
            
        content = gr.Textbox(
            label="Content", 
            lines=6, 
            value=default_value,
            interactive=True
        )
    return dropdown, refresh_btn, load_btn, new_name, save_btn, content

def setup_template_manager_logic(
    base_dir: str,
    dropdown: gr.Dropdown,
    refresh_btn: gr.Button,
    load_btn: gr.Button,
    new_name: gr.Textbox,
    save_btn: gr.Button,
    content: gr.Textbox
):
    """Wires up the logic for the template manager UI."""
    os.makedirs(base_dir, exist_ok=True)

    def list_templates():
        if not os.path.exists(base_dir): return gr.update(choices=[])
        files = [f.replace(".txt", "") for f in os.listdir(base_dir) if f.endswith(".txt")]
        return gr.update(choices=sorted(files))

    def load_template(name):
        if not name: return None
        path = os.path.join(base_dir, f"{name}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_template(name, text):
        if not name: 
            gr.Warning("Please enter a template name.")
            return gr.update()
        path = os.path.join(base_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return list_templates()

    refresh_btn.click(list_templates, outputs=[dropdown])
    load_btn.click(load_template, inputs=[dropdown], outputs=[content])
    save_btn.click(save_template, inputs=[new_name, content], outputs=[dropdown])

async def update_mcp_server(mcp_file: str, webui_manager):
    """
    Update the MCP server.
    """
    # Check for Browser Use Agent controller
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Closing Browser Use controller because MCP file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    # Check for Deep Research Agent
    if hasattr(webui_manager, "dr_agent") and webui_manager.dr_agent:
        logger.warning("⚠️ Closing Deep Research Agent MCP client because MCP file has changed!")
        await webui_manager.dr_agent.close_mcp_client()

    if not mcp_file:
        return None, gr.update(visible=False)

    if not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)

async def initialize_agent_llms(settings: Dict[str, Any]):
    """Initializes all LLMs based on settings."""
    main_llm = await initialize_llm(
        settings["llm_provider_name"],
        settings["llm_model_name"],
        settings["llm_temperature"],
        settings["llm_base_url"],
        settings["llm_api_key"],
        settings["ollama_num_ctx"] if settings["llm_provider_name"] == "ollama" else None,
    )

    planner_llm = None
    if settings["planner_llm_provider_name"]:
        planner_llm = await initialize_llm(
            settings["planner_llm_provider_name"],
            settings["planner_llm_model_name"],
            settings["planner_llm_temperature"],
            settings["planner_llm_base_url"],
            settings["planner_llm_api_key"],
            settings["planner_ollama_num_ctx"] if settings["planner_llm_provider_name"] == "ollama" else None,
        )

    confirmer_llm = None
    if settings["confirmer_llm_provider_name"]:
        confirmer_llm = await initialize_llm(
            settings["confirmer_llm_provider_name"],
            settings["confirmer_llm_model_name"],
            settings["confirmer_llm_temperature"],
            settings["confirmer_llm_base_url"],
            settings["confirmer_llm_api_key"],
            settings["confirmer_ollama_num_ctx"] if settings["confirmer_llm_provider_name"] == "ollama" else None,
        )

    smart_retry_llm = None
    if settings["enable_smart_retry"] and settings["smart_retry_llm_provider"] and settings["smart_retry_llm_model_name"]:
        smart_retry_llm = await initialize_llm(
            settings["smart_retry_llm_provider"],
            settings["smart_retry_llm_model_name"],
            settings["smart_retry_llm_temperature"],
            settings["smart_retry_llm_base_url"],
            settings["smart_retry_llm_api_key"]
        )

    cheap_llm = None
    if settings["enable_cost_saver"] and settings["cheap_llm_provider"] and settings["cheap_llm_model_name"]:
        cheap_llm = await initialize_llm(
            settings["cheap_llm_provider"],
            settings["cheap_llm_model_name"],
            settings["cheap_llm_temperature"],
            settings["cheap_llm_base_url"],
            settings["cheap_llm_api_key"]
        )
        
    return main_llm, planner_llm, confirmer_llm, smart_retry_llm, cheap_llm

def read_text_file(path: str) -> str:
    """Safely reads text from a file with error handling."""
    return IOManager.read_file_sync(path) or ""

def save_text_file(path: str, content: str, mode: str = "w") -> bool:
    """Safely saves text to a file with error handling, creating directories if needed."""
    return IOManager.write_file_sync(path, content, mode=mode)

def rename_file(old_path: str, new_path: str) -> bool:
    """Safely renames/moves a file."""
    if not old_path or not os.path.exists(old_path):
        return False
    try:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)
        return True
    except Exception as e:
        logger.error(f"Error renaming file {old_path} to {new_path}: {e}")
        return False

def save_uploaded_file(temp_path: str, dest_path: str) -> bool:
    """Safely saves an uploaded temp file to a destination."""
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(temp_path, dest_path)
        return True
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return False

def format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    if not model_output:
        return ""

    content = []
    try:
        # 1. Thoughts
        thought = getattr(model_output, "thought", None)
        if thought:
            parsed = parse_agent_thought(thought)
            if any(parsed.values()):
                content.append("<div class='agent-thought-container'>")
                if parsed["Status"]:
                    content.append(f"<div style='margin-bottom:8px;'><span class='thought-badge'>Status</span><span style='font-weight:600'>{parsed['Status']}</span></div>")
                
                for key, label, color in [
                    ("Reasoning", "🤔 Reasoning:", None),
                    ("Analysis", "🔎 Analysis:", None),
                    ("Challenge", "⚠️ Challenge:", "var(--error-text-color)"),
                    ("Next Steps", "⏭️ Next Steps:", None)
                ]:
                    if parsed[key]:
                        style = f" style='color: {color};'" if color else ""
                        content.append(f"<div style='margin-bottom:6px;'{style}><strong>{label}</strong> {parsed[key]}</div>")
                content.append("</div>")
            else:
                content.append(f"<div class='agent-thought-container'><strong>Thought:</strong> {thought}</div>")

        # 2. Validation/Confirmer
        validation_info = getattr(model_output, "validation", None) or getattr(model_output, "confirmer", None)
        if validation_info:
            content.append(f"<div style='margin-top:10px; padding:8px; border-left: 3px solid #10b981; background-color: rgba(16, 185, 129, 0.1);'><strong>✅ Confirmer:</strong><br/>{validation_info}</div>")

        # 3. Actions
        actions = getattr(model_output, "action", [])
        if actions:
            content.append("<div style='margin-top:10px;'><strong>🛠️ Actions:</strong><ul style='margin-top:5px; padding-left:20px;'>")
            for action in actions:
                act_str = str(action.model_dump(exclude_none=True)) if hasattr(action, "model_dump") else str(action)
                content.append(f"<li>{act_str}</li>")
            content.append("</ul></div>")

    except Exception as e:
        logger.error(f"Error formatting agent output: {e}", exc_info=True)
        return f"Error formatting output: {e}"

    return "".join(content)

def render_plan_markdown(plan_data) -> str:
    """Renders the execution plan as Markdown with status icons."""
    if not plan_data:
        return ""
    
    md = "### 📋 Execution Plan\n"
    for i, item in enumerate(plan_data):
        icon = "⬜"
        style = ""
        if item["status"] == "completed":
            icon = "✅"
            style = "text-decoration: line-through; color: gray;"
        elif item["status"] == "in_progress":
            icon = "👉"
            style = "font-weight: bold; color: #3b82f6;" # Blue
        elif item["status"] == "failed":
            icon = "❌"
            style = "color: red;"
        
        step_text = item["step"]
        if style:
            step_text = f"<span style='{style}'>{step_text}</span>"
        if "action" in item:
             step_text += f" <span style='font-size:0.8em; color:gray; font-family:monospace;'>[{item['action']}]</span>"
        
        md += f"{i+1}. {icon} {step_text}\n"
    return md

async def generate_knowledge_suggestion(history: AgentHistoryList, llm: BaseChatModel) -> str:
    """Generates a knowledge base entry suggestion from the agent history."""
    if not history or not history.history:
        return ""
    
    # Summarize history to fit context if needed, or pass full history if small
    # For now, we pass a simplified string representation
    history_lines = []
    for i, item in enumerate(history.history):
        step_str = f"Step {i+1}:"
        if item.model_output:
            thought = getattr(item.model_output, "thought", None)
            if thought:
                step_str += f"\n  Thought: {thought}"
            actions = getattr(item.model_output, "action", None)
            if actions:
                action_strs = []
                for a in actions:
                    if hasattr(a, "model_dump"):
                        action_strs.append(str(a.model_dump(exclude_none=True)))
                    else:
                        action_strs.append(str(a))
                step_str += f"\n  Actions: {', '.join(action_strs)}"
        if item.result:
            # Truncate result if too long to avoid context limit issues
            res = str(item.result)
            if len(res) > 500: res = res[:500] + "..."
            step_str += f"\n  Result: {res}"
        history_lines.append(step_str)
    
    history_str = "\n".join(history_lines)
    
    messages = [SystemMessage(content=KNOWLEDGE_EXTRACTION_PROMPT), HumanMessage(content=f"Session History:\n{history_str}")]
    
    try:
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating knowledge: {e}")
        return ""

async def process_knowledge_generation(
    history: AgentHistoryList, 
    llm: BaseChatModel, 
    enable_auto_save: bool, 
    memory_file_path: str
) -> Dict[str, Any]:
    """
    Generates knowledge from history and handles auto-saving if enabled.
    Returns a dict with keys: 'status', 'title', 'content', 'message', 'saved_path'.
    """
    if not history or not history.history:
        return {"status": "no_history"}

    logger.info("Generating knowledge suggestion...")
    kb_response = await generate_knowledge_suggestion(history, llm)

    if not kb_response or "NONE" in kb_response:
        return {"status": "no_knowledge"}

    # Robust parsing using regex
    title = "knowledge_snippet.md"
    content = kb_response

    title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", kb_response, re.IGNORECASE)
    content_match = re.search(r"CONTENT:\s*(.*)", kb_response, re.IGNORECASE | re.DOTALL)

    if title_match and content_match:
        title = title_match.group(1).strip()
        content = content_match.group(1).strip()
    elif "TITLE:" in kb_response and "CONTENT:" in kb_response:
        # Fallback to simple split if regex fails but keywords exist
        try:
            parts = kb_response.split("CONTENT:")
            title = parts[0].replace("TITLE:", "").replace("Title:", "").strip()
            content = parts[1].strip()
        except Exception:
            return {"status": "parse_error", "raw_response": kb_response}

    result = {"status": "generated", "title": title, "content": content}

    if enable_auto_save:
        base_dir = os.path.dirname(memory_file_path) if memory_file_path else "./tmp/memory"
        save_path = os.path.join(base_dir, title)
        if save_text_file(save_path, content):
            result.update({"status": "saved", "saved_path": save_path, "message": f"✅ Auto-saved knowledge to {title}"})
        else:
            result.update({"status": "save_error", "message": f"❌ Failed to auto-save knowledge to {title}"})
    else:
        result["message"] = "🧠 I have generated some potential knowledge from this session. Please review it below."

    return result
