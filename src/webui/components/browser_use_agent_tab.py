import asyncio
import json
import logging
import os
import uuid
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

import gradio as gr

# from browser_use.agent.service import Agent
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.webui.llm_helper import initialize_llm
from src.webui.webui_manager import WebuiManager

from src.utils.prompts import FULL_SYSTEM_PROMPT
from src.utils.utils import get_progress_bar_html, parse_agent_thought
logger = logging.getLogger(__name__)

# --- Helper Functions --- (Defined at module level)

def _format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    content = ""
    if model_output:
        try:
            # Try to format nicely
            try:
                thought = getattr(model_output, "thought", None)
            except Exception:
                thought = None
            if thought:
                parsed = parse_agent_thought(thought)
                if any(parsed[k] for k in ["Status", "Challenge", "Analysis", "Next Steps"]):
                    content += "<div class='agent-thought-container'>"
                    if parsed["Status"]: 
                        content += f"<div style='margin-bottom:8px;'><span class='thought-badge'>Status</span><span style='font-weight:600'>{parsed['Status']}</span></div>"
                    if parsed["Reasoning"]: 
                        content += f"<div style='margin-bottom:6px;'><strong>🤔 Reasoning:</strong> {parsed['Reasoning']}</div>"
                    if parsed["Analysis"]: 
                        content += f"<div style='margin-bottom:6px;'><strong>🔎 Analysis:</strong> {parsed['Analysis']}</div>"
                    if parsed["Challenge"]: 
                        content += f"<div style='margin-bottom:6px; color: var(--error-text-color);'><strong>⚠️ Challenge:</strong> {parsed['Challenge']}</div>"
                    if parsed["Next Steps"]: 
                        content += f"<div><strong>⏭️ Next Steps:</strong> {parsed['Next Steps']}</div>"
                    content += "</div>"
                else:
                    content += f"<div class='agent-thought-container'><strong>Thought:</strong> {thought}</div>"

            # --- Confirmer/Validation Visualization ---
            # Attempt to retrieve validation info from the output object
            validation_info = getattr(model_output, "validation", None) or getattr(model_output, "confirmer", None)
            
            if not validation_info and hasattr(model_output, "model_dump"):
                dump = model_output.model_dump(exclude_none=True)
                validation_info = dump.get("validation") or dump.get("confirmer")

            if validation_info:
                content += f"<div style='margin-top:10px; padding:8px; border-left: 3px solid #10b981; background-color: rgba(16, 185, 129, 0.1);'>"
                content += f"<strong>✅ Confirmer:</strong><br/>"
                if isinstance(validation_info, dict):
                    for k, v in validation_info.items():
                        content += f"- {k}: {v}\n"
                else:
                    content += f"{validation_info}\n"
                content += "</div>"

            actions = getattr(model_output, "action", [])
            if actions:
                content += "<div style='margin-top:10px;'><strong>🛠️ Actions:</strong><ul style='margin-top:5px; padding-left:20px;'>"
                for action in actions:
                    # action is likely a Pydantic model
                    if hasattr(action, "model_dump"):
                        act_dict = action.model_dump(exclude_none=True)
                        for k, v in act_dict.items():
                            content += f"<li><strong>{k}:</strong> {v}</li>"
                    else:
                        content += f"<li>{str(action)}</li>"
                content += "</ul></div>"
            
            if not content:
                # Fallback to JSON
                dump = model_output.model_dump(exclude_none=True) if hasattr(model_output, "model_dump") else str(model_output)
                json_string = json.dumps(dump, indent=2, ensure_ascii=False)
                content = f"```json\n{json_string}\n```"

        except AttributeError as ae:
            logger.error(
                f"AttributeError during model dump: {ae}."
            )
            content = f"Error: Could not format agent output (AttributeError: {ae}).\nRaw output: {str(model_output)}"
        except Exception as e:
            logger.error(f"Error formatting agent output: {e}", exc_info=True)
            # Fallback to simple string representation on error
            content = f"Error formatting agent output.\nRaw output:\n{str(model_output)}"

    return content.strip()


# --- Updated Callback Implementation ---


async def _handle_new_step(
        webui_manager: WebuiManager, state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback for each step taken by the agent, including screenshot display."""

    # Use the correct chat history attribute name from the user's code
    if not hasattr(webui_manager, "bu_chat_history"):
        logger.error(
            "Attribute 'bu_chat_history' not found in webui_manager! Cannot add chat message."
        )
        # Initialize it maybe? Or raise an error? For now, log and potentially skip chat update.
        webui_manager.bu_chat_history = []  # Initialize if missing (consider if this is the right place)
        # return # Or stop if this is critical
    step_num -= 1
    logger.info(f"Step {step_num} completed.")

    # --- Calculate Progress ---
    max_steps = webui_manager.bu_max_steps if webui_manager.bu_max_steps > 0 else 100
    progress = min(100, int((step_num / max_steps) * 100))

    # --- Update Status ---
    status_text = f"### Step {step_num}/{max_steps}\n"
    
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller and getattr(webui_manager.bu_controller, "fast_mode", False):
         status_text += "⚡ **FAST MODE ACTIVE**\n"
         
    if getattr(webui_manager.bu_agent, "switched_to_retry_model", False):
         status_text += "<span class='retry-badge'>🧠 Smart Retry Active</span>\n"

    if getattr(webui_manager.bu_agent, "using_cheap_model", False):
         status_text += "<span class='retry-badge' style='background-color: #10b981;'>💰 Cost Saver Active</span>\n"

    if hasattr(webui_manager, "bu_controller") and getattr(webui_manager.bu_controller, "require_confirmation", False):
         status_text += "<span class='retry-badge' style='background-color: #f59e0b;'>✋ Human in Loop</span>\n"

    status_text += get_progress_bar_html(progress, label="Task Progress")
    status_text += f"**URL:** {state.url}\n"
    status_text += f"**Title:** {state.title}\n"
    
    try:
        thought = getattr(output, "thought", None)
    except Exception:
        thought = None
    if thought:
        parsed = parse_agent_thought(thought)
        if any(parsed[k] for k in ["Status", "Reasoning", "Challenge", "Analysis", "Next Steps"]):
             status_text += "\n<div style='background-color: var(--background-fill-secondary); padding: 10px; border-radius: 8px; margin-top: 10px;'>"
             if parsed["Status"]: status_text += f"<div><strong>Status:</strong> {parsed['Status']}</div>"
             if parsed["Reasoning"]: status_text += f"<div style='margin-top:5px;'><strong>Reasoning:</strong> {parsed['Reasoning']}</div>"
             if parsed["Analysis"]: status_text += f"<div style='margin-top:5px;'><strong>Analysis:</strong> {parsed['Analysis']}</div>"
             if parsed["Challenge"]: status_text += f"<div style='margin-top:5px; color: var(--error-text-color);'><strong>Challenge:</strong> {parsed['Challenge']}</div>"
             if parsed["Next Steps"]: status_text += f"<div style='margin-top:5px;'><strong>Next Steps:</strong> {parsed['Next Steps']}</div>"
             status_text += "</div>"
        else:
             status_text += f"\n**Thought:** {thought}\n"
    webui_manager.bu_agent_status = status_text

    # --- Screenshot Handling ---
    screenshot_html = ""
    # Ensure state.screenshot exists and is not empty before proceeding
    # Use getattr for safer access
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            # Basic validation: check if it looks like base64
            if (
                    isinstance(screenshot_data, str) and len(screenshot_data) > 100
            ):  # Arbitrary length check
                # *** UPDATED STYLE: Removed centering, adjusted width ***
                img_tag = f'<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'
                screenshot_html = (
                        img_tag + "<br/>"
                )  # Use <br/> for line break after inline-block image
            else:
                logger.warning(
                    f"Screenshot for step {step_num} seems invalid (type: {type(screenshot_data)}, len: {len(screenshot_data) if isinstance(screenshot_data, str) else 'N/A'})."
                )
                screenshot_html = "**[Invalid screenshot data]**<br/>"

        except Exception as e:
            logger.error(
                f"Error processing or formatting screenshot for step {step_num}: {e}",
                exc_info=True,
            )
            screenshot_html = "**[Error displaying screenshot]**<br/>"
    else:
        logger.debug(f"No screenshot available for step {step_num}.")

    # --- Format Agent Output ---
    formatted_output = _format_agent_output(output)  # Use the updated function

    # --- Combine and Append to Chat ---
    step_header = f"--- **Step {step_num}** ---"
    # Combine header, image (with line break), and JSON block
    final_content = step_header + "<br/>" + screenshot_html + formatted_output

    chat_message = {
        "role": "assistant",
        "content": final_content.strip(),  # Remove leading/trailing whitespace
    }

    # Append to the correct chat history list
    webui_manager.bu_chat_history.append(chat_message)


def _handle_done(webui_manager: WebuiManager, history: AgentHistoryList):
    """Callback when the agent finishes the task (success or failure)."""
    logger.info(
        f"Agent task finished. Duration: {history.total_duration_seconds():.2f}s, Tokens: {history.total_input_tokens()}"
    )
    final_summary = "**Task Completed**\n"
    final_summary += f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
    final_summary += f"- Total Input Tokens: {history.total_input_tokens()}\n"  # Or total tokens if available

    final_result = history.final_result()
    if final_result:
        final_summary += f"- Final Result: {final_result}\n"

    errors = history.errors()
    if errors and any(errors):
        final_summary += f"- **Errors:**\n```\n{errors}\n```\n"
    else:
        final_summary += "- Status: Success\n"

    webui_manager.bu_chat_history.append(
        {"role": "assistant", "content": final_summary}
    )


async def _ask_assistant_callback(
        webui_manager: WebuiManager, query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")

    if not hasattr(webui_manager, "_chat_history"):
        logger.error("Chat history not found in webui_manager during ask_assistant!")
        return {"response": "Internal Error: Cannot display help request."}

    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )

    # Use state stored in webui_manager
    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None  # Reset previous response

    try:
        logger.info("Waiting for user response event...")
        await asyncio.wait_for(
            webui_manager.bu_response_event.wait(), timeout=3600.0
        )  # Long timeout
        logger.info("User response event received.")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for user assistance.")
        webui_manager.bu_chat_history.append(
            {
                "role": "assistant",
                "content": "**Timeout:** No response received. Trying to proceed.",
            }
        )
        webui_manager.bu_response_event = None  # Clear the event
        return {"response": "Timeout: User did not respond."}  # Inform the agent

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append(
        {"role": "user", "content": response}
    )  # Show user response in chat
    webui_manager.bu_response_event = (
        None  # Clear the event for the next potential request
    )
    return {"response": response}


async def _confirm_action_callback(
        webui_manager: WebuiManager, action_name: str, params: Dict[str, Any], browser_context: BrowserContext
) -> str:
    """Callback triggered by the controller before executing a critical action."""
    logger.info(f"Action '{action_name}' requires confirmation.")

    if not hasattr(webui_manager, "bu_chat_history"):
        return "yes" # Default to proceed if UI not ready

    # Format params for display
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    
    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"✋ **Confirmation Required**\n\nI am about to execute:\n**Action:** `{action_name}`\n**Params:** `{params_str}`\n\n"
                       f"- **Approve:** Press 'Submit Response' (or type 'yes')\n"
                       f"- **Reject:** Type 'no' or 'skip'\n"
                       f"- **Modify:** Type `modify: {{ \"key\": \"value\" }}` to change parameters (e.g., `modify: {{ \"text\": \"new text\" }}`)"
        }
    )

    # Reuse the response event mechanism
    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None

    try:
        logger.info("Waiting for user confirmation...")
        await asyncio.wait_for(
            webui_manager.bu_response_event.wait(), timeout=3600.0
        )
        logger.info("User confirmation received.")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for confirmation.")
        webui_manager.bu_response_event = None
        return "yes" # Default to proceed on timeout? Or maybe "no"? Let's proceed to avoid blocking forever.

    response = webui_manager.bu_user_help_response or "yes"
    webui_manager.bu_chat_history.append({"role": "user", "content": response})
    webui_manager.bu_response_event = None
    return response


async def handle_resume_session(webui_manager: WebuiManager, history_file: str, components: Dict[gr.components.Component, Any]):
    """Resumes a session from a history file."""
    if not history_file or not os.path.exists(history_file):
        gr.Warning("History file not found.")
        yield {}
        return

    logger.info(f"Resuming session from: {history_file}")
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        # Reconstruct AgentHistoryList
        history = AgentHistoryList.model_validate(history_data)
        
        # Initialize agent with this history
        # We need to trigger the initialization logic similar to run_agent_task but without running it immediately
        # For simplicity, we'll use a modified version of the initialization part of run_agent_task
        
        # ... (Initialization logic reused/adapted) ...
        # Since we can't easily extract just the init logic without refactoring run_agent_task completely,
        # we will assume the user has set up the environment (browser, etc.) and we just need to create the agent.
        
        # Ensure browser/context exists (similar to run_agent_task)
        # ... (This part relies on the user having clicked "Run" at least once or we duplicate the init logic)
        # Ideally, we should refactor run_agent_task to separate initialization. 
        # For now, we will trigger a "dry run" init if needed.
        
        # Let's try to initialize the agent using the current settings from components
        # This requires calling the same setup code.
        
        # HACK: We will yield a generator that calls run_agent_task but with a special flag or just sets up the agent.
        # Actually, we can just set up the agent here if we have access to the necessary helpers.
        # We have access to `initialize_llm` and `CustomBrowser`, etc.
        
        # ... (Setup LLM, Browser, Context - simplified for resume) ...
        # We assume the user wants to use the *current* settings for the resumed session (e.g. model), 
        # but the *history* from the file.
        
        # We need to yield UI updates to show we are loading
        yield {
            webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(value="### Agent Status\nResuming Session...")
        }
        
        # We will use a trick: we will set a flag in webui_manager to indicate we want to resume with specific history,
        # and then trigger the run_agent_task logic but pause it immediately? 
        # No, better to just initialize the agent object and populate chat.
        
        # 1. Initialize Agent (using a helper if we had one, or copy-paste essential init)
        # For now, let's assume the user has to click "Run" to start the browser, so we just load the history into memory
        # and when they click "Run", we use it? No, "Resume" should probably restore the state visible in UI.
        
        webui_manager.bu_chat_history = []
        
        # Replay history to chat
        for i, item in enumerate(history.history):
            step_num = i + 1
            state = item.state
            output = item.model_output
            
            # Reconstruct chat message
            # We can reuse _handle_new_step logic but synchronously and without real screenshots (unless saved?)
            # Saved history usually doesn't have base64 screenshots embedded (too large), unless configured.
            # If they are missing, we just show text.
            
            formatted_output = _format_agent_output(output)
            step_header = f"--- **Step {step_num} (Restored)** ---"
            
            screenshot_html = ""
            if state.screenshot:
                 screenshot_html = f'<img src="data:image/jpeg;base64,{state.screenshot}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" /><br/>'
            
            final_content = step_header + "<br/>" + screenshot_html + formatted_output
            webui_manager.bu_chat_history.append({"role": "assistant", "content": final_content.strip()})
            
        # Store history to be used when "Run" is clicked next? 
        # Or initialize the agent NOW.
        # To initialize the agent now, we need the browser.
        # If we don't have a browser, we can't fully init the agent.
        # So we will store this history in `webui_manager.bu_resumed_history` and use it in `run_agent_task`.
        
        webui_manager.bu_resumed_history = history
        webui_manager.bu_agent_status = f"### Agent Status\nSession Restored ({len(history.history)} steps)."
        
        yield {
            webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(value=webui_manager.bu_chat_history),
            webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(value=webui_manager.bu_agent_status),
            webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(placeholder="Session restored. Enter next task..."),
        }
        
        gr.Info(f"Session restored with {len(history.history)} steps. Click 'Submit Task' to continue.")

    except Exception as e:
        logger.error(f"Error resuming session: {e}", exc_info=True)
        gr.Error(f"Failed to resume session: {e}")


# --- Core Agent Execution Logic --- (Needs access to webui_manager)


async def run_agent_task(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""

    # --- Get Components ---
    # Need handles to specific UI components to update them
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.stop_button"
    )
    pause_resume_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.pause_resume_button"
    )
    clear_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.clear_button"
    )
    retry_button_comp = webui_manager.get_component_by_id("browser_use_agent.retry_button")
    chatbot_comp = webui_manager.get_component_by_id("browser_use_agent.chatbot")
    agent_status_comp = webui_manager.get_component_by_id("browser_use_agent.agent_status")
    history_file_comp = webui_manager.get_component_by_id(
        "browser_use_agent.agent_history_file"
    )
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    browser_view_comp = webui_manager.get_component_by_id(
        "browser_use_agent.browser_view"
    )
    brain_file_comp = webui_manager.get_component_by_id("browser_use_agent.brain_file")
    memory_file_comp = webui_manager.get_component_by_id("browser_use_agent.memory_file")

    # --- 1. Get Task and Initial UI Update ---
    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    # Save task for retry
    webui_manager.bu_last_task_prompt = task

    # Set running state indirectly via _current_task
    webui_manager.bu_chat_history.append({"role": "user", "content": task})
    webui_manager.bu_agent_status = "### Agent Status\nRunning..."

    yield {
        user_input_comp: gr.Textbox(
            value="", interactive=True, placeholder="Agent is running... Enter text to steer/add instructions."
        ),
        run_button_comp: gr.Button(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
        retry_button_comp: gr.Button(interactive=False),
        stop_button_comp: gr.Button(interactive=True),
        pause_resume_button_comp: gr.Button(value="⏸️ Pause", interactive=True),
        clear_button_comp: gr.Button(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        agent_status_comp: gr.update(value=webui_manager.bu_agent_status),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None),
    }

    # --- Agent Settings ---
    # Access settings values via components dict, getting IDs from webui_manager
    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"agent_settings.{key}")
        val = components.get(comp, default) if comp else default
        return val if val is not None else default

    override_system_prompt = get_setting("override_system_prompt") or None
    extend_system_prompt = get_setting("extend_system_prompt") or None
    llm_provider_name = get_setting(
        "llm_provider", None
    )  # Default to None if not found
    llm_model_name = get_setting("llm_model_name", None)
    llm_temperature = get_setting("llm_temperature", 0.6)
    use_vision = get_setting("use_vision", True)
    ollama_num_ctx = get_setting("ollama_num_ctx", 16000)
    llm_base_url = get_setting("llm_base_url") or None
    llm_api_key = get_setting("llm_api_key") or None
    max_steps = get_setting("max_steps", 100)
    webui_manager.bu_max_steps = max_steps
    max_actions = get_setting("max_actions", 10)
    fast_mode = get_setting("fast_mode", False)
    auto_save_on_stuck = get_setting("auto_save_on_stuck", True)
    require_confirmation = get_setting("require_confirmation", False)
    
    if fast_mode:
        webui_manager.bu_agent_status += "\n⚡ **Fast Mode Active**"

    max_input_tokens = get_setting("max_input_tokens", 128000)
    tool_calling_str = get_setting("tool_calling_method", "auto")
    tool_calling_method = tool_calling_str if tool_calling_str != "None" else None
    mcp_server_config_comp = webui_manager.id_to_component.get(
        "agent_settings.mcp_server_config"
    )
    mcp_server_config_str = (
        components.get(mcp_server_config_comp) if mcp_server_config_comp else None
    )
    mcp_server_config = (
        json.loads(mcp_server_config_str) if mcp_server_config_str else None
    )

    # --- Brain & Memory ---
    brain_file = components.get(brain_file_comp)
    memory_file = components.get(memory_file_comp)

    if brain_file and os.path.exists(brain_file):
        try:
            with open(brain_file, "r", encoding="utf-8") as f:
                override_system_prompt = f.read()
            logger.info(f"Loaded Brain from {brain_file}")
        except Exception as e:
            logger.error(f"Failed to load Brain file: {e}")

    if memory_file and os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memory_content = f.read()
            if memory_content.strip():
                extend_system_prompt = (extend_system_prompt or "") + f"\n\nLONG TERM MEMORY:\n{memory_content}"
            logger.info(f"Loaded Memory from {memory_file}")
        except Exception as e:
            logger.error(f"Failed to load Memory file: {e}")
    
    # Note: If memory file doesn't exist, we still pass the path to the controller so it can create it on write.

    # --- Learning Prompt Injection ---
    extend_system_prompt = (extend_system_prompt or "") + FULL_SYSTEM_PROMPT

    # Planner LLM Settings (Optional)
    planner_llm_provider_name = get_setting("planner_llm_provider") or None
    planner_llm = None
    planner_use_vision = False
    if planner_llm_provider_name:
        planner_llm_model_name = get_setting("planner_llm_model_name")
        planner_llm_temperature = get_setting("planner_llm_temperature", 0.6)
        planner_ollama_num_ctx = get_setting("planner_ollama_num_ctx", 16000)
        planner_llm_base_url = get_setting("planner_llm_base_url") or None
        planner_llm_api_key = get_setting("planner_llm_api_key") or None
        planner_use_vision = get_setting("planner_use_vision", False)

        planner_llm = await initialize_llm(
            planner_llm_provider_name,
            planner_llm_model_name,
            planner_llm_temperature,
            planner_llm_base_url,
            planner_llm_api_key,
            planner_ollama_num_ctx if planner_llm_provider_name == "ollama" else None,
        )

    # Confirmer LLM Settings (Optional)
    confirmer_llm_provider_name = get_setting("confirmer_llm_provider") or None
    confirmer_llm = None
    confirmer_strictness = 5
    confirmer_use_vision = False
    if confirmer_llm_provider_name:
        confirmer_llm_model_name = get_setting("confirmer_llm_model_name")
        confirmer_llm_temperature = get_setting("confirmer_llm_temperature", 0.6)
        confirmer_ollama_num_ctx = get_setting("confirmer_ollama_num_ctx", 16000)
        confirmer_llm_base_url = get_setting("confirmer_llm_base_url") or None
        confirmer_llm_api_key = get_setting("confirmer_llm_api_key") or None
        confirmer_strictness = get_setting("confirmer_strictness", 5)
        confirmer_use_vision = get_setting("confirmer_use_vision", False)
        
        if fast_mode:
            confirmer_strictness = 1
            
        confirmer_llm = await initialize_llm(
            confirmer_llm_provider_name,
            confirmer_llm_model_name,
            confirmer_llm_temperature,
            confirmer_llm_base_url,
            confirmer_llm_api_key,
            confirmer_ollama_num_ctx if confirmer_llm_provider_name == "ollama" else None,
        )
        
    # Smart Retry LLM Settings
    enable_smart_retry = get_setting("enable_smart_retry", False)
    smart_retry_llm = None
    if enable_smart_retry:
        retry_provider = get_setting("smart_retry_llm_provider")
        retry_model = get_setting("smart_retry_llm_model_name")
        retry_temp = get_setting("smart_retry_llm_temperature", 0.6)
        retry_base_url = get_setting("smart_retry_llm_base_url") or None
        retry_api_key = get_setting("smart_retry_llm_api_key") or None
        
        if retry_provider and retry_model:
            smart_retry_llm = await initialize_llm(
                retry_provider, retry_model, retry_temp, retry_base_url, retry_api_key
            )

    # Cost Saver LLM Settings
    enable_cost_saver = get_setting("enable_cost_saver", False)
    cheap_llm = None
    if enable_cost_saver:
        cheap_provider = get_setting("cheap_llm_provider")
        cheap_model = get_setting("cheap_llm_model_name")
        cheap_temp = get_setting("cheap_llm_temperature", 0.6)
        cheap_base_url = get_setting("cheap_llm_base_url") or None
        cheap_api_key = get_setting("cheap_llm_api_key") or None
        
        if cheap_provider and cheap_model:
            cheap_llm = await initialize_llm(
                cheap_provider, cheap_model, cheap_temp, cheap_base_url, cheap_api_key
            )

    # --- Browser Settings ---
    def get_browser_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"browser_settings.{key}")
        val = components.get(comp, default) if comp else default
        return val if val is not None else default

    browser_binary_path = get_browser_setting("browser_binary_path") or None
    browser_user_data_dir = get_browser_setting("browser_user_data_dir") or None
    use_own_browser = get_browser_setting(
        "use_own_browser", False
    )  # Logic handled by CDP/WSS presence
    keep_browser_open = get_browser_setting("keep_browser_open", False)
    headless = get_browser_setting("headless", False)
    disable_security = get_browser_setting("disable_security", False)
    window_w = int(get_browser_setting("window_w", 1280))
    window_h = int(get_browser_setting("window_h", 1100))
    cdp_url = get_browser_setting("cdp_url") or None
    wss_url = get_browser_setting("wss_url") or None
    save_recording_path = get_browser_setting("save_recording_path") or None
    save_trace_path = get_browser_setting("save_trace_path") or None
    save_agent_history_path = get_browser_setting(
        "save_agent_history_path", "./tmp/agent_history"
    )
    save_download_path = get_browser_setting("save_download_path", "./tmp/downloads")

    stream_vw = 70
    stream_vh = int(70 * window_h // window_w)

    os.makedirs(save_agent_history_path, exist_ok=True)
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
    if save_trace_path:
        os.makedirs(save_trace_path, exist_ok=True)
    if save_download_path:
        os.makedirs(save_download_path, exist_ok=True)

    # --- 2. Initialize LLM ---
    main_llm = await initialize_llm(
        llm_provider_name,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        ollama_num_ctx if llm_provider_name == "ollama" else None,
    )

    # Pass the webui_manager instance to the callback when wrapping it
    async def ask_callback_wrapper(
            query: str, browser_context: BrowserContext
    ) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    async def confirm_callback_wrapper(
            action_name: str, params: Dict[str, Any], browser_context: BrowserContext
    ) -> str:
        return await _confirm_action_callback(webui_manager, action_name, params, browser_context)

    if not webui_manager.bu_controller:
        webui_manager.bu_controller = CustomController(
            ask_assistant_callback=ask_callback_wrapper,
            confirm_action_callback=confirm_callback_wrapper
        )
        webui_manager.bu_controller.set_fast_mode(fast_mode)
        await webui_manager.bu_controller.setup_mcp_client(mcp_server_config)
    else:
        # Update callbacks if controller exists (in case webui_manager reference changed or just to be safe)
        webui_manager.bu_controller.ask_assistant_callback = ask_callback_wrapper
        webui_manager.bu_controller.confirm_action_callback = confirm_callback_wrapper

    # Set memory file in controller (Update existing or new controller)
    webui_manager.bu_controller.set_memory_file(memory_file if memory_file else None)
    
    # Reset loop detection history
    webui_manager.bu_controller.reset_loop_history()
    webui_manager.bu_controller.set_fast_mode(fast_mode)
    webui_manager.bu_controller.set_require_confirmation(require_confirmation)

    # --- 4. Initialize Browser and Context ---
    should_close_browser_on_finish = not keep_browser_open

    try:
        # Close existing resources if not keeping open
        if not keep_browser_open:
            if webui_manager.bu_browser_context:
                logger.info("Closing previous browser context.")
                await webui_manager.bu_browser_context.close()
                webui_manager.bu_browser_context = None
            if webui_manager.bu_browser:
                logger.info("Closing previous browser.")
                await webui_manager.bu_browser.close()
                webui_manager.bu_browser = None

        # Create Browser if needed
        if not webui_manager.bu_browser:
            logger.info("Launching new browser instance.")
            extra_args = []
            if use_own_browser:
                browser_binary_path = os.getenv("BROWSER_PATH", None) or browser_binary_path
                if browser_binary_path == "":
                    browser_binary_path = None
                browser_user_data = browser_user_data_dir or os.getenv("BROWSER_USER_DATA", None)
                if browser_user_data:
                    extra_args += [f"--user-data-dir={browser_user_data}"]
            else:
                browser_binary_path = None

            webui_manager.bu_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    browser_binary_path=browser_binary_path,
                    extra_browser_args=extra_args,
                    wss_url=wss_url,
                    cdp_url=cdp_url,
                    new_context_config=BrowserContextConfig(
                        window_width=window_w,
                        window_height=window_h,
                    )
                )
            )

        # Create Context if needed
        if not webui_manager.bu_browser_context:
            logger.info("Creating new browser context.")
            context_config = BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path
                if save_recording_path
                else None,
                save_downloads_path=save_download_path if save_download_path else None,
                window_height=window_h,
                window_width=window_w,
            )
            if not webui_manager.bu_browser:
                raise ValueError("Browser not initialized, cannot create context.")
            webui_manager.bu_browser_context = (
                await webui_manager.bu_browser.new_context(config=context_config)
            )

        # --- 5. Initialize or Update Agent ---
        webui_manager.bu_agent_task_id = str(uuid.uuid4())  # New ID for this task run
        os.makedirs(
            os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id),
            exist_ok=True,
        )
        history_file = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.json",
        )
        gif_path = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.gif",
        )

        # Pass the webui_manager to callbacks when wrapping them
        async def step_callback_wrapper(
                state: BrowserState, output: AgentOutput, step_num: int
        ):
            await _handle_new_step(webui_manager, state, output, step_num)

        def done_callback_wrapper(history: AgentHistoryList):
            _handle_done(webui_manager, history)

        if not webui_manager.bu_agent:
            logger.info(f"Initializing new agent for task: {task}")
            if not webui_manager.bu_browser or not webui_manager.bu_browser_context:
                raise ValueError(
                    "Browser or Context not initialized, cannot create agent."
                )
            webui_manager.bu_agent = BrowserUseAgent(
                task=task,
                llm=main_llm,
                browser=webui_manager.bu_browser,
                browser_context=webui_manager.bu_browser_context,
                controller=webui_manager.bu_controller,
                register_new_step_callback=step_callback_wrapper,
                register_done_callback=done_callback_wrapper,
                use_vision=use_vision,
                override_system_message=override_system_prompt,
                extend_system_message=extend_system_prompt,
                max_input_tokens=max_input_tokens,
                max_actions_per_step=max_actions,
                tool_calling_method=tool_calling_method,
                planner_llm=planner_llm,
                use_vision_for_planner=planner_use_vision if planner_llm else False,
                confirmer_llm=confirmer_llm,
                confirmer_strictness=confirmer_strictness,
                use_vision_for_confirmer=confirmer_use_vision,
                smart_retry_llm=smart_retry_llm,
                auto_save_on_stuck=auto_save_on_stuck,
                cheap_llm=cheap_llm,
                enable_cost_saver=enable_cost_saver,
                source="webui",
                history=getattr(webui_manager, "bu_resumed_history", None)
            )
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.settings.generate_gif = gif_path
        else:
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.add_new_task(task)
            webui_manager.bu_agent.settings.generate_gif = gif_path
            webui_manager.bu_agent.browser = webui_manager.bu_browser
            webui_manager.bu_agent.browser_context = webui_manager.bu_browser_context
            webui_manager.bu_agent.controller = webui_manager.bu_controller

        # Clear resumed history after using it
        webui_manager.bu_resumed_history = None

        # --- 6. Run Agent Task and Stream Updates ---
        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task  # Store the task
        webui_manager.bu_latest_screenshot = None # Reset screenshot

        # --- Start Background Screenshot Task ---
        async def screenshot_loop():
            while not agent_task.done():
                if headless and webui_manager.bu_browser_context:
                    try:
                        webui_manager.bu_latest_screenshot = await webui_manager.bu_browser_context.take_screenshot()
                    except Exception: pass
                
                # Dynamic sleep based on state to save resources
                sleep_time = 0.1
                if webui_manager.bu_agent and webui_manager.bu_agent.state.paused:
                    sleep_time = 1.0
                await asyncio.sleep(sleep_time)
        screenshot_task = asyncio.create_task(screenshot_loop())

        last_chat_len = len(webui_manager.bu_chat_history)
        last_status = webui_manager.bu_agent_status
        last_screenshot_time = 0.0
        last_screenshot_b64 = None
        while not agent_task.done():
            is_paused = webui_manager.bu_agent.state.paused
            is_stopped = webui_manager.bu_agent.state.stopped

            # Check for pause state
            if is_paused:
                yield {
                    pause_resume_button_comp: gr.update(
                        value="▶️ Resume", interactive=True
                    ),
                    stop_button_comp: gr.update(interactive=True),
                }
                # Wait until pause is released or task is stopped/done
                while is_paused and not agent_task.done():
                    # Re-check agent state in loop
                    is_paused = webui_manager.bu_agent.state.paused
                    is_stopped = webui_manager.bu_agent.state.stopped
                    if is_stopped:  # Stop signal received while paused
                        break
                    await asyncio.sleep(0.2)

                if (
                        agent_task.done() or is_stopped
                ):  # If stopped or task finished while paused
                    break

                # If resumed, yield UI update
                yield {
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=True
                    ),
                    run_button_comp: gr.update(
                        value="⬆️ Add Instruction", interactive=True, variant="secondary"
                    ),
                }

            # Check if agent stopped itself or stop button was pressed (which sets agent.state.stopped)
            if is_stopped:
                logger.info("Agent has stopped (internally or via stop button).")
                if not agent_task.done():
                    # Ensure the task coroutine finishes if agent just set flag
                    try:
                        await asyncio.wait_for(
                            agent_task, timeout=1.0
                        )  # Give it a moment to exit run()
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Agent task did not finish quickly after stop signal, cancelling."
                        )
                        agent_task.cancel()
                    except Exception:  # Catch task exceptions if it errors on stop
                        pass
                break  # Exit the streaming loop

            # Check if agent is asking for help (via response_event)
            update_dict = {}
            if webui_manager.bu_response_event is not None:
                update_dict = {
                    user_input_comp: gr.update(
                        placeholder="Agent needs help. Enter response and submit.",
                        interactive=True,
                    ),
                    run_button_comp: gr.update(
                        value="✔️ Submit Response", interactive=True
                    ),
                    retry_button_comp: gr.update(interactive=False),
                    pause_resume_button_comp: gr.update(interactive=False),
                    stop_button_comp: gr.update(interactive=False),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
                last_chat_len = len(webui_manager.bu_chat_history)
                yield update_dict
                # Wait until response is submitted or task finishes
                await webui_manager.bu_response_event.wait()

                # Restore UI after response submitted or if task ended unexpectedly
                if not agent_task.done():
                    yield {
                        user_input_comp: gr.update(
                            placeholder="Agent is running... Enter text to steer/add instructions.", interactive=True
                        ),
                        run_button_comp: gr.update(
                            value="⬆️ Add Instruction", interactive=True, variant="secondary"
                        ),
                        retry_button_comp: gr.update(interactive=False),
                        pause_resume_button_comp: gr.update(interactive=True),
                        stop_button_comp: gr.update(interactive=True),
                    }
                else:
                    break  # Task finished while waiting for response

            # Update Chatbot if new messages arrived via callbacks
            if len(webui_manager.bu_chat_history) > last_chat_len:
                update_dict[chatbot_comp] = gr.update(
                    value=webui_manager.bu_chat_history
                )
                last_chat_len = len(webui_manager.bu_chat_history)

            # Update Status
            if webui_manager.bu_agent_status != last_status:
                update_dict[agent_status_comp] = gr.update(value=webui_manager.bu_agent_status)
                last_status = webui_manager.bu_agent_status
            
            # Check for validating state
            if getattr(webui_manager.bu_agent, "is_validating", False):
                update_dict[agent_status_comp] = gr.update(value=webui_manager.bu_agent_status + "\n\n🔍 **Validating Output with Confirmer LLM...**")
            
            # Check for smart retry state update in real-time
            if getattr(webui_manager.bu_agent, "switched_to_retry_model", False) and "Smart Retry Active" not in webui_manager.bu_agent_status:
                 webui_manager.bu_agent_status = "<span class='retry-badge'>🧠 Smart Retry Active</span>\n" + webui_manager.bu_agent_status
            
            # Check for cost saver state update in real-time
            if getattr(webui_manager.bu_agent, "using_cheap_model", False) and "Cost Saver Active" not in webui_manager.bu_agent_status:
                 webui_manager.bu_agent_status = "<span class='retry-badge' style='background-color: #10b981;'>💰 Cost Saver Active</span>\n" + webui_manager.bu_agent_status

            # Update Browser View
            if headless and webui_manager.bu_browser_context:
                screenshot_b64 = webui_manager.bu_latest_screenshot
                if screenshot_b64:
                    if screenshot_b64 != last_screenshot_b64:
                        last_screenshot_b64 = screenshot_b64
                        html_content = f'<img src="data:image/jpeg;base64,{screenshot_b64}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                elif last_screenshot_b64 != "waiting":
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                    update_dict[browser_view_comp] = gr.update(
                        value=html_content, visible=True
                    )
                    last_screenshot_b64 = "waiting"
            else:
                if last_screenshot_b64 is not None:
                    update_dict[browser_view_comp] = gr.update(visible=False)
                    last_screenshot_b64 = None

            # Yield accumulated updates
            if update_dict:
                yield update_dict

            await asyncio.sleep(0.05)  # Polling interval

        # --- 7. Task Finalization ---
        webui_manager.bu_agent.state.paused = False
        webui_manager.bu_agent.state.stopped = False
        final_update = {}
        try:
            logger.info("Agent task completing...")
            # Await the task ensure completion and catch exceptions if not already caught
            if not agent_task.done():
                await agent_task  # Retrieve result/exception
            elif agent_task.exception():  # Check if task finished with exception
                agent_task.result()  # Raise the exception to be caught below
            logger.info("Agent task completed processing.")

            logger.info(f"Explicitly saving agent history to: {history_file}")
            webui_manager.bu_agent.save_history(history_file)

            if os.path.exists(history_file):
                final_update[history_file_comp] = gr.File(value=history_file)

            if gif_path and os.path.exists(gif_path):
                logger.info(f"GIF found at: {gif_path}")
                final_update[gif_comp] = gr.Image(value=gif_path)

        except asyncio.CancelledError:
            logger.info("Agent task was cancelled.")
            if not any(
                    "Cancelled" in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": "**Task Cancelled**."}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = (
                f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            )
            if not any(
                    error_message in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
            gr.Error(f"Agent execution failed: {e}")

        finally:
            webui_manager.bu_current_task = None  # Clear the task reference
            
            if not screenshot_task.done():
                screenshot_task.cancel()

            # Close browser/context if requested
            if should_close_browser_on_finish:
                if webui_manager.bu_browser_context:
                    logger.info("Closing browser context after task.")
                    await webui_manager.bu_browser_context.close()
                    webui_manager.bu_browser_context = None
                if webui_manager.bu_browser:
                    logger.info("Closing browser after task.")
                    await webui_manager.bu_browser.close()
                    webui_manager.bu_browser = None

            # --- 8. Final UI Update ---
            final_update.update(
                {
                    user_input_comp: gr.update(
                        value="",
                        interactive=True,
                        placeholder="Enter your next task...",
                    ),
                    run_button_comp: gr.update(value="▶️ Submit Task", interactive=True, variant="primary"),
                    retry_button_comp: gr.update(interactive=True),
                    stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=False
                    ),
                    clear_button_comp: gr.update(interactive=True),
                    # Ensure final chat history is shown
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                    agent_status_comp: gr.update(value=webui_manager.bu_agent_status),
                }
            )
            yield final_update

    except Exception as e:
        # Catch errors during setup (before agent run starts)
        logger.error(f"Error setting up agent task: {e}", exc_info=True)
        webui_manager.bu_current_task = None  # Ensure state is reset
        yield {
            user_input_comp: gr.update(
                interactive=True, placeholder="Error during setup. Enter task..."
            ),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True, variant="primary"),
            retry_button_comp: gr.update(interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
            clear_button_comp: gr.update(interactive=True),
            chatbot_comp: gr.update(
                value=webui_manager.bu_chat_history
                      + [{"role": "assistant", "content": f"**Setup Error:** {e}"}]
            ),
        }


# --- Button Click Handlers --- (Need access to webui_manager)


async def handle_submit(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the main 'Submit' button."""
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    user_input_value = components.get(user_input_comp, "").strip()

    # Check if waiting for user assistance
    if webui_manager.bu_response_event and not webui_manager.bu_response_event.is_set():
        logger.info(f"User submitted assistance: {user_input_value}")
        webui_manager.bu_user_help_response = (
            user_input_value if user_input_value else "User provided no text response."
        )
        webui_manager.bu_response_event.set()
        # UI updates handled by the main loop reacting to the event being set
        yield {
            user_input_comp: gr.update(
                value="",
                interactive=False,
                placeholder="Waiting for agent to continue...",
            ),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
        }
    # Check if a task is currently running (using _current_task)
    elif webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        # Handle steering/adding instructions while running
        if user_input_value:
            logger.info(f"Steering agent with message: {user_input_value}")
            webui_manager.bu_chat_history.append({"role": "user", "content": user_input_value})
            
            # Fire and forget the injection to avoid blocking UI
            async def inject_task_async():
                try:
                    agent = webui_manager.bu_agent
                    # Attempt to inject message into agent's message manager
                    # Use add_new_task if available as it ensures the agent treats this as a new instruction/goal
                    if hasattr(agent, "add_new_task"):
                        agent.add_new_task(user_input_value)
                        logger.info(f"Steering instruction injected via add_new_task: {user_input_value}")
                    elif hasattr(agent, "agent") and hasattr(agent.agent, "add_new_task"):
                        agent.agent.add_new_task(user_input_value)
                        logger.info(f"Steering instruction injected via agent.agent.add_new_task: {user_input_value}")
                    elif hasattr(agent, "message_manager"):
                        agent.message_manager.add_user_message(user_input_value)
                    elif hasattr(agent, "agent") and hasattr(agent.agent, "message_manager"):
                        agent.agent.message_manager.add_user_message(user_input_value)
                    else:
                        logger.warning("Agent does not support dynamic steering (message_manager not found).")
                except Exception as e:
                    logger.error(f"Error steering agent: {e}")
            
            asyncio.create_task(inject_task_async())
            
            yield {
                user_input_comp: gr.update(value=""),
                webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(value=webui_manager.bu_chat_history)
            }
        else:
            yield {} # No change if empty input
    else:
        # Handle submission for a new task
        logger.info("Submit button clicked for new task.")
        # Use async generator to stream updates from run_agent_task
        async for update in run_agent_task(webui_manager, components):
            yield update

async def handle_retry(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the 'Retry' button."""
    if not webui_manager.bu_last_task_prompt:
        gr.Warning("No previous task to retry.")
        yield {}
        return

    # Inject the last task into the components dictionary so run_agent_task sees it
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    components[user_input_comp] = webui_manager.bu_last_task_prompt
    
    logger.info(f"Retrying task: {webui_manager.bu_last_task_prompt}")
    
    # Delegate to run_agent_task
    async for update in run_agent_task(webui_manager, components):
        yield update

async def handle_stop(webui_manager: WebuiManager):
    """Handles clicks on the 'Stop' button."""
    logger.info("Stop button clicked.")
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        # Signal the agent to stop by setting its internal flag
        agent.state.stopped = True
        agent.state.paused = False  # Ensure not paused if stopped
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False, value="⏹️ Stopping..."),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=False),
        }
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")
        # Reset UI just in case it's stuck
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=True, value="▶️ Submit Task", variant="primary"),
            webui_manager.get_component_by_id(
                "browser_use_agent.retry_button"
            ): gr.update(interactive=True),
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.clear_button"
            ): gr.update(interactive=True),
        }


async def handle_pause_resume(webui_manager: WebuiManager):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        if agent.state.paused:
            logger.info("Resume button clicked.")
            agent.resume()
            # UI update happens in main loop
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="⏸️ Pause", interactive=True)
            }  # Optimistic update
        else:
            logger.info("Pause button clicked.")
            agent.pause()
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="▶️ Resume", interactive=True)
            }  # Optimistic update
    else:
        logger.warning(
            "Pause/Resume clicked but agent is not running or doesn't support state."
        )
        return {}  # No change


async def handle_reload_memory(webui_manager: WebuiManager, memory_file_path: str):
    """Handles clicks on the 'Reload Memory' button."""
    logger.info(f"Reload Memory button clicked for path: {memory_file_path}")
    memory_content_comp = webui_manager.get_component_by_id("browser_use_agent.memory_content")
    
    if not memory_file_path:
        return {memory_content_comp: gr.update(value="No memory file path provided.")}
        
    if not os.path.exists(memory_file_path):
        return {memory_content_comp: gr.update(value="Memory file does not exist yet.")}

    try:
        with open(memory_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {memory_content_comp: gr.update(value=content)}
    except Exception as e:
        logger.error(f"Failed to read memory file: {e}")
        return {memory_content_comp: gr.update(value=f"Error reading file: {e}")}

async def handle_update_kb_list(memory_file_path: str):
    """Scans the directory of the memory file for other knowledge bases."""
    if not memory_file_path:
        return gr.update(choices=[])
    
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    if not os.path.exists(base_dir):
        return gr.update(choices=[])
    
    try:
        files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and (f.endswith('.md') or f.endswith('.txt'))]
        files.sort()
        return gr.update(choices=files)
    except Exception as e:
        logger.error(f"Error scanning KB directory: {e}")
        return gr.update(choices=[])

async def handle_load_kb_file(webui_manager: WebuiManager, memory_file_path: str, selected_file: str):
    """Loads content of the selected KB file."""
    memory_content_comp = webui_manager.get_component_by_id("browser_use_agent.memory_content")
    if not memory_file_path or not selected_file:
        return {memory_content_comp: gr.update(value="")}
        
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    filepath = os.path.join(base_dir, selected_file)
    
    if not os.path.exists(filepath):
        return {memory_content_comp: gr.update(value="File not found.")}
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {memory_content_comp: gr.update(value=content)}
    except Exception as e:
        return {memory_content_comp: gr.update(value=f"Error reading file: {e}")}

async def handle_save_chat(webui_manager: WebuiManager):
    """Handles clicks on the 'Save Chat' button."""
    logger.info("Save Chat button clicked.")

    history = webui_manager.bu_chat_history
    if not history:
        logger.info("No chat history to save.")
        return {}

    # Default path
    save_path = "./tmp/chat_logs"
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_log_{timestamp}.json"
    full_path = os.path.join(save_path, filename)

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        logger.info(f"Chat history saved to {full_path}")
        return {webui_manager.get_component_by_id("browser_use_agent.chat_log_file"): gr.File(value=full_path, visible=True)}
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
        return {}

async def handle_clear(webui_manager: WebuiManager):
    """Handles clicks on the 'Clear' button."""
    logger.info("Clear button clicked.")

    # Stop any running task first
    task = webui_manager.bu_current_task
    if task and not task.done():
        logger.info("Clearing requires stopping the current task.")
        webui_manager.bu_agent.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)  # Wait briefly
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error stopping task on clear: {e}")
    webui_manager.bu_current_task = None

    if webui_manager.bu_controller:
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None
    webui_manager.bu_agent = None

    # Reset state stored in manager
    webui_manager.bu_chat_history = []
    webui_manager.bu_response_event = None
    webui_manager.bu_user_help_response = None
    webui_manager.bu_agent_task_id = None
    webui_manager.bu_agent_status = "### Agent Status\nReady"
    webui_manager.bu_last_task_prompt = None
    webui_manager.bu_resumed_history = None

    logger.info("Agent state and browser resources cleared.")

    # Reset UI components
    return {
        webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(
            value=[]
        ),
        webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(
            value="", placeholder="Enter your task here..."
        ),
        webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(
            value=webui_manager.bu_agent_status
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.agent_history_file"
        ): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.recording_gif"): gr.update(
            value=None
        ),
        webui_manager.get_component_by_id("browser_use_agent.browser_view"): gr.update(
            value="<div style='...'>Browser Cleared</div>"
        ),
        webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(
            value="▶️ Submit Task", interactive=True, variant="primary"
        ),
        webui_manager.get_component_by_id("browser_use_agent.retry_button"): gr.update(
            interactive=False
        ),
        webui_manager.get_component_by_id("browser_use_agent.stop_button"): gr.update(
            interactive=False
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.pause_resume_button"
        ): gr.update(value="⏸️ Pause", interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.clear_button"): gr.update(
            interactive=True
        ),
        webui_manager.get_component_by_id("browser_use_agent.chat_log_file"): gr.update(
            value=None
        ),
    }


async def handle_refresh_history_files(save_agent_history_path: str):
    """Refreshes the list of available history files."""
    if not os.path.exists(save_agent_history_path):
        return gr.update(choices=[])
    
    files = []
    for task_id in os.listdir(save_agent_history_path):
        task_dir = os.path.join(save_agent_history_path, task_id)
        if os.path.isdir(task_dir):
            json_file = os.path.join(task_dir, f"{task_id}.json")
            if os.path.exists(json_file):
                # Use modification time for sorting
                mtime = os.path.getmtime(json_file)
                files.append((json_file, mtime))
    
    # Sort by mtime descending
    files.sort(key=lambda x: x[1], reverse=True)
    choices = [f[0] for f in files]
    return gr.update(choices=choices)


# --- Tab Creation Function ---


def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """
    Create the run agent tab, defining UI, state, and handlers.
    """
    webui_manager.init_browser_use_agent()

    # --- Define UI Components ---
    tab_components = {}
    
    with gr.Row(equal_height=False):
        # Left Column: Chat & Controls
        with gr.Column(scale=3):
            agent_status = gr.Markdown(value="### Agent Status\nReady", label="Agent Status")
            chatbot = gr.Chatbot(
                lambda: webui_manager.bu_chat_history,  # Load history dynamically
                elem_id="browser_use_chatbot",
                label="Agent Interaction",
                type="messages",
                height=600,
                show_copy_button=True,
            )
            user_input = gr.Textbox(
                label="Your Task or Response",
                placeholder="Enter your task here or provide assistance when asked.",
                lines=3,
                interactive=True,
                elem_id="user_input",
            )
            with gr.Row():
                stop_button = gr.Button(
                    "⏹️ Stop", interactive=False, variant="stop", scale=1
                )
                pause_resume_button = gr.Button(
                    "⏸️ Pause", interactive=False, variant="secondary", scale=1, visible=True
                )
                clear_button = gr.Button(
                    "🗑️ Clear", interactive=True, variant="secondary", scale=1
                )
                save_chat_button = gr.Button(
                    "💾 Save Chat", interactive=True, variant="secondary", scale=1
                )
                retry_button = gr.Button(
                    "🔄 Retry", interactive=True, variant="secondary", scale=1
                )
                run_button = gr.Button("▶️ Submit Task", variant="primary", scale=2)
            
            with gr.Accordion("🧠 Brain & Memory", open=False):
                with gr.Row():
                    brain_file = gr.Textbox(label="Brain File (System Prompt)", value="./brain.md", placeholder="Path to markdown file")
                    memory_file = gr.Textbox(label="Memory File (Read/Write)", value="./memory.txt", placeholder="Path to text file")
                with gr.Row():
                    kb_files_dropdown = gr.Dropdown(label="Knowledge Base Files", choices=[], value=None, interactive=True, scale=3)
                    refresh_kb_btn = gr.Button("🔄 Scan Files", variant="secondary", scale=1)
                    load_kb_btn = gr.Button("📂 Load Content", variant="secondary", scale=1)
                memory_content = gr.TextArea(label="Memory Content", interactive=False, lines=10)
            
            with gr.Accordion("📂 Session Management", open=False):
                with gr.Row():
                    history_files_dropdown = gr.Dropdown(label="Saved Sessions", choices=[], interactive=True, scale=3)
                    refresh_history_btn = gr.Button("🔄 Scan", variant="secondary", scale=1)
                    resume_session_btn = gr.Button("📂 Resume Session", variant="secondary", scale=1)

        # Right Column: Browser View & Outputs
        with gr.Column(scale=2):
            browser_view = gr.HTML(
                value="<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",
                label="Browser Live View",
                elem_id="browser_view",
                visible=False,
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
            memory_content=memory_content,
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
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(
            *args
    ) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_submit that yields its results."""
        components_dict = dict(zip(all_managed_components, args))
        async for update in handle_submit(webui_manager, components_dict):
            yield update

    async def retry_wrapper(
            *args
    ) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_retry."""
        components_dict = dict(zip(all_managed_components, args))
        async for update in handle_retry(webui_manager, components_dict):
            yield update

    async def stop_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_stop."""
        update_dict = await handle_stop(webui_manager)
        yield update_dict

    async def pause_resume_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_pause_resume."""
        update_dict = await handle_pause_resume(webui_manager)
        yield update_dict

    async def clear_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_clear."""
        update_dict = await handle_clear(webui_manager)
        yield update_dict

    async def save_chat_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_save_chat."""
        update_dict = await handle_save_chat(webui_manager)
        yield update_dict

    async def reload_memory_wrapper(path: str) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_reload_memory."""
        update_dict = await handle_reload_memory(webui_manager, path)
        yield update_dict

    async def update_kb_list_wrapper(path: str) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_update_kb_list."""
        update = await handle_update_kb_list(path)
        yield {kb_files_dropdown: update}

    async def load_kb_file_wrapper(path: str, filename: str) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_load_kb_file."""
        update_dict = await handle_load_kb_file(webui_manager, path, filename)
        yield update_dict

    async def refresh_history_wrapper(path: str) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_refresh_history_files."""
        update = await handle_refresh_history_files(path)
        yield {history_files_dropdown: update}

    async def resume_session_wrapper(history_file: str, *args) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_resume_session."""
        components_dict = dict(zip(all_managed_components, args))
        async for update in handle_resume_session(webui_manager, history_file, components_dict):
            yield update

    # --- Connect Event Handlers using the Wrappers --
    run_button.click(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs, trigger_mode="multiple"
    )
    user_input.submit(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs
    )
    retry_button.click(fn=retry_wrapper, inputs=all_managed_components, outputs=run_tab_outputs)
    stop_button.click(fn=stop_wrapper, inputs=None, outputs=run_tab_outputs)
    pause_resume_button.click(
        fn=pause_resume_wrapper, inputs=None, outputs=run_tab_outputs
    )
    save_chat_button.click(fn=save_chat_wrapper, inputs=None, outputs=run_tab_outputs)
    clear_button.click(fn=clear_wrapper, inputs=None, outputs=run_tab_outputs)
    refresh_kb_btn.click(fn=update_kb_list_wrapper, inputs=[memory_file], outputs=[kb_files_dropdown])
    load_kb_btn.click(fn=load_kb_file_wrapper, inputs=[memory_file, kb_files_dropdown], outputs=run_tab_outputs)
    refresh_history_btn.click(fn=refresh_history_wrapper, inputs=[agent_history_file], outputs=[history_files_dropdown]) # Note: agent_history_file is the output file component, we need the path setting.
    # We need the save path setting. It's not directly exposed as a component input here easily without getting it from settings.
    # Let's use a lambda to get it from the manager's components or just pass the default path for now, or better, bind to the settings component if possible.
    # Actually, we can use the `save_agent_history_path` component from browser settings if we can find it.
    # But it's in another tab.
    # Let's just use a hardcoded path or try to find it.
    # Better: The wrapper can look it up.
    
    # Re-binding refresh_history_btn to use a function that looks up the path
    async def refresh_history_click_handler(*args):
        # Find the save path from components
        components_dict = dict(zip(all_managed_components, args))
        # We need to find the component ID "browser_settings.save_agent_history_path"
        save_path_comp = webui_manager.id_to_component.get("browser_settings.save_agent_history_path")
        save_path = components_dict.get(save_path_comp, "./tmp/agent_history")
        update = await handle_refresh_history_files(save_path)
        yield {history_files_dropdown: update}

    refresh_history_btn.click(fn=refresh_history_click_handler, inputs=all_managed_components, outputs=[history_files_dropdown])
    resume_session_btn.click(fn=resume_session_wrapper, inputs=[history_files_dropdown] + all_managed_components, outputs=run_tab_outputs)
