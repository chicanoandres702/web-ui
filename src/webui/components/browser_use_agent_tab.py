import asyncio
import json
import logging
import os
import uuid
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr

# from browser_use.agent.service import Agent
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils.browser_factory import create_browser, create_context
from src.controller.custom_controller import CustomController
from src.webui.llm_helper import initialize_llm
from src.webui.components.shared import create_agent_control_buttons, get_agent_settings_values, get_browser_settings_values, safe_execution, initialize_agent_llms, read_text_file, save_text_file, get_valid_input_components, map_dict_to_gradio_outputs, format_agent_output, render_plan_markdown, process_knowledge_generation
from src.webui.webui_manager import WebuiManager
from src.webui.components.agent_logic import initialize_browser_infrastructure, configure_controller, construct_agent, prepare_directories

from src.utils.prompts import FULL_SYSTEM_PROMPT # Still needed for memory loading logic if kept here
from src.utils.utils import get_progress_bar_html, parse_agent_thought
from src.utils.prompt_library import get_all_prompts, save_custom_prompt, delete_custom_prompt, get_prompt_by_name
from src.webui.components.knowledge_base_logic import list_kb_files, load_kb_content
logger = logging.getLogger(__name__)

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
    formatted_output = format_agent_output(output)  # Use the updated function

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
        
        # We need to yield UI updates to show we are loading
        yield {
            webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(value="### Agent Status\nResuming Session...")
        }
        
        webui_manager.bu_chat_history = []
        
        # Replay history to chat
        for i, item in enumerate(history.history):
            step_num = i + 1
            state = item.state
            output = item.model_output
            
            formatted_output = format_agent_output(output)
            step_header = f"--- **Step {step_num} (Restored)** ---"
            
            screenshot_html = ""
            if state.screenshot:
                 screenshot_html = f'<img src="data:image/jpeg;base64,{state.screenshot}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" /><br/>'
            
            final_content = step_header + "<br/>" + screenshot_html + formatted_output
            webui_manager.bu_chat_history.append({"role": "assistant", "content": final_content.strip()})
            
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
    plan_status_comp = webui_manager.get_component_by_id("browser_use_agent.plan_status")
    plan_editor_comp = webui_manager.get_component_by_id("browser_use_agent.plan_editor")
    plan_editor_acc_comp = webui_manager.get_component_by_id("browser_use_agent.plan_editor_accordion")
    kb_gen_acc_comp = webui_manager.get_component_by_id("browser_use_agent.kb_gen_accordion")
    kb_gen_title_comp = webui_manager.get_component_by_id("browser_use_agent.kb_gen_title")
    kb_gen_content_comp = webui_manager.get_component_by_id("browser_use_agent.kb_gen_content")
    kb_files_dropdown_comp = webui_manager.get_component_by_id("browser_use_agent.kb_files_dropdown")
    extraction_model_comp = webui_manager.get_component_by_id("browser_use_agent.extraction_model_dropdown")
    
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
        user_input_comp: gr.update(
            value="", interactive=True, placeholder="Agent is running... Enter text to steer/add instructions."
        ),
        run_button_comp: gr.update(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
        retry_button_comp: gr.update(interactive=False),
        stop_button_comp: gr.update(interactive=True),
        pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=True),
        clear_button_comp: gr.update(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        agent_status_comp: gr.update(value=webui_manager.bu_agent_status),
        plan_status_comp: gr.update(value="", visible=False),
        plan_editor_comp: gr.update(value=None),
        plan_editor_acc_comp: gr.update(visible=False),
        kb_gen_acc_comp: gr.update(visible=False, open=False),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None),
    }

    # --- Retrieve Settings ---
    agent_settings = get_agent_settings_values(webui_manager, components)
    browser_settings = get_browser_settings_values(webui_manager, components)

    max_steps = agent_settings["max_steps"]
    webui_manager.bu_max_steps = max_steps
    
    if agent_settings["fast_mode"]:
        webui_manager.bu_agent_status += "\n⚡ **Fast Mode Active**"

    # --- Brain & Memory ---
    brain_file = components.get(brain_file_comp)
    memory_file = components.get(memory_file_comp)
    
    # Load file content into settings
    if brain_file and os.path.exists(brain_file):
        try:
            with open(brain_file, "r", encoding="utf-8") as f:
                agent_settings["override_system_prompt"] = f.read()
            logger.info(f"Loaded Brain from {brain_file}")
        except Exception as e:
            logger.error(f"Failed to load Brain file: {e}")

    # Ensure memory directory and file exist to prevent errors
    if memory_file:
        try:
            mem_path = os.path.abspath(memory_file)
            os.makedirs(os.path.dirname(mem_path), exist_ok=True)
            if not os.path.exists(mem_path):
                with open(mem_path, 'w', encoding='utf-8') as f:
                    f.write("")
        except Exception as e:
            logger.error(f"Error initializing memory file: {e}")

    if memory_file and os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memory_content = f.read()
            if memory_content.strip():
                agent_settings["extend_system_prompt"] = (agent_settings["extend_system_prompt"] or "") + f"\n\nLONG TERM MEMORY:\n{memory_content}"
            logger.info(f"Loaded Memory from {memory_file}")
        except Exception as e:
            logger.error(f"Failed to load Memory file: {e}")
    
    # --- Initialize LLMs ---
    main_llm, planner_llm, confirmer_llm, smart_retry_llm, cheap_llm = await initialize_agent_llms(agent_settings)
    llms = (main_llm, planner_llm, confirmer_llm, smart_retry_llm, cheap_llm)

    # Reset plan state
    webui_manager.bu_plan = []

    # --- Hierarchical Planning ---
    if agent_settings["enable_hierarchical_planning"]:
        webui_manager.bu_agent_status = "### Agent Status\n🧠 Generating Plan..."
        yield {agent_status_comp: gr.update(value=webui_manager.bu_agent_status)}
        
        planning_llm = planner_llm if planner_llm else main_llm
        if planning_llm:
            from src.agent.planner import generate_hierarchical_plan
            plan = await generate_hierarchical_plan(planning_llm, task, agent_settings["planner_system_prompt"])
            
            if plan:
                # Initialize plan state for visualization
                webui_manager.bu_plan = [{"step": step, "status": "pending"} for step in plan]
                plan_md = render_plan_markdown(webui_manager.bu_plan)
                yield {
                    plan_status_comp: gr.update(value=plan_md, visible=True),
                    plan_editor_comp: gr.update(value=json.dumps(webui_manager.bu_plan, indent=2)),
                    plan_editor_acc_comp: gr.update(visible=True)
                }

                formatted_plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
                webui_manager.bu_chat_history.append({
                    "role": "assistant", 
                    "content": f"**🧠 Proposed Plan:**\n{formatted_plan_text}"
                })
                yield {chatbot_comp: gr.update(value=webui_manager.bu_chat_history)}
                
                task = f"Goal: {task}\n\nExecute the following plan step-by-step:\n{formatted_plan_text}\n\nIMPORTANT: Use the `update_plan_step(step_index, status)` tool to mark steps as 'in_progress' or 'completed' as you execute them to keep the user informed."
                webui_manager.bu_agent_status = "### Agent Status\nRunning Plan..."

    # --- Prepare Directories ---
    webui_manager.bu_agent_task_id = str(uuid.uuid4())
    history_file, gif_path = await prepare_directories(browser_settings, webui_manager.bu_agent_task_id)

    # Pass the webui_manager instance to the callback when wrapping it
    async def ask_callback_wrapper(
            query: str, browser_context: BrowserContext
    ) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    async def confirm_callback_wrapper(
            action_name: str, params: Dict[str, Any], browser_context: BrowserContext
    ) -> str:
        return await _confirm_action_callback(webui_manager, action_name, params, browser_context)

    async def update_plan_callback_wrapper(step_index: int, status: str):
        if hasattr(webui_manager, "bu_plan") and webui_manager.bu_plan:
            idx = step_index - 1
            if 0 <= idx < len(webui_manager.bu_plan):
                webui_manager.bu_plan[idx]["status"] = status
                webui_manager.bu_plan_updated = True
                
                # Auto-Pause Logic
                if status == "failed" and agent_settings.get("enable_auto_pause", False):
                    if webui_manager.bu_agent:
                        webui_manager.bu_agent.pause()

    # --- Configure Controller ---
    callbacks = {
        "ask_assistant": ask_callback_wrapper,
        "confirm_action": confirm_callback_wrapper,
        "update_plan": update_plan_callback_wrapper
    }
    await configure_controller(
        webui_manager, 
        agent_settings, 
        memory_file, 
        components.get(extraction_model_comp), 
        callbacks
    )

    # --- 4. Initialize Browser and Context ---
    should_close_browser_on_finish = not browser_settings["keep_browser_open"]

    try:
        await initialize_browser_infrastructure(webui_manager, browser_settings)

        # --- 5. Initialize or Update Agent ---
        # Pass the webui_manager to callbacks when wrapping them
        async def step_callback_wrapper(
                state: BrowserState, output: AgentOutput, step_num: int
        ):
            await _handle_new_step(webui_manager, state, output, step_num)

        def done_callback_wrapper(history: AgentHistoryList):
            _handle_done(webui_manager, history)

        agent_callbacks = {
            "step_callback": step_callback_wrapper,
            "done_callback": done_callback_wrapper
        }
        
        await construct_agent(
            webui_manager, 
            task, 
            agent_settings, 
            llms, 
            history_file, 
            gif_path, 
            agent_callbacks
        )

        # --- 6. Run Agent Task and Stream Updates ---
        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task  # Store the task
        webui_manager.bu_latest_screenshot = None # Reset screenshot

        # --- Start Background Screenshot Task ---
        stream_vw = 70
        stream_vh = int(70 * browser_settings["window_h"] // browser_settings["window_w"])
        async def screenshot_loop():
            while not agent_task.done():
                if browser_settings["headless"] and webui_manager.bu_browser_context:
                    try:
                        webui_manager.bu_latest_screenshot = await webui_manager.bu_browser_context.take_screenshot()
                    except Exception: pass
                
                # Dynamic sleep based on state to save resources
                sleep_time = 0.5
                if webui_manager.bu_agent and webui_manager.bu_agent.state.paused:
                    sleep_time = 1.0
                await asyncio.sleep(sleep_time)
        screenshot_task = asyncio.create_task(screenshot_loop()) if browser_settings["headless"] else None

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

            # Update Plan View
            if getattr(webui_manager, "bu_plan_updated", False):
                plan_md = render_plan_markdown(webui_manager.bu_plan)
                update_dict[plan_status_comp] = gr.update(value=plan_md)
                update_dict[plan_editor_comp] = gr.update(value=json.dumps(webui_manager.bu_plan, indent=2))
                webui_manager.bu_plan_updated = False

            # Update Browser View
            if browser_settings["headless"] and webui_manager.bu_browser_context:
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

            await asyncio.sleep(0.2)  # Polling interval - increased to prevent UI flooding

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

            # --- Knowledge Generation ---
            if main_llm and webui_manager.bu_agent.state.history.history:
                kb_result = await process_knowledge_generation(
                    webui_manager.bu_agent.state.history,
                    main_llm,
                    agent_settings["enable_kb_auto_save"],
                    memory_file
                )

                if kb_result["status"] == "saved":
                    webui_manager.bu_chat_history.append({"role": "assistant", "content": kb_result["message"]})
                    final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
                    # Auto-refresh the dropdown
                    if memory_file:
                        base_dir = os.path.dirname(memory_file)
                        try:
                            files = list_kb_files(base_dir)
                            final_update[kb_files_dropdown_comp] = gr.update(choices=files)
                        except Exception: pass
                elif kb_result["status"] == "generated":
                    final_update[kb_gen_title_comp] = gr.update(value=kb_result["title"])
                    final_update[kb_gen_content_comp] = gr.update(value=kb_result["content"])
                    final_update[kb_gen_acc_comp] = gr.update(visible=True, open=True)
                    webui_manager.bu_chat_history.append({"role": "assistant", "content": kb_result["message"]})
                    final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)


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
            
            if screenshot_task and not screenshot_task.done():
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
        # UI updates are handled by the main run loop polling the state
        return None
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")
        return None


async def handle_pause_resume(webui_manager: WebuiManager):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        if agent.state.paused:
            logger.info("Resume button clicked.")
            agent.resume()
            return None
        else:
            logger.info("Pause button clicked.")
            agent.pause()
            return None
    else:
        logger.warning(
            "Pause/Resume clicked but agent is not running or doesn't support state."
        )
        return None


async def handle_reload_memory(webui_manager: WebuiManager, memory_file_path: str):
    """Handles clicks on the 'Reload Memory' button."""
    logger.info(f"Reload Memory button clicked for path: {memory_file_path}")
    memory_content_comp = webui_manager.get_component_by_id("browser_use_agent.memory_content")
    
    if not memory_file_path:
        return {memory_content_comp: gr.update(value="No memory file path provided.")}
        
    if not os.path.exists(memory_file_path):
        return {memory_content_comp: gr.update(value="Memory file does not exist yet.")}

    content = read_text_file(memory_file_path)
    return {memory_content_comp: gr.update(value=content)}

async def handle_update_kb_list(memory_file_path: str):
    """Scans the directory of the memory file for other knowledge bases."""
    if not memory_file_path:
        return gr.update(choices=[])
    
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    files = list_kb_files(base_dir)
    # Ensure files is a list of strings to avoid [object Object] in UI
    if files and isinstance(files, list):
        return gr.update(choices=files)
    else:
        return gr.update(choices=[])

async def handle_load_kb_file(webui_manager: WebuiManager, memory_file_path: str, selected_file: str):
    """Loads content of the selected KB file."""
    memory_content_comp = webui_manager.get_component_by_id("browser_use_agent.memory_content")
    if not memory_file_path or not selected_file:
        return {memory_content_comp: gr.update(value="")}
        
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    content = load_kb_content(base_dir, selected_file)
    
    return {memory_content_comp: gr.update(value=content)}

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

async def handle_update_plan(webui_manager: WebuiManager, new_plan_json: str):
    """Updates the plan from the editor."""
    plan_status_comp = webui_manager.get_component_by_id("browser_use_agent.plan_status")
    
    if not new_plan_json:
        return {}
        
    try:
        new_plan = json.loads(new_plan_json)
        if isinstance(new_plan, list):
            webui_manager.bu_plan = new_plan
            webui_manager.bu_plan_updated = True
            
            # Notify agent of the change
            if webui_manager.bu_agent:
                msg = f"NOTE: The plan has been manually updated by the user. Current Plan Status: {json.dumps(new_plan)}"
                if hasattr(webui_manager.bu_agent, "message_manager"):
                     webui_manager.bu_agent.message_manager.add_user_message(msg)
            
            plan_md = render_plan_markdown(new_plan)
            return {
                plan_status_comp: gr.update(value=plan_md)
            }
        else:
            gr.Warning("Plan must be a JSON list.")
            return {}
    except Exception as e:
        gr.Warning(f"Invalid JSON: {e}")
        return {}

async def handle_save_generated_kb(webui_manager: WebuiManager, title: str, content: str, memory_file_path: str):
    """Saves the generated knowledge to a file."""
    if not title or not content:
        gr.Warning("Title or content is missing.")
        return {webui_manager.get_component_by_id("browser_use_agent.kb_gen_accordion"): gr.update(open=True)}
    
    base_dir = os.path.dirname(memory_file_path) if memory_file_path else "./tmp/memory"
    os.makedirs(base_dir, exist_ok=True)
    
    save_path = os.path.join(base_dir, title)
    if save_text_file(save_path, content):
        gr.Info(f"Saved knowledge to {title}")
        return {webui_manager.get_component_by_id("browser_use_agent.kb_gen_accordion"): gr.update(visible=False, open=False)}
    else:
        gr.Error("Failed to save file.")
        return {webui_manager.get_component_by_id("browser_use_agent.kb_gen_accordion"): gr.update(open=True)}

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
                extraction_model_dropdown = gr.Dropdown(label="Select Extraction Model", choices=[], value=None, interactive=True)
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

    async def refresh_history_wrapper(path: str) -> AsyncGenerator[List[Any], None]:
        """Wrapper for handle_refresh_history_files."""
        async def _logic():
            update = await handle_refresh_history_files(path)
            yield {history_files_dropdown: update}
        async for update in safe_execution(_logic):
            yield map_dict_to_gradio_outputs(update, [history_files_dropdown])

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
    refresh_history_btn.click(fn=refresh_history_wrapper, inputs=[agent_history_file], outputs=[history_files_dropdown]) # Note: agent_history_file is the output file component, we need the path setting.
    # We need the save path setting. It's not directly exposed as a component input here easily without getting it from settings.
    # Let's use a lambda to get it from the manager's components or just pass the default path for now, or better, bind to the settings component if possible.
    # Actually, we can use the `save_agent_history_path` component from browser settings if we can find it.
    # But it's in another tab.
    # Let's just use a hardcoded path or try to find it.
    # Better: The wrapper can look it up.
    
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