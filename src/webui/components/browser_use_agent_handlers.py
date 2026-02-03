import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

import gradio as gr
from browser_use.agent.views import AgentHistoryList, AgentOutput
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState

if TYPE_CHECKING:
    from src.webui.webui_manager import WebuiManager

from src.webui.components.agent_logic import initialize_browser_infrastructure, configure_controller, construct_agent, prepare_directories
from src.webui.components.shared import get_agent_settings_values, get_browser_settings_values, initialize_agent_llms, read_text_file, save_text_file, format_agent_output, render_plan_markdown, process_knowledge_generation
from src.utils.utils import get_progress_bar_html, parse_agent_thought
from src.webui.components.knowledge_base_logic import list_kb_files, load_kb_content

logger = logging.getLogger(__name__)

# --- Callbacks ---

async def _handle_new_step(
        webui_manager: "WebuiManager", state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback for each step taken by the agent, including screenshot display."""
    if not hasattr(webui_manager, "bu_chat_history"):
        webui_manager.bu_chat_history = []
    
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
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            if isinstance(screenshot_data, str) and len(screenshot_data) > 100:
                img_tag = f'<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'
                screenshot_html = img_tag + "<br/>"
            else:
                screenshot_html = "**[Invalid screenshot data]**<br/>"
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            screenshot_html = "**[Error displaying screenshot]**<br/>"

    # --- Format Agent Output ---
    formatted_output = format_agent_output(output)

    # --- Combine and Append to Chat ---
    step_header = f"--- **Step {step_num}** ---"
    final_content = step_header + "<br/>" + screenshot_html + formatted_output

    chat_message = {
        "role": "assistant",
        "content": final_content.strip(),
    }
    webui_manager.bu_chat_history.append(chat_message)


def _handle_done(webui_manager: "WebuiManager", history: AgentHistoryList):
    """Callback when the agent finishes the task."""
    logger.info(
        f"Agent task finished. Duration: {history.total_duration_seconds():.2f}s, Tokens: {history.total_input_tokens()}"
    )
    final_summary = "**Task Completed**\n"
    final_summary += f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
    final_summary += f"- Total Input Tokens: {history.total_input_tokens()}\n"

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
        webui_manager: "WebuiManager", query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")

    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )

    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None

    try:
        await asyncio.wait_for(webui_manager.bu_response_event.wait(), timeout=3600.0)
    except asyncio.TimeoutError:
        webui_manager.bu_chat_history.append(
            {"role": "assistant", "content": "**Timeout:** No response received. Trying to proceed."}
        )
        webui_manager.bu_response_event = None
        return {"response": "Timeout: User did not respond."}

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append({"role": "user", "content": response})
    webui_manager.bu_response_event = None
    return {"response": response}


async def _confirm_action_callback(
        webui_manager: "WebuiManager", action_name: str, params: Dict[str, Any], browser_context: BrowserContext
) -> str:
    """Callback triggered by the controller before executing a critical action."""
    logger.info(f"Action '{action_name}' requires confirmation.")

    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"✋ **Confirmation Required**\n\nI am about to execute:\n**Action:** `{action_name}`\n**Params:** `{params_str}`\n\n"
                       f"- **Approve:** Press 'Submit Response' (or type 'yes')\n"
                       f"- **Reject:** Type 'no' or 'skip'\n"
                       f"- **Modify:** Type `modify: {{ \"key\": \"value\" }}` to change parameters"
        }
    )

    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None

    try:
        await asyncio.wait_for(webui_manager.bu_response_event.wait(), timeout=3600.0)
    except asyncio.TimeoutError:
        webui_manager.bu_response_event = None
        return "yes"

    response = webui_manager.bu_user_help_response or "yes"
    webui_manager.bu_chat_history.append({"role": "user", "content": response})
    webui_manager.bu_response_event = None
    return response


# --- Main Execution Logic ---

async def run_agent_task(
        webui_manager: "WebuiManager", components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""

    # --- Get Components ---
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id("browser_use_agent.stop_button")
    pause_resume_button_comp = webui_manager.get_component_by_id("browser_use_agent.pause_resume_button")
    clear_button_comp = webui_manager.get_component_by_id("browser_use_agent.clear_button")
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
    history_file_comp = webui_manager.get_component_by_id("browser_use_agent.agent_history_file")
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    browser_view_comp = webui_manager.get_component_by_id("browser_use_agent.browser_view")
    brain_file_comp = webui_manager.get_component_by_id("browser_use_agent.brain_file")
    memory_file_comp = webui_manager.get_component_by_id("browser_use_agent.memory_file")

    # --- 1. Get Task and Initial UI Update ---
    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    webui_manager.bu_last_task_prompt = task
    webui_manager.bu_chat_history.append({"role": "user", "content": task})
    webui_manager.bu_agent_status = "### Agent Status\nRunning..."

    yield {
        user_input_comp: gr.update(value="", interactive=True, placeholder="Agent is running... Enter text to steer/add instructions."),
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
    
    if brain_file and os.path.exists(brain_file):
        try:
            with open(brain_file, "r", encoding="utf-8") as f:
                agent_settings["override_system_prompt"] = f.read()
            logger.info(f"Loaded Brain from {brain_file}")
        except Exception as e:
            logger.error(f"Failed to load Brain file: {e}")

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

    async def ask_callback_wrapper(query: str, browser_context: BrowserContext) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    async def confirm_callback_wrapper(action_name: str, params: Dict[str, Any], browser_context: BrowserContext) -> str:
        return await _confirm_action_callback(webui_manager, action_name, params, browser_context)

    async def update_plan_callback_wrapper(step_index: int, status: str):
        if hasattr(webui_manager, "bu_plan") and webui_manager.bu_plan:
            idx = step_index - 1
            if 0 <= idx < len(webui_manager.bu_plan):
                webui_manager.bu_plan[idx]["status"] = status
                webui_manager.bu_plan_updated = True
                if status == "failed" and agent_settings.get("enable_auto_pause", False):
                    if webui_manager.bu_agent:
                        webui_manager.bu_agent.pause()

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

    should_close_browser_on_finish = not browser_settings["keep_browser_open"]

    try:
        await initialize_browser_infrastructure(webui_manager, browser_settings)

        async def step_callback_wrapper(state: BrowserState, output: AgentOutput, step_num: int):
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

        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task
        webui_manager.bu_latest_screenshot = None

        stream_vw = 70
        stream_vh = int(70 * browser_settings["window_h"] // browser_settings["window_w"])
        async def screenshot_loop():
            while not agent_task.done():
                if browser_settings["headless"] and webui_manager.bu_browser_context:
                    try:
                        webui_manager.bu_latest_screenshot = await webui_manager.bu_browser_context.take_screenshot()
                    except Exception: pass
                sleep_time = 0.5
                if webui_manager.bu_agent and webui_manager.bu_agent.state.paused:
                    sleep_time = 1.0
                await asyncio.sleep(sleep_time)
        screenshot_task = asyncio.create_task(screenshot_loop()) if browser_settings["headless"] else None

        last_chat_len = len(webui_manager.bu_chat_history)
        last_status = webui_manager.bu_agent_status
        last_screenshot_b64 = None
        
        while not agent_task.done():
            is_paused = webui_manager.bu_agent.state.paused
            is_stopped = webui_manager.bu_agent.state.stopped

            if is_paused:
                yield {
                    pause_resume_button_comp: gr.update(value="▶️ Resume", interactive=True),
                    stop_button_comp: gr.update(interactive=True),
                }
                while is_paused and not agent_task.done():
                    is_paused = webui_manager.bu_agent.state.paused
                    is_stopped = webui_manager.bu_agent.state.stopped
                    if is_stopped: break
                    await asyncio.sleep(0.2)

                if agent_task.done() or is_stopped: break

                yield {
                    pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=True),
                    run_button_comp: gr.update(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
                }

            if is_stopped:
                logger.info("Agent has stopped.")
                if not agent_task.done():
                    try:
                        await asyncio.wait_for(agent_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        agent_task.cancel()
                    except Exception: pass
                break

            update_dict = {}
            if webui_manager.bu_response_event is not None:
                update_dict = {
                    user_input_comp: gr.update(placeholder="Agent needs help. Enter response and submit.", interactive=True),
                    run_button_comp: gr.update(value="✔️ Submit Response", interactive=True),
                    retry_button_comp: gr.update(interactive=False),
                    pause_resume_button_comp: gr.update(interactive=False),
                    stop_button_comp: gr.update(interactive=False),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
                last_chat_len = len(webui_manager.bu_chat_history)
                yield update_dict
                await webui_manager.bu_response_event.wait()

                if not agent_task.done():
                    yield {
                        user_input_comp: gr.update(placeholder="Agent is running... Enter text to steer/add instructions.", interactive=True),
                        run_button_comp: gr.update(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
                        retry_button_comp: gr.update(interactive=False),
                        pause_resume_button_comp: gr.update(interactive=True),
                        stop_button_comp: gr.update(interactive=True),
                    }
                else:
                    break

            if len(webui_manager.bu_chat_history) > last_chat_len:
                update_dict[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
                last_chat_len = len(webui_manager.bu_chat_history)

            if webui_manager.bu_agent_status != last_status:
                update_dict[agent_status_comp] = gr.update(value=webui_manager.bu_agent_status)
                last_status = webui_manager.bu_agent_status
            
            if getattr(webui_manager.bu_agent, "is_validating", False):
                update_dict[agent_status_comp] = gr.update(value=webui_manager.bu_agent_status + "\n\n🔍 **Validating Output with Confirmer LLM...**")
            
            if getattr(webui_manager.bu_agent, "switched_to_retry_model", False) and "Smart Retry Active" not in webui_manager.bu_agent_status:
                 webui_manager.bu_agent_status = "<span class='retry-badge'>🧠 Smart Retry Active</span>\n" + webui_manager.bu_agent_status
            
            if getattr(webui_manager.bu_agent, "using_cheap_model", False) and "Cost Saver Active" not in webui_manager.bu_agent_status:
                 webui_manager.bu_agent_status = "<span class='retry-badge' style='background-color: #10b981;'>💰 Cost Saver Active</span>\n" + webui_manager.bu_agent_status

            if getattr(webui_manager, "bu_plan_updated", False):
                plan_md = render_plan_markdown(webui_manager.bu_plan)
                update_dict[plan_status_comp] = gr.update(value=plan_md)
                update_dict[plan_editor_comp] = gr.update(value=json.dumps(webui_manager.bu_plan, indent=2))
                webui_manager.bu_plan_updated = False

            if browser_settings["headless"] and webui_manager.bu_browser_context:
                screenshot_b64 = webui_manager.bu_latest_screenshot
                if screenshot_b64:
                    if screenshot_b64 != last_screenshot_b64:
                        last_screenshot_b64 = screenshot_b64
                        html_content = f'<img src="data:image/jpeg;base64,{screenshot_b64}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        update_dict[browser_view_comp] = gr.update(value=html_content, visible=True)
                elif last_screenshot_b64 != "waiting":
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                    update_dict[browser_view_comp] = gr.update(value=html_content, visible=True)
                    last_screenshot_b64 = "waiting"
            else:
                if last_screenshot_b64 is not None:
                    update_dict[browser_view_comp] = gr.update(visible=False)
                    last_screenshot_b64 = None

            if update_dict:
                yield update_dict

            await asyncio.sleep(0.2)

        webui_manager.bu_agent.state.paused = False
        webui_manager.bu_agent.state.stopped = False
        final_update = {}
        try:
            logger.info("Agent task completing...")
            if not agent_task.done():
                await agent_task
            elif agent_task.exception():
                agent_task.result()
            logger.info("Agent task completed processing.")

            webui_manager.bu_agent.save_history(history_file)
            if os.path.exists(history_file):
                final_update[history_file_comp] = gr.File(value=history_file)

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
                final_update[gif_comp] = gr.Image(value=gif_path)

        except asyncio.CancelledError:
            logger.info("Agent task was cancelled.")
            if not any("Cancelled" in msg.get("content", "") for msg in webui_manager.bu_chat_history if msg.get("role") == "assistant"):
                webui_manager.bu_chat_history.append({"role": "assistant", "content": "**Task Cancelled**."})
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            if not any(error_message in msg.get("content", "") for msg in webui_manager.bu_chat_history if msg.get("role") == "assistant"):
                webui_manager.bu_chat_history.append({"role": "assistant", "content": error_message})
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
            gr.Error(f"Agent execution failed: {e}")

        finally:
            webui_manager.bu_current_task = None
            if screenshot_task and not screenshot_task.done():
                screenshot_task.cancel()

            if should_close_browser_on_finish:
                if webui_manager.bu_browser_context:
                    await webui_manager.bu_browser_context.close()
                    webui_manager.bu_browser_context = None
                if webui_manager.bu_browser:
                    await webui_manager.bu_browser.close()
                    webui_manager.bu_browser = None

            final_update.update({
                user_input_comp: gr.update(value="", interactive=True, placeholder="Enter your next task..."),
                run_button_comp: gr.update(value="▶️ Submit Task", interactive=True, variant="primary"),
                retry_button_comp: gr.update(interactive=True),
                stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
                pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
                clear_button_comp: gr.update(interactive=True),
                chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                agent_status_comp: gr.update(value=webui_manager.bu_agent_status),
            })
            yield final_update

    except Exception as e:
        logger.error(f"Error setting up agent task: {e}", exc_info=True)
        webui_manager.bu_current_task = None
        yield {
            user_input_comp: gr.update(interactive=True, placeholder="Error during setup. Enter task..."),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True, variant="primary"),
            retry_button_comp: gr.update(interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
            clear_button_comp: gr.update(interactive=True),
            chatbot_comp: gr.update(value=webui_manager.bu_chat_history + [{"role": "assistant", "content": f"**Setup Error:** {e}"}]),
        }

# --- Handlers ---

async def handle_submit(webui_manager: "WebuiManager", components: Dict[gr.components.Component, Any]):
    """Handles clicks on the main 'Submit' button."""
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    user_input_value = components.get(user_input_comp, "").strip()

    if webui_manager.bu_response_event and not webui_manager.bu_response_event.is_set():
        webui_manager.bu_user_help_response = user_input_value if user_input_value else "User provided no text response."
        webui_manager.bu_response_event.set()
        yield {
            user_input_comp: gr.update(value="", interactive=False, placeholder="Waiting for agent to continue..."),
            webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(value="⬆️ Add Instruction", interactive=True, variant="secondary"),
        }
    elif webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        if user_input_value:
            webui_manager.bu_chat_history.append({"role": "user", "content": user_input_value})
            async def inject_task_async():
                try:
                    agent = webui_manager.bu_agent
                    if hasattr(agent, "add_new_task"):
                        agent.add_new_task(user_input_value)
                    elif hasattr(agent, "agent") and hasattr(agent.agent, "add_new_task"):
                        agent.agent.add_new_task(user_input_value)
                    elif hasattr(agent, "message_manager"):
                        agent.message_manager.add_user_message(user_input_value)
                    elif hasattr(agent, "agent") and hasattr(agent.agent, "message_manager"):
                        agent.agent.message_manager.add_user_message(user_input_value)
                except Exception as e:
                    logger.error(f"Error steering agent: {e}")
            asyncio.create_task(inject_task_async())
            yield {user_input_comp: gr.update(value="")}
        else:
            yield {}
    else:
        async for update in run_agent_task(webui_manager, components):
            yield update

async def handle_retry(webui_manager: "WebuiManager", components: Dict[gr.components.Component, Any]):
    """Handles clicks on the 'Retry' button."""
    if not webui_manager.bu_last_task_prompt:
        gr.Warning("No previous task to retry.")
        yield {}
        return
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    components[user_input_comp] = webui_manager.bu_last_task_prompt
    async for update in run_agent_task(webui_manager, components):
        yield update

async def handle_stop(webui_manager: "WebuiManager"):
    """Handles clicks on the 'Stop' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task
    if agent and task and not task.done():
        agent.state.stopped = True
        agent.state.paused = False
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")

async def handle_pause_resume(webui_manager: "WebuiManager"):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task
    if agent and task and not task.done():
        if agent.state.paused:
            agent.resume()
        else:
            agent.pause()
    else:
        logger.warning("Pause/Resume clicked but agent is not running.")

async def handle_reload_memory(webui_manager: "WebuiManager", memory_file_path: str):
    """Handles clicks on the 'Reload Memory' button."""
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
    if files and isinstance(files, list):
        return gr.update(choices=files)
    else:
        return gr.update(choices=[])

async def handle_load_kb_file(webui_manager: "WebuiManager", memory_file_path: str, selected_file: str):
    """Loads content of the selected KB file."""
    memory_content_comp = webui_manager.get_component_by_id("browser_use_agent.memory_content")
    if not memory_file_path or not selected_file:
        return {memory_content_comp: gr.update(value="")}
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    content = load_kb_content(base_dir, selected_file)
    return {memory_content_comp: gr.update(value=content)}

async def handle_save_chat(webui_manager: "WebuiManager"):
    """Handles clicks on the 'Save Chat' button."""
    history = webui_manager.bu_chat_history
    if not history:
        return {}
    save_path = "./tmp/chat_logs"
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_log_{timestamp}.json"
    full_path = os.path.join(save_path, filename)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return {webui_manager.get_component_by_id("browser_use_agent.chat_log_file"): gr.File(value=full_path, visible=True)}
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
        return {}

async def handle_clear(webui_manager: "WebuiManager"):
    """Handles clicks on the 'Clear' button."""
    task = webui_manager.bu_current_task
    if task and not task.done():
        webui_manager.bu_agent.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError): pass
        except Exception: pass
    webui_manager.bu_current_task = None

    if webui_manager.bu_controller:
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None
    webui_manager.bu_agent = None

    webui_manager.bu_chat_history = []
    webui_manager.bu_response_event = None
    webui_manager.bu_user_help_response = None
    webui_manager.bu_agent_task_id = None
    webui_manager.bu_agent_status = "### Agent Status\nReady"
    webui_manager.bu_last_task_prompt = None
    webui_manager.bu_resumed_history = None

    return {
        webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(value=[]),
        webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(value="", placeholder="Enter your task here..."),
        webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(value=webui_manager.bu_agent_status),
        webui_manager.get_component_by_id("browser_use_agent.agent_history_file"): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.recording_gif"): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.browser_view"): gr.update(value="<div style='...'>Browser Cleared</div>"),
        webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(value="▶️ Submit Task", interactive=True, variant="primary"),
        webui_manager.get_component_by_id("browser_use_agent.retry_button"): gr.update(interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.stop_button"): gr.update(interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.pause_resume_button"): gr.update(value="⏸️ Pause", interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.clear_button"): gr.update(interactive=True),
        webui_manager.get_component_by_id("browser_use_agent.chat_log_file"): gr.update(value=None),
    }

async def handle_refresh_history_files(save_agent_history_path: str):
    """Refreshes the list of available history files."""
    if not save_agent_history_path:
        return gr.update(choices=[])
    if not os.path.exists(save_agent_history_path):
        return gr.update(choices=[])
    files = []
    for task_id in os.listdir(save_agent_history_path):
        task_dir = os.path.join(save_agent_history_path, task_id)
        if os.path.isdir(task_dir):
            json_file = os.path.join(task_dir, f"{task_id}.json")
            if os.path.exists(json_file):
                mtime = os.path.getmtime(json_file)
                files.append((json_file, mtime))
    files.sort(key=lambda x: x[1], reverse=True)
    choices = [f[0] for f in files]
    return gr.update(choices=choices)

async def handle_update_plan(webui_manager: "WebuiManager", new_plan_json: str):
    """Updates the plan from the editor."""
    plan_status_comp = webui_manager.get_component_by_id("browser_use_agent.plan_status")
    if not new_plan_json: return {}
    try:
        new_plan = json.loads(new_plan_json)
        if isinstance(new_plan, list):
            webui_manager.bu_plan = new_plan
            webui_manager.bu_plan_updated = True
            if webui_manager.bu_agent:
                msg = f"NOTE: The plan has been manually updated by the user. Current Plan Status: {json.dumps(new_plan)}"
                if hasattr(webui_manager.bu_agent, "message_manager"):
                     webui_manager.bu_agent.message_manager.add_user_message(msg)
            plan_md = render_plan_markdown(new_plan)
            return {plan_status_comp: gr.update(value=plan_md)}
        else:
            gr.Warning("Plan must be a JSON list.")
            return {}
    except Exception as e:
        gr.Warning(f"Invalid JSON: {e}")
        return {}

async def handle_save_generated_kb(webui_manager: "WebuiManager", title: str, content: str, memory_file_path: str):
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

async def handle_resume_session(webui_manager: "WebuiManager", history_file: str, components: Dict[gr.components.Component, Any]):
    """Resumes a session from a history file."""
    if not history_file or not os.path.exists(history_file):
        gr.Warning("History file not found.")
        yield {}
        return
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        history = AgentHistoryList.model_validate(history_data)
        yield {webui_manager.get_component_by_id("browser_use_agent.agent_status"): gr.update(value="### Agent Status\nResuming Session...")}
        webui_manager.bu_chat_history = []
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
