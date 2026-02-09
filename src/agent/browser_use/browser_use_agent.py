from __future__ import annotations

import abc
import asyncio
import logging
import os
import json
import re
import sys
from pathlib import Path
import json_repair
from typing import List, Dict, Optional, Any
from datetime import datetime
from typing import Callable, Union
# Ensure project root is in sys.path so 'src' imports work
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

# from lmnr.sdk.decorators import observe
from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc # AIMessage is from langchain_core.messages
from browser_use.agent.views import (
    ActionResult,
    AgentSettings,
    AgentHistory,
    AgentOutput,
    AgentHistoryList,
    AgentStepInfo,
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async
from dotenv import load_dotenv

# from browser_use.agent.message_manager.utils import is_model_without_tool_support
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from src.utils.utils import retry_async, parse_json_safe # Ensure parse_json_safe is imported
from langchain_core.runnables import Runnable, RunnableLambda # Import RunnableLambda
from langchain_core.runnables import chain
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD
from src.utils.memory_utils import get_memory_manager
from src.agent.browser_use import agent_utils  # Import the new module
from src.agent.browser_use.agent_heuristics import AgentHeuristics

load_dotenv()
logger = logging.getLogger(__name__)

from src.utils.io_manager import IOManager

SKIP_LLM_API_KEY_VERIFICATION = True

class InstructionHandlerInterface(abc.ABC):
    @abc.abstractmethod
    async def handle_instructions(self, task: Task, decision: Dict[str, str]) -> str:
        pass



class AgentLLMManager:
    """Manages LLM-related operations for the agent, including model switching and patching."""

    def __init__(self, agent, model_priority_list: Optional[List[Dict]] = None):
        self.agent = agent
        self.model_priority_list = model_priority_list or []
        self.main_llm = agent.llm  # Store the primary (high-quality) LLM
        self.current_model_index = -1  # -1 indicates main_llm

    def set_llm(self, llm):
        """Sets the current LLM for the agent."""
        self.agent.llm = llm

    def get_current_llm(self):
        """Returns the current LLM."""
        return self.agent.llm

    def _patch_llm_with_qwen_fix(self, llm: Runnable) -> Runnable:
        """Wraps the LLM with a RunnableLambda that applies the Qwen/Ollama JSON fix logic."""
        llm_with_aimessage_output = llm | RunnableLambda(self._convert_agent_output_to_aimessage)
        return llm_with_aimessage_output | RunnableLambda(self._apply_qwen_fix_logic)

    def _convert_agent_output_to_aimessage(self, agent_output: AgentOutput) -> BaseMessage:
        tool_calls = []
        json_content = agent_output.model_dump_json()
        if agent_output.action:
            actions_list = agent_output.action if isinstance(agent_output.action, list) else [agent_output.action]
            for action_model in actions_list:
                if hasattr(action_model, "model_dump"):  # Ensure it's an ActionModel-like object
                    action_args = action_model.model_dump(exclude_unset=True)
                    tool_name = action_args.pop("name", None)  # Extract 'name' and remove from args

                    if tool_name:
                        tool_calls.append({
                            "name": tool_name,
                            "args": action_args
                        })
                    else:
                        logger.warning(
                            f"ActionModel found without a 'name' attribute: {action_model}. Skipping tool call conversion.")
                else:
                    logger.warning(
                        f"Unexpected action type in AgentOutput: {type(action_model)}. Skipping tool call conversion.")

        return AIMessage(content=json_content, tool_calls=tool_calls)

    def _apply_qwen_fix_logic(self, response: BaseMessage) -> BaseMessage:
        """Applies the Qwen/Ollama JSON fix logic to an LLM response (BaseMessage)."""
        if not hasattr(response, 'content') or not isinstance(response.content, str):
            return response

        content = response.content.strip()
        if not content:
            return response

        try:
            logger.debug(f"QwenFixWrapper: Attempting to parse raw LLM output: '{content}'")
            data = parse_json_safe(content)  # Use parse_json_safe from utils
            if isinstance(data, (dict, list)):
                # Specific fix for AgentOutput wrapped in a dict
                if isinstance(data, dict) and data.get("name") == "AgentOutput" and "arguments" in data:
                    logger.info("🔧 Fixed Qwen/Ollama output format by extracting 'arguments'.")
                    response.content = json.dumps(data["arguments"])
                else:
                    response.content = json.dumps(data)
        except Exception as e:
            logger.debug(f"QwenFixWrapper: Initial json_repair failed: {e}. Attempting more aggressive repair.")
            # Fallback to more aggressive repair or default to empty JSON
            try:
                if '{' in content and not content.endswith('}'):
                    fixed_content = content + '}' * (content.count('{') - content.count('}'))
                    data = json_repair.loads(fixed_content)
                    response.content = json.dumps(data)
            except Exception as inner_e:
                logger.warning(f"QwenFixWrapper: Inner repair attempt failed: {inner_e}. Original: '{content}'")
                response.content = "{}"
        return response

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.agent.settings.tool_calling_method

        if tool_calling_method == 'auto':
            if 'Ollama' in self.chat_model_library:
                return 'raw'
            elif any(m in self.agent.model_name.lower() for m in ['qwen', 'deepseek', 'gpt', 'claude']) and 'Ollama' not in self.chat_model_library:
                return 'function_calling'
            elif self.agent.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.agent.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.agent.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method


class ConfirmationManager:
    """
    Manages confirmation requests and responses, allowing indefinite waiting
    for external approval.
    """

    def __init__(self, send_agent_message_callback, confirmation_response_queue):
        self.send_agent_message_callback = send_agent_message_callback
        self.confirmation_response_queue = confirmation_response_queue

    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Requests confirmation/input from the user via WebSocket, allowing indefinite waiting.
        """
        if not self.send_agent_message_callback or not self.confirmation_response_queue:
            return True

        await self.send_agent_message_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review. Awaiting external device approval.",
            "intel": intel,
            "next_task": next_task
        })

        response_data = await self.confirmation_response_queue.get()
        response = response_data.get("response")
        custom_task = response_data.get("custom_task")

        if response == "yes":
            return True
        elif response == "no":
            return False
        elif custom_task:
            return custom_task
        else:
            return False





class BrowserActionHandler:
    """
    Handles browser-specific actions and logic, decoupling it from the agent's core.
    """
    def __init__(self, browser_context, controller, heuristics):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics


class AgentLoopHandler:
    """
    Handles the main execution loop of the agent, decoupling it from the agent's core.
    """

    def __init__(self, agent, settings):
        self.agent = agent
        self.settings = settings

    async def run_step(self, step, on_step_start, on_step_end):
        """Executes a single step in the agent's execution loop."""
        try:
            if await self._handle_pre_step():
                 return True

            if on_step_start is not None:
                await on_step_start(self.agent)

            step_info = AgentStepInfo(step_number=step, max_steps=self.settings.max_steps)
            await self.agent.step(step_info)

            if on_step_end is not None:
                await on_step_end(self.agent)

            if await self._handle_post_step(step, self.settings.max_steps):
                return True

        except Exception as e:
            raise e

ToolCallingMethod = str

class BrowserUseAgent(Agent):
    async def execute_initial_actions(self, initial_actions) -> List[str]:
        """
        Executes initial actions directly using the controller.
        This bypasses the standard agent loop for forced plan steps.
        Returns a list of result strings.
        """
        if not initial_actions:
            return []

        results = []
        for action_data in initial_actions:
            # Handle Pydantic models (ActionModel) if browser_use converted them
            if hasattr(action_data, "model_dump"):
                action_data = action_data.model_dump(exclude_none=True)
            elif hasattr(action_data, "dict"):
                action_data = action_data.dict(exclude_none=True)

            for action_name, params in action_data.items():
                try:
                    logger.info(f"🚀 Executing initial action: {action_name} with params: {params}")
                    if self.controller and hasattr(self.controller, "execute_action_by_name"):
                        result = await self.controller.execute_action_by_name(action_name, params, self.browser_context)
                        results.append(f"Action '{action_name}' executed. Result: {result}")
                        logger.info(f"✅ Result: {result}")
                    else:
                        results.append(f"Action '{action_name}' skipped (Controller missing).")
                except Exception as e:
                    logger.error(f"❌ Error executing initial action {action_name}: {e}")
                    results.append(f"Action '{action_name}' failed: {e}")
        return results

    async def manage_tabs(self):
        """Closes extraneous tabs to keep the agent focused."""
        if self.controller and hasattr(self.controller, "nav_controller"):
            await self.controller.nav_controller.manage_tabs(self.browser_context)

    async def inject_site_knowledge(self, state, memory_manager, last_domain, heuristics):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        try:
            if not state.history.history:
                return
                
            current_url = state.history.history[-1].state.url
            if not current_url:
                return

            from urllib.parse import urlparse
            domain = urlparse(current_url).netloc.replace("www.", "")
            
            if domain != last_domain:
                last_domain = domain
                knowledge = memory_manager.get_site_knowledge(current_url)
                if knowledge:
                    msg = f"🧠 Memory Retrieval: I have previous knowledge about {domain}:\n{knowledge}"
                    heuristics.inject_message(msg)
                    logger.info(f"Injected site knowledge for {domain}")
        except Exception as e:
            logger.warning(f"Failed to inject site knowledge: {e}")
        return last_domain

    async def check_navigation_recovery(self):
        """Checks if navigation recovery is needed using specialized assessment logic."""
        await self.heuristics.check_navigation_recovery()

    async def check_and_add_subtasks(self):
        """
        Checks if the current page contains a list of subtasks relevant to the goal.
        If found, automatically adds them to the plan via the controller.
        """
        await self.heuristics.check_and_add_subtasks()

    async def check_blocking_elements(self):
        """Checks for blocking elements (ads/popups) and injects a warning."""
        await self.heuristics.check_blocking_elements()

    async def check_completion_heuristics(self):
        """Checks for common completion signals in the page content."""
        await self.heuristics.check_completion_heuristics()

    async def check_login_status(self):
        """Checks if the user is logged in and injects a notification."""
        await self.heuristics.check_login_status()


class AgentLoopHandler:
    """
    Handles the main execution loop of the agent, decoupling it from the agent's core.
    """

    def __init__(self, agent, settings):
        self.agent = agent
        self.settings = settings

    async def run_step(self, step, on_step_start, on_step_end):
        """Executes a single step in the agent's execution loop."""
        try:
            # return True

            if on_step_start is not None:
                await on_step_start(self.agent)

            step_info = AgentStepInfo(step_number=step, max_steps=self.settings.max_steps)
            await self.agent.step(step_info)

            if on_step_end is not None:
                await on_step_end(self.agent)

            if await self._handle_post_step(step, self.settings.max_steps):
                return True

        except Exception as e:
            raise e

ToolCallingMethod = str

class BrowserUseAgent(Agent):
    def __init__(
        self,
        confirmer_llm=None,
        confirmer_strictness=None,
        use_vision_for_confirmer=True,
        model_priority_list: Optional[List[Dict]] = None,
        enable_cost_saver=False,
        enable_smart_retry=False,
        auto_save_on_stuck=False,
        source=None,
        history: AgentHistoryList = None,
        save_history_path: str = None,
        system_prompt: str = None,
        inhibit_close: bool = False,
        current_step_index: int = None,
        validation_callback=None,
        browser_context = None,
        send_agent_message_callback: Optional[Callable] = None,
        confirmation_event: Optional[asyncio.Event] = None,
        confirmation_response_queue: Optional[asyncio.Queue] = None,
        max_consecutive_failures: int = 5, instruction_handler: Optional[InstructionHandlerInterface] = None, **kwargs

    ):
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness = confirmer_strictness
        self.browser_context = browser_context
        self.use_vision_for_confirmer = use_vision_for_confirmer
        self.model_priority_list = model_priority_list or []
        self.enable_cost_saver = enable_cost_saver
        self.enable_smart_retry = enable_smart_retry
        self.auto_save_on_stuck = auto_save_on_stuck
        self.source = source
        self.save_history_path = save_history_path
        self.inhibit_close = inhibit_close
        self.is_validating = False
        self.switched_to_retry_model = False
        self.using_cheap_model = False
        self.settings = kwargs.get("settings")
        self.current_model_index = -1 # -1 indicates main_llm
        self.current_step_index = current_step_index
        self.validation_callback = validation_callback
        self.max_consecutive_failures = max_consecutive_failures
        self.send_agent_message_callback = send_agent_message_callback
        self.confirmation_event = confirmation_event
        self.confirmation_response_queue = confirmation_response_queue
        self.instruction_handler = instruction_handler

        # Prepare arguments for the parent Agent class
        agent_kwargs = kwargs.copy()

        
        # Extract use_memory to avoid passing it to parent Agent which doesn't support it
        # We intercept 'use_memory' to use our own SimpleMemoryManager instead of mem0
        self.use_custom_memory = agent_kwargs.pop('use_memory', False)
        
        # Extract agent-specific args that should not be passed to the base Agent class
        self.planner_llm = agent_kwargs.pop('planner_llm', None)
        self.planner_interval = agent_kwargs.pop('planner_interval', 5.0)
        self.cookie_path = agent_kwargs.pop('cookie_path', None)
        self.enable_user_interaction_dialog = agent_kwargs.pop('enable_user_interaction_dialog', False)
        self.agent_control_queue = agent_kwargs.pop('agent_control_queue', None)
        
        self.planner_task = None
        
        # Note: We do NOT pass 'use_memory' to super().__init__ to avoid TypeError if the parent class doesn't support it.

        if confirmer_llm:
            agent_kwargs['validate_output'] = True
            

        if history:
            agent_kwargs['history'] = history

        # Patch LLM to fix Qwen/Ollama JSON format issues BEFORE passing to parent
        # We patch the instance directly to satisfy Pydantic validation checks for BaseChatModel
        if 'llm' in agent_kwargs:
            original_llm = agent_kwargs['llm']
            

            # Apply structured output using AgentOutput schema
            # This instructs the LLM to produce output conforming to AgentOutput
            if isinstance(original_llm, Runnable):
                structured_llm = original_llm.with_structured_output(AgentOutput)
                logger.info("Applied AgentOutput schema to LLM using with_structured_output.")
            else:
                logger.warning("LLM is not a Runnable, cannot apply with_structured_output. Proceeding without structured output enforcement.")
                structured_llm = original_llm
            logger.debug(f"Type of LLM after structured output: {type(structured_llm)}")

        super().__init__(**agent_kwargs)
        
        # Apply system prompt override if provided

        self.llm_manager = AgentLLMManager(self,model_priority_list)




        if system_prompt:
            if hasattr(self, "message_manager"):
                self.message_manager.system_message = SystemMessage(content=system_prompt)
        # Link controller to this agent for HUD control
        if self.controller and hasattr(self.controller, "set_agent"):
            self.controller.set_agent(self)


        if model_priority_list and 'llm' in model_priority_list[0]:
            self.llm = model_priority_list[0]['llm']
        else:
            self.llm_manager.set_llm(agent_kwargs['llm'])
        self.last_domain = None
        self.successful_steps_since_switch = 0

        self.llm = agent_kwargs['llm']
        self.confirmation_manager = ConfirmationManager(send_agent_message_callback, confirmation_response_queue)
        self.memory_manager = get_memory_manager() if self.use_custom_memory else None
        self.heuristics = AgentHeuristics(self)

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method

        if tool_calling_method == 'auto':
            if 'Ollama' in self.chat_model_library:
                return 'raw'
            elif any(m in self.model_name.lower() for m in ['qwen', 'deepseek', 'gpt', 'claude']) and 'Ollama' not in self.chat_model_library:
                return 'function_calling'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    async def _setup_components(self, agent_kwargs: Dict[str, Any]) -> None:
        """
        Extracts and sets up agent components.
        """

        self.use_custom_memory = agent_kwargs.pop('use_memory', False)
        self.planner_llm = agent_kwargs.pop('planner_llm', None)
        self.planner_interval = agent_kwargs.pop('planner_interval', 5.0)
        self.cookie_path = agent_kwargs.pop('cookie_path', None)
        self.enable_user_interaction_dialog = agent_kwargs.pop('enable_user_interaction_dialog', False)
        self.agent_control_queue = agent_kwargs.pop('agent_control_queue', None)

        self.planner_task = None

        self.loop_handler = AgentLoopHandler(self, self.settings)

    @time_execution_async("--run (agent)")
    async def run(
            self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None,
            on_step_end: AgentHookFunc | None = None
    ) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()

        if self.send_agent_message_callback:
            await self.send_agent_message_callback({"type": "agent_goal", "goal": self.task})
            await self.send_agent_message_callback({"type": "agent_status", "status": "Participating"})

        try:
            self._log_agent_run()

            # Start Planner Loop if enabled
            if self.planner_llm:
                self.planner_task = asyncio.create_task(self._planner_loop())

            # Execute initial actions if provided
            if self.initial_actions:
                # Use custom execution to ensure actions are run reliably via controller
                results = await self._execute_initial_actions()
                if results:
                    # Inject into message history so the agent knows what happened
                    summary = "\n".join(results)
                    msg = f"SYSTEM NOTE: The following actions were already executed for this step:\n{summary}\n\nCheck the current state. If the goal is achieved, output {{'finish': ...}}. Do NOT repeat the above actions."
                    if hasattr(self, "message_manager"):
                        if hasattr(self.message_manager, "add_user_message"):
                            self.message_manager.add_user_message(msg)
                        elif hasattr(self.message_manager, "add_message"):
                            self.message_manager.add_message(HumanMessage(content=msg))
                    # Also set last_result for good measure if the parent class uses it
                    self.state.last_result = [ActionResult(extracted_content=summary, include_in_memory=True)]

            for step in range(max_steps):
                try:
                    if await self._handle_pre_step():
                        break

                    if on_step_start is not None:
                        await on_step_start(self)

                    step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                    await self.step(step_info)

                    if on_step_end is not None:
                        await on_step_end(self)

                    if await self._handle_post_step(step, max_steps):
                        break

                    if step == 41:
                        logger.warning("Agent is hanging at step 42. Investigate potential issues.")
                        self.heuristics.inject_message("SYSTEM: Agent is potentially hanging at step 42. Please investigate.")

                    # If a custom task was provided, the agent.task would have been updated.
                    # We should continue the loop to process the new task.

                except Exception as e:
                    if "Timeout" in str(e) and "exceeded" in str(e):
                        logger.warning(f"⚠️ Step {step + 1} timed out: {e}. Attempting to recover by reloading page.")
                        self.state.consecutive_failures += 1
                        try:
                            logger.info("Attempting to reload the page to recover...")
                            page = await self.heuristics._get_current_page()
                            if page and not page.is_closed():
                                await page.reload(wait_until="domcontentloaded", timeout=15000)
                                self.heuristics.inject_message("SYSTEM: A timeout occurred. I have reloaded the page to recover. Please re-assess the current state.")
                        except Exception as reload_e:
                            logger.error(f"Failed to reload page after timeout: {reload_e}")
                        continue
                    logger.error(f"An unexpected error occurred during step {step + 1}: {e}")
                    self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step + 1}: {e}")
                    raise e
            else:
                self._handle_max_steps_reached()

            return self.state.history

        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history
        except asyncio.CancelledError:
            logger.info('Agent execution cancelled')
            return self.state.history

        finally:
            await self._handle_cleanup()

    async def _planner_loop(self):
        """Background task to dynamically update the plan."""
        if not self.planner_llm:
            return

        logger.info("🧠 Planner Loop started.")
        
        while not self.state.stopped:
            try:
                await asyncio.sleep(self.planner_interval)

                if self.state.paused or self.is_validating or self.state.history.is_done() or len(self.state.history.history) < 2:
                    continue

                # Check dependencies
                if not self.controller or not hasattr(self.controller, "webui_manager"):
                    continue
                
                manager = self.controller.webui_manager
                if not manager or not hasattr(manager, "bu_plan") or not manager.bu_plan:
                    continue

                # --- 1. Gather Context for Planner ---
                last_step = self.state.history.history[-1]
                plan_str = json.dumps(manager.bu_plan, indent=2)
                page_summary = (last_step.state.element_tree or "")[:4000]
                last_thought = getattr(last_step.model_output, "thought", "N/A") if last_step.model_output else "N/A"

                # --- 2. Construct Prompt ---
                # NOTE: This requires PLANNER_PROMPT to be defined in src/utils/prompts.py
                from src.utils.prompts import PLANNER_PROMPT
                prompt = PLANNER_PROMPT.format(
                    goal=self.task,
                    plan=plan_str,
                    last_thought=last_thought,
                    page_summary=page_summary
                )

                # --- 3. Call Planner LLM & Parse ---
                response = await self.planner_llm.ainvoke([HumanMessage(content=prompt)])
                plan_update = json_repair.loads(response.content.strip())

                # --- 4. Apply Plan Update ---
                if isinstance(plan_update, dict) and "action" in plan_update:
                    action = plan_update.get("action")
                    logger.info(f"🧠 Planner decided to '{action}' the plan.")

                    if action == "add" and "step_description" in plan_update:
                        desc = plan_update["step_description"]
                        after_idx = plan_update.get("after_index") # 1-based index to insert after
                        insert_pos = int(after_idx) if after_idx is not None else len(manager.bu_plan)
                        manager.add_plan_step(desc, index=insert_pos)
                        self.heuristics.inject_message(f"SYSTEM (Planner): I've added a new step to the plan: '{desc}'")

                    elif action == "update" and "step_index" in plan_update and "status" in plan_update:
                        idx = int(plan_update["step_index"]) - 1 # Convert 1-based to 0-based
                        status = plan_update["status"]
                        reason = plan_update.get("reason", "")
                        manager.update_plan_step_status(idx, status, result=reason)
                        self.heuristics.inject_message(f"SYSTEM (Planner): I've updated step {idx + 1} to '{status}'. Reason: {reason}")
                
            except asyncio.CancelledError:
                logger.info("🧠 Planner Loop: Cancelled")
                break
            except Exception as e:
                logger.error(f"🧠 Planner Loop Error: {e}")
                await asyncio.sleep(10)

    async def _execute_initial_actions(self) -> List[str]:
        """
        Executes initial actions directly using the controller.
        This bypasses the standard agent loop for forced plan steps.
        Returns a list of result strings.
        """
        if not self.initial_actions:
            return []

        results = []
        for action_data in self.initial_actions:
            # Handle Pydantic models (ActionModel) if browser_use converted them
            if hasattr(action_data, "model_dump"):
                action_data = action_data.model_dump(exclude_none=True)
            elif hasattr(action_data, "dict"):
                action_data = action_data.dict(exclude_none=True)

            for action_name, params in action_data.items():
                try:
                    logger.info(f"🚀 Executing initial action: {action_name} with params: {params}")
                    if self.controller and hasattr(self.controller, "execute_action_by_name"):
                        result = await self.controller.execute_action_by_name(action_name, params, self.browser_context)
                        results.append(f"Action '{action_name}' executed. Result: {result}")
                        logger.info(f"✅ Result: {result}")
                    else:
                        results.append(f"Action '{action_name}' skipped (Controller missing).")
                except Exception as e:
                    logger.error(f"❌ Error executing initial action {action_name}: {e}")
                    results.append(f"Action '{action_name}' failed: {e}")
        return results

    async def _validate_output(self) -> bool:
        """
        Validate the output of the agent using the confirmer LLM if available.
        """
        if not self.confirmer_llm:
            return await super()._validate_output()

        self.is_validating = True
        try:
            logger.info(f"🔍 Validating output with Confirmer LLM (Strictness: {self.confirmer_strictness})")

            system_prompt = self._get_confirmer_prompt()
            messages = self._build_confirmer_messages(system_prompt)

            async def _call_llm():
                return await self.confirmer_llm.ainvoke(messages)

            response = await retry_async(
                _call_llm,
                logger=logger,
                error_message="Confirmer LLM call failed"
            )
            response_text = response.content.strip()
            logger.info(f"✅ Confirmer response: {response_text}")

            # Clean response of potential think tags (DeepSeek style)
            think_match = re.search(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else None

            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

            is_confirmed = clean_response.upper().startswith("YES")
            
            # Extract reason
            if is_confirmed:
                reason = clean_response[3:].strip(" .:-")
            elif clean_response.upper().startswith("NO"):
                reason = clean_response[2:].strip(" .:-")
            else:
                reason = clean_response

            if not reason:
                reason = "Confirmed by LLM." if is_confirmed else "Rejected by LLM."

            if self.validation_callback:
                try:
                    if asyncio.iscoroutinefunction(self.validation_callback):
                        await self.validation_callback(think_content, reason, is_confirmed)
                    else:
                        self.validation_callback(think_content, reason, is_confirmed)
                except Exception as e:
                    logger.error(f"Validation callback failed: {e}")

            # Update plan status immediately and handle dynamic plan changes
            if self.current_step_index is not None and self.controller and hasattr(self.controller, "webui_manager"):
                    manager = self.controller.webui_manager
                    if manager and hasattr(manager, "update_plan_step_status"):
                        
                        # 1. Check for Critical Blockers to update Plan (Dynamic Planning)
                        if not is_confirmed:
                            new_step_desc = None
                            reason_upper = reason.upper()
                            if "CAPTCHA" in reason_upper:
                                new_step_desc = "Solve CAPTCHA detected on page"
                            elif "LOGIN" in reason_upper or "SIGN IN" in reason_upper:
                                new_step_desc = "Perform Login/Authentication"
                            
                            if new_step_desc:
                                insert_idx = self.current_step_index + 1
                                manager.add_plan_step(new_step_desc, index=insert_idx)
                                manager.update_plan_step_status(self.current_step_index, "failed", result=f"Blocked: {reason}")
                                logger.info(f"🚨 Confirmer detected blocker. Added recovery step: '{new_step_desc}' at index {insert_idx}")
                                self.state.stopped = True # Stop current agent to switch to new step
                                return False

                        # 2. Normal Status Update
                        status = "completed" if is_confirmed else "in_progress"
                        display_result = reason
                        if not is_confirmed:
                            display_result = f"⚠️ Rejection: {reason}"
                        
                        manager.update_plan_step_status(self.current_step_index, status, result=display_result)
                        logger.info(f"✅ Confirmer updated plan step {self.current_step_index + 1}. Status: {status}. Reason: {reason}")

            if is_confirmed:
                return True
            
            self._handle_confirmer_rejection(reason)
            return False

        except Exception as e:
            logger.error(f"Error during confirmation: {e}")
            return True
        finally:
            self.is_validating = False

    def _get_confirmer_prompt(self) -> str:
        """Selects the appropriate system prompt based on strictness settings."""
        if self.confirmer_strictness and self.confirmer_strictness <= 3:
            return CONFIRMER_PROMPT_FAST.format(task=self.task)
        return CONFIRMER_PROMPT_STANDARD.format(task=self.task, strictness=self.confirmer_strictness or 5)

    def _build_confirmer_messages(self, system_prompt: str) -> list[BaseMessage]:
        """Constructs the message history for the confirmer LLM."""
        history = self.state.history
        last_item = history.history[-1] if history.history else None
        
        content = [{"type": "text", "text": f"User Task: {self.task}"}]

        if last_item:
            state_desc = f"Current URL: {last_item.state.url}\nPage Title: {last_item.state.title}\n"
            if last_item.model_output:
                thought = getattr(last_item.model_output, "thought", None)
                if thought:
                    state_desc += f"Agent Thought: {thought}\n"
            
            content.append({"type": "text", "text": state_desc})

            if last_item.state.screenshot and self.use_vision_for_confirmer:
                image_url = f"data:image/jpeg;base64,{last_item.state.screenshot}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        return [SystemMessage(content=system_prompt), HumanMessage(content=content)]

    def _handle_confirmer_rejection(self, response_text: str):
        """Handles the logic when the confirmer rejects the completion."""
        msg = f"Confirmer rejected completion: {response_text}"
        
        # Inject strategy hint to force re-assessment
        strategy_hint = " STRATEGY UPDATE: The previous action did NOT complete the task. You MUST change your approach. 1. Do NOT repeat the same action. 2. Use `assess_page_section` or `get_page_structure` to re-orient. 3. Check for popups or errors."
        full_msg = msg + strategy_hint

        self.heuristics.inject_message(full_msg)

    def inject_notification(self, content: str):
        """
        Injects a system notification into the message manager as a HumanMessage.
        This is used by heuristics to guide the agent.
        """
        if not hasattr(self, "message_manager"):
            return

        msg = HumanMessage(content=f"System Notification: {content}")

        try:
            if hasattr(self.message_manager, "add_message"):
                self.message_manager.add_message(msg)
            elif hasattr(self.message_manager, "add_new_task"):
                self.message_manager.add_new_task(msg.content)
            elif hasattr(self.message_manager, "messages") and isinstance(self.message_manager.messages, list):
                self.message_manager.messages.append(msg)
        except Exception as e:
            logger.warning(f"Failed to inject notification: {e}")

    def _manage_model_switching(self):
        """Handles Cost Saver and Smart Retry logic."""
        self.heuristics.manage_model_switching()
            
    def _suggest_alternative_strategy(self):
        """Injects strategy hints if the agent is failing."""
        self.heuristics.suggest_alternative_strategy()

    async def _execute_step_callbacks(self, step: int):
        """Executes registered step callbacks."""
        if hasattr(self, "step_callback"):
            try:
                if self.state.history.history:
                    last_state = self.state.history.history[-1].state
                    last_output = self.state.history.history[-1].model_output
                    if asyncio.iscoroutinefunction(self.step_callback):
                        await self.step_callback(last_state, last_output, step + 1)
                    else:
                        self.step_callback(last_state, last_output, step + 1)
            except Exception as e:
                logger.error(f"Error in step_callback: {e}")

    async def _execute_done_callback(self):
        """Executes registered done callbacks."""
        if hasattr(self, "done_callback"):
            try:
                if asyncio.iscoroutinefunction(self.done_callback):
                    await self.done_callback(self.state.history)
                else:
                    self.done_callback(self.state.history)
            except Exception as e:
                logger.error(f"Error in done_callback: {e}")

    async def _process_control_queue(self):
        """Processes commands from the control queue."""
        if not self.agent_control_queue:
            return

        while not self.agent_control_queue.empty():
            try:
                control_data = self.agent_control_queue.get_nowait()
                command = control_data.get("command")
                value = control_data.get("value")
                
                logger.info(f"Processing control command: {command}")

                if command == "pause":
                    self.pause()
                    if self.send_agent_message_callback:
                        await self.send_agent_message_callback({"type": "agent_status", "status": "Paused ⏸️"})
                elif command == "resume":
                    self.resume()
                    if self.send_agent_message_callback:
                        await self.send_agent_message_callback({"type": "agent_status", "status": "Participating"})
                elif command == "stop":
                    self.stop()
                elif command == "add_task":
                    if hasattr(self, "message_manager"):
                         self.message_manager.add_user_message(value)
                         self.heuristics.inject_message(f"SYSTEM: User added new task to queue: '{value}'. Update your plan.")
                
            except asyncio.QueueEmpty:
                break

    async def _handle_pre_step(self) -> bool:
        """
        Handles pre-step checks: pause, stop, model switching, failures.

        Also manages cookie persistence across sessions

        Returns True if the agent should stop/break the loop.
        """
        # Process control queue
        await self._process_control_queue()

        # Manage Model Switching (Cost Saver / Smart Retry)
        self._manage_model_switching()

        # Suggest alternative strategies on failure
        self._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self._manage_tabs()

        # Custom Memory Injection
        if self.use_custom_memory and self.memory_manager:
            await self._inject_site_knowledge(self.state, self.memory_manager, self.last_domain, self.heuristics)


        # Check if we should stop due to too many failures
        # Note: self.heuristics is a direct attribute of BrowserUseAgent
        if await self.heuristics.check_max_failures():
           return True

        # Check if MFA is required and wait for approval
        if self.heuristics.is_waiting_for_mfa:
            logger.info("MFA required, waiting for approval from another device...")
            if self.send_agent_message_callback:
                await self.send_agent_message_callback({

                    "type": "mfa_required",
                    "message": "MFA Required: Approve from your other device to continue."
           })
            while self.heuristics.is_waiting_for_mfa and not self.state.stopped:
                await asyncio.sleep(1)  # Check every second
            logger.info("MFA approved or agent stopped.")
            self.heuristics.is_waiting_for_mfa = False  # Reset flag

            # Verify there is still a valid page and browser context
            if not self.browser_context or not hasattr(self.browser_context, "get_current_page"):
                logger.warning("Browser context is no longer valid after MFA, stopping agent.")
                self.state.stopped = True
                return True

            return True

        # Check control flags before each step
        if self.state.stopped:
            logger.info('Agent stopped')
            return True

        while self.state.paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self.state.stopped:  # Allow stopping while paused
                return True
        # Load cookies at the beginning of each session
        if self.cookie_path and os.path.exists(self.cookie_path):
            cookies = await self._load_cookies(self.cookie_path)
            await self.browser_context.add_cookies(cookies)
            logger.info(f"Loaded cookies from {self.cookie_path}")

        return False

    async def _inject_site_knowledge(self, state, memory_manager, last_domain, heuristics):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        return await self.inject_site_knowledge(state, memory_manager, last_domain, heuristics)

    async def _manage_tabs(self):
        """Closes extraneous tabs to keep the agent focused."""
        if self.controller and hasattr(self.controller, "nav_controller"):
            await self.controller.nav_controller.manage_tabs(self.browser_context)

    async def _inject_site_knowledge(self):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        try:
            if not self.state.history.history:
                return
                
            current_url = self.state.history.history[-1].state.url
            if not current_url:
                return

            from urllib.parse import urlparse
            domain = urlparse(current_url).netloc.replace("www.", "")
            
            if domain != self.last_domain:
                self.last_domain = domain
                knowledge = self.memory_manager.get_site_knowledge(current_url)
                if knowledge:
                    msg = f"🧠 Memory Retrieval: I have previous knowledge about {domain}:\n{knowledge}"
                    self.heuristics.inject_message(msg)
                    logger.info(f"Injected site knowledge for {domain}")
        except Exception as e:
            logger.warning(f"Failed to inject site knowledge: {e}")

    async def _handle_post_step(self, step: int, max_steps: int) -> bool:
        """
        Handles post-step logic: callbacks, persistence, completion check.
        Returns True if the agent should stop/break the loop (task done).
        """
        # Execute WebUI Callbacks
        await self._execute_step_callbacks(step)

        # Persistence: Auto-save history after each step
        if self.save_history_path:
            try:
                await self.save_history_async(self.save_history_path)
            except Exception as e:
                logger.warning(f"Failed to auto-save history: {e}")

        # Heuristic check for completion (e.g. Quiz done)
        await self.heuristics.check_completion_heuristics()

        # Check for loops
        self.heuristics.detect_loop()

        # Check for progress indicators
        await self.heuristics.detect_progress()

        # Check for blocking elements
        await self.heuristics.check_blocking_elements()

        # Check navigation recovery
        await self.heuristics.check_navigation_recovery()
        
        await agent_utils.after_navigation_checks(self)

        # Check for complex page structure and add subtasks
        await self.heuristics.check_and_add_subtasks()

        # Check login status (New)
        await self.heuristics.check_login_status()

        # --- User Interaction Dialog ---
        if await self._handle_user_interaction():
            return True # Stop requested

        if self.state.history.is_done():
            if self.settings.validate_output and step < max_steps - 1:
                if not await self._validate_output():
                    return False # Validation failed, continue loop

            await self.log_completion()
            return True # Task done, break loop
        
        return False

    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.enable_user_interaction_dialog:
            return False

        last_step_info = self.state.history.history[-1] if self.state.history.history else None
        
        intel = "No specific new information from this step."
        if last_step_info and last_step_info.model_output and last_step_info.model_output.thought:
            intel = last_step_info.model_output.thought
        
        next_action_desc = "Agent is considering its next move."
        if last_step_info and last_step_info.model_output and last_step_info.model_output.action:
            actions_list = last_step_info.model_output.action if isinstance(last_step_info.model_output.action, list) else [last_step_info.model_output.action]
            action_strings = []
            for action_model in actions_list:
                if hasattr(action_model, "name") and hasattr(action_model, "args"):
                    action_strings.append(f"{action_model.name}({json.dumps(action_model.args)})")
                else:
                    action_strings.append(str(action_model)) # Fallback for non-standard actions
            next_action_desc = ", ".join(action_strings)
            if not next_action_desc:
                next_action_desc = "Agent is considering its next move."
        
        user_decision = await self.request_user_decision(intel, next_action_desc)

        if user_decision is False: # User chose No/Abort
            self.state.stopped = True
            return True # Break the loop
        elif isinstance(user_decision, str): # User provided a custom task
            self.task = user_decision # Update the main task for the agent to re-evaluate
            self.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")
            return False # Continue the loop, agent will now work on the new task
        
        return False

    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Requests confirmation/input from the user via WebSocket.
        """
        if not self.send_agent_message_callback or not self.confirmation_response_queue:
            return True

        await self.send_agent_message_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review.",
            "intel": intel,
            "next_task": next_task
        })

        while True:
            # Create tasks for both queues
            confirmation_task = asyncio.create_task(self.confirmation_response_queue.get())
            control_task = asyncio.create_task(self.agent_control_queue.get()) if self.agent_control_queue else None
            
            tasks = [confirmation_task]
            if control_task:
                tasks.append(control_task)
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if confirmation_task in done:
                response_data = confirmation_task.result()
                response = response_data.get("response")
                custom_task = response_data.get("custom_task")

                if response == "yes": return True
                if response == "no": return False
                if custom_task: return custom_task
            
            if control_task and control_task in done:
                control_data = control_task.result()
                command = control_data.get("command")
                
                if command == "pause":
                    self.pause()
                    logger.info("Agent paused via user request during confirmation.")
                    if self.send_agent_message_callback:
                        await self.send_agent_message_callback({"type": "agent_status", "status": "Paused ⏸️"})
                elif command == "resume":
                    self.resume()
                    logger.info("Agent resumed via user request during confirmation.")
                    if self.send_agent_message_callback:
                        await self.send_agent_message_callback({"type": "agent_status", "status": "Participating"})
                elif command == "stop":
                    self.stop()
                    return False

    def _handle_max_steps_reached(self):
        """Handles the case where the agent reaches maximum steps without completion."""
        error_message = 'Failed to complete task in maximum steps'
        self.state.history.history.append(
            AgentHistory(
                model_output=None,
                result=[ActionResult(error=error_message, include_in_memory=True)],
                state=BrowserStateHistory(
                    url='',
                    title='',
                    tabs=[],
                    interacted_element=[],
                    screenshot=None,
                ),
                metadata=None,
            )
        )
        logger.info(f'❌ {error_message}')

    async def save_history_async(self, path: str):
        """Saves the agent history to a file asynchronously."""
        try:
            await IOManager.write_file(path, self.state.history.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save history asynchronously: {e}")

    async def _handle_cleanup(self):
        """Handles cleanup after run."""
        if self.planner_task and not self.planner_task.done():
            self.planner_task.cancel()
            try:
                await self.planner_task
            except asyncio.CancelledError:
                pass

        if self.inhibit_close:
            logger.info("🚫 Cleanup inhibited: Browser will remain open.")
        else:
            try:
                await self.close()
            except Exception as e:
                # Log as debug to avoid noise during forced shutdowns/interrupts
                logger.debug(f"Error during agent cleanup (likely benign): {e}")

        if self.settings.generate_gif:
            output_path: str = 'agent_history.gif'
            if isinstance(self.settings.generate_gif, str):
                output_path = self.settings.generate_gif
            create_history_gif(task=self.task, history=self.state.history, output_path=output_path)

        # Execute Done Callback
        await self._execute_done_callback()
