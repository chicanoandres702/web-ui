from __future__ import annotations
import abc
import asyncio
from asyncio.log import logger
import inspect
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast
import json
import logging
import os
import re
from pathlib import Path
import sys

import json_repair
from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda

sys.path.append(str(Path(__file__).resolve().parents[3]))
from browser_use.agent.gif import create_history_gif
from browser_use.agent.views import (
    ActionResult,
    AgentHistoryList,
    AgentOutput,
    BrowserStateHistory,
    AgentStepInfo)
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

from browser_use.agent.service import Agent, AgentHookFunc  # AIMessage is from langchain_core.messages
from browser_use.agent.views import AgentHistory
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD
from src.agent.browser_use import agent_utils
from src.agent.browser_use.agent_components import CookieHandler
from src.agent.browser_use.agent_heuristics import AgentHeuristics
from src.agent.browser_use import agent_utils  # Import the new module
from src.agent.browser_use.components.site_knowledge_injector import SiteKnowledgeInjector

from src.agent.browser_use.components.planner import AgentPlanner
from src.agent.browser_use.components.browser_factory import BrowserFactory
from src.utils.io_manager import IOManager

from src.agent.browser_use.user_interaction_handler import UserInteractionHandler
from src.agent.browser_use.components.knowledge_base_manager import KnowledgeBaseManager


class InstructionHandlerInterface(abc.ABC):
    @abc.abstractmethod
    async def handle_instructions(self, task: Any, decision: Dict[str, str]) -> str:
        pass


def _settings_to_dict(settings: Any) -> Dict[str, Any]:
    """
    Converts settings to a dictionary, handling RunnableSequence objects.
    """
    if isinstance(settings, dict): # type: ignore
        return settings.copy()
    elif hasattr(settings, "dict") and callable(settings.dict):
        return cast(Dict[str, Any], settings.dict())
    else:
        logger.warning(f"Settings object is not a dictionary.  Returning empty dictionary.")
        return {}

load_dotenv()

class AgentLLMManager:
    """Manages LLM-related operations for the agent, including model switching and patching."""
    # The AgentLLMManager is responsible for handling LLM related tasks such as:
    # -Setting the LLM for the agent
    # -Applying fixes to the LLM responses
    # -Setting the tool calling method
    def __init__(self, agent: Any, model_priority_list: Optional[List[Dict]] = None):
        self.agent = agent
        self.model_priority_list = model_priority_list
    
    def set_llm(self, llm):
        """Sets the current LLM for the agent."""
        self.agent.llm = llm

    def get_current_llm(self):
        """Returns the current LLM."""
        return self.agent.llm

    def _patch_llm_with_qwen_fix(self, llm: Runnable) -> Runnable:
        """Wraps the LLM with a RunnableLambda that applies Qwen/Ollama JSON fix logic to handle responses which require JSON fixes."""
        # This step converts the LLM output to an AIMessage format
        llm_with_aimessage_output = llm | RunnableLambda(self._convert_agent_output_to_aimessage)
        # This step applies the actual fix
        return llm_with_aimessage_output | RunnableLambda(self._apply_qwen_fix_logic)

    def _convert_agent_output_to_aimessage(self, agent_output: AgentOutput) -> BaseMessage:
        """Converts structured agent output to an AIMessage, extracting tool calls for structured tool use."""
        tool_calls = []
        json_content = agent_output.model_dump_json()
        if agent_output.action:
            actions_list: List[Any] = agent_output.action if isinstance(agent_output.action, list) else [agent_output.action]
            for action_model in actions_list:
                if hasattr(action_model, "model_dump"):
                    action_args = action_model.model_dump(exclude_unset=True)
                    tool_name = action_args.pop("name", None)
                    if tool_name:
                        tool_calls.append({"name": tool_name, "args": action_args})
        return AIMessage(content=json_content, tool_calls=tool_calls)

    def _apply_qwen_fix_logic(self, response: BaseMessage) -> BaseMessage:
        """Logic to fix potential JSON formatting issues, which is common in Qwen models, ensuring proper data structure."""
        """Applies the Qwen/Ollama JSON fix logic to an LLM response (BaseMessage)."""
        if not hasattr(response, 'content') or not isinstance(response.content, str): # type: ignore
            return response

        return response


    def _set_tool_calling_method(self) -> str | None: # type: ignore
        """Automatically determines and sets the correct tool calling method based on the LLM being used, and the agent settings."""
        """Sets the tool calling method based on model and configuration, with auto-detection for Ollama and other models."""
        settings = getattr(self.agent, 'settings', None) # type: ignore
        if not settings: # type: ignore
            return None

        tool_calling_method = getattr(settings, 'tool_calling_method', 'auto')
        chat_model_library = getattr(self.agent, 'chat_model_library', '')
        model_name = getattr(self.agent, 'model_name', '').lower()

        if tool_calling_method == 'auto':
            if 'Ollama' in chat_model_library:
                return None
            elif any(m in model_name for m in ['qwen', 'deepseek', 'gpt', 'claude']) and 'Ollama' not in chat_model_library:
                return 'function_calling'
            return None
        else:
            return tool_calling_method


    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]: # type: ignore
        """
        Asynchronously requests a decision from the user, using a WebSocket to communicate the request and await the response.
        Requests confirmation/input from the user via WebSocket.
        """
        if not self.agent.send_agent_message_callback or not self.agent.confirmation_response_queue: # type: ignore
            return True

        await self.agent.send_agent_message_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review.",
            "intel": intel,
            "next_task": next_task
        })

        while True:
            # Create tasks for both queues
            confirmation_task = asyncio.create_task(self.agent.confirmation_response_queue.get())
            control_task = asyncio.create_task(self.agent.agent_control_queue.get()) if self.agent.agent_control_queue else None

            tasks = [confirmation_task]
            if control_task:
                tasks.append(control_task)

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in pending: # type: ignore
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
                command = control_data.get("command") # type: ignore
                
                if command == "pause":
                    self.agent.pause()
                    logger.info("Agent paused via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Paused ⏸️"})
                elif command == "resume":
                    self.agent.resume()
                    logger.info("Agent resumed via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Participating"})
                elif command == 'stop':
                    self.agent.stop()
                    return False


ToolCallingMethod = str

def _prepare_agent_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares and validates the agent settings before passing them to AgentSettings.
    """
    if isinstance(settings, dict): # type: ignore
        prepared_settings = settings.copy()

        # Convert relevant arguments to the expected type
        for key in ['max_steps', 'confirmer_strictness']:
            if key in prepared_settings and isinstance(prepared_settings[key], str):
                try:
                    prepared_settings[key] = int(prepared_settings[key])
                except ValueError:
                    logger.warning(f"Could not convert {key} to int, using default value.")
                    prepared_settings.pop(key)
        return prepared_settings
    return {}


class BrowserUseAgent(Agent):

    def __init__(
        self,
        state,
        version: str,
        settings:Dict[str, Any],
        task,
        controller,

        message_manager,
        confirmer_strictness=7,

        model_priority_list: Optional[List[Dict]] = None,
        auto_save_on_stuck=True,
        system_prompt: Optional[str] = None,
        validation_callback=None,
        browser_context=None,

        send_agent_message_callback: Optional[Callable] = None,
        confirmation_event: Optional[asyncio.Event] = None,
        confirmation_response_queue: Optional[asyncio.Queue] = None,
        enable_smart_retry=False,
        enable_cost_saver=False,
        confirmer_llm=None,
        instruction_handler: Optional[InstructionHandlerInterface] = None,
        **kwargs
    ):
        self.version = version # type: ignore
        self.last_domain = None
        self.task = task
        self.state = state
        self.browser_context = browser_context
        self.send_agent_message_callback = send_agent_message_callback
        self.confirmation_response_queue = confirmation_response_queue
        self.successful_steps_since_switch = 0
        self.switched_to_retry_model = False
        self.planner_task: Optional[asyncio.Task] = None
        self.settings = settings

        agent_kwargs = kwargs.copy() # type: ignore
        self.use_custom_memory = agent_kwargs.pop('use_memory', True) # type: ignore
        # Extract agent-specific args that should not be passed to the base Agent classs
        self.save_history_path = agent_kwargs.pop('save_history_path', None) # type: ignore
        self.planner_llm = agent_kwargs.pop('planner_llm', None)
        self.planner_interval = agent_kwargs.pop('planner_interval', 5.0)
        self.cookie_path = agent_kwargs.pop('cookie_path', "./tmp/cookies.json")
        self.initial_actions = agent_kwargs.pop('initial_actions', None)

        self.agent_control_queue = agent_kwargs.pop('agent_control_queue', None)
        if confirmer_llm:
            agent_kwargs['validate_output'] = True # type: ignore

        self.validation_callback = validation_callback
        self.step_callback: Optional[Callable] = kwargs.get("step_callback")

        self.done_callback: Optional[Callable] = kwargs.get("done_callback")

        self.inhibit_close = agent_kwargs.pop('inhibit_close', False)
        self.enable_smart_retry = enable_smart_retry
        self.enable_cost_saver = enable_cost_saver
        self.enable_user_interaction_dialog = agent_kwargs.pop('enable_user_interaction_dialog', True)
        self.auto_save_on_stuck = auto_save_on_stuck

        self.instruction_handler = instruction_handler # type: ignore

        # Safely get the llm
        original_llm = agent_kwargs.get('llm')
        if not original_llm:
            raise ValueError("LLM must be provided in agent_kwargs or model_priority_list")
        
        # Patch LLM if it's a model that needs JSON fixes (e.g., Qwen/Ollama)
        self.llm_manager = AgentLLMManager(self, model_priority_list)

        # Extract raw LLM from RunnableSequence if necessary to check model name
        raw_llm = original_llm
        if hasattr(original_llm, "first"):
            raw_llm = original_llm.first

        model_name = getattr(raw_llm, "model_name", "").lower() or getattr(raw_llm, "model", "").lower()

        # This instructs the LLM to produce output conforming to AgentOutput -type: ignore
        if hasattr(original_llm, "with_structured_output"):
            structured_llm = original_llm.with_structured_output(AgentOutput)
            logger.info("Applied AgentOutput schema to LLM using with_structured_output.") #type: ignore
        else:
            logger.warning("LLM does not support with_structured_output. Proceeding without structured output enforcement.")
            structured_llm = original_llm
        if any(m in model_name for m in ['qwen', 'ollama']):
            structured_llm = self.llm_manager._patch_llm_with_qwen_fix(structured_llm)
        agent_kwargs['llm'] = structured_llm


        # super().__init__(**agent_kwargs)
        super().__init__(task=task, **agent_kwargs)
        self.state = state if state is not None else BrowserStateHistory(url='', title='', tabs=[], interacted_element=[], screenshot=None) #type: ignore

        self.heuristics = AgentHeuristics(self)
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness: int = confirmer_strictness
        self.controller = controller

        if model_priority_list and model_priority_list[0].get('llm'):

            self.llm_manager.set_llm(model_priority_list[0]['llm'])
        else:
            self.llm_manager.set_llm(agent_kwargs['llm'])
        
        self._initialize_internal_components(agent_kwargs, system_prompt)
        

    def _initialize_internal_components(self, agent_kwargs, system_prompt):
        """Initializes modular components for the agent."""
        if not self.browser_context:
            return # No browser context to work with
        self.control_queue_handler = ControlQueueHandler(self)

        self.cleanup_handler = AgentCleanupHandler(self)
        self.action_handler = BrowserActionHandler(self.browser_context, self.controller, self.heuristics)
        self.post_step_handler = PostStepHandler(self)
        self.user_interaction_handler = UserInteractionHandler(self)
        self.site_knowledge_injector = SiteKnowledgeInjector(self)
        self.knowledge_base_manager = KnowledgeBaseManager(base_dir=agent_kwargs.get('kb_dir', './tmp/memory'))

        if system_prompt:
            ComponentInitializer(self, agent_kwargs, system_prompt) # type: ignore


    def _set_tool_calling_method(self) -> str:
        return self.llm_manager._set_tool_calling_method() or "auto" # type: ignore
    
    @property # type: ignore
    def _tool_calling_supported(self) -> bool:
        # Whether tool calling is supported by the agent.
        return self._set_tool_calling_method() == "function_calling" # type: ignore
    def _initialize_components(self, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None) -> None:

        """
        Initializes agent components and applies system prompt override.
        """
    
        # ComponentInitializer(self, agent_kwargs, system_prompt) # type: ignore

        # Initialize AgentLoopHandler
        self.planner_task = None


        # Initialize SiteKnowledgeInjector
        self.site_knowledge_injector = SiteKnowledgeInjector(self)
        
    async def run(self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None) -> AgentHistoryList:


        loop = asyncio.get_event_loop()

        # Execute initial actions if provided
        if hasattr(self, 'initial_actions') and self.initial_actions:
            # Use custom execution to ensure actions are run reliably via controller
            results = await self._execute_initial_actions()
            if results:
                # If initial actions are provided, run them before the main loop
                # Inject into message history so the agent knows what happened
                summary = "\n".join(results) # type: ignore
                msg = f"SYSTEM NOTE: The following actions were already executed for this step:\n{summary}\n\nCheck the current state. If the goal is achieved, output {{'finish': ...}}. Do NOT repeat the above actions."
                if hasattr(self, "message_manager") and self.message_manager:
                    add_msg_func = getattr(self.message_manager, "add_message", None)
                    if callable(add_msg_func):
                        add_msg_func(HumanMessage(content=msg))
                # BrowserStateHistory does not have last_result; it is stored in the Agent's state
                if hasattr(self, 'last_result'):
                    self.last_result = [ActionResult(extracted_content=summary, include_in_memory=True)]

        if self.send_agent_message_callback:

            await self.send_agent_message_callback({"type": "agent_status", "status": "Participating"})

        try:
            self._log_agent_run()

            # Start Planner Loop if enabled

            for step in range(max_steps): # type: ignore
                # Main agent loop
                try:
                    if await self._handle_pre_step():
                        break

                    if on_step_start is not None:
                        await on_step_start(self)

                    step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                    await self._execute_agent_step(step_info)

                    if on_step_end is not None:
                        await on_step_end(self)

                    if await self.post_step_handler.handle_post_step(step, max_steps):
                        break
                except Exception as e:
                    logger.error(f"Error in step {step}: {e}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        break

                    # If a custom task was provided, the agent.task would have been updated.
                    # We should continue the loop to process the new task. # type: ignore
            else:
                await self.cleanup_handler._handle_max_steps_reached()
                # If agent does not complete the task in the given number of steps, handle clean up

            return self.state.history

        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history
        except asyncio.CancelledError:

            logger.info('Agent execution cancelled')
            return self.state.history
        finally:
            await self.cleanup_handler._handle_shutdown()

    
    async def _execute_agent_step(self, step_info: AgentStepInfo):
        """Executes a single step of the agent."""
        try:
            await self.step(step_info)
        except Exception as e:
            if "Timeout" in str(e) and "exceeded" in str(e):
                logger.warning(f"⚠️ Step {step_info.step_number + 1} timed out: {e}. Attempting to recover by reloading page.")
                self.state.consecutive_failures += 1

            else:
                logger.error(f"An unexpected error occurred during step {step_info.step_number + 1}: {e}")
                self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step_info.step_number + 1}: {e}")
                raise e

    async def _recover_from_timeout(self):
        """Attempts to recover from a timeout error by reloading the page."""
        try:
            if self.browser_context:
                page = await self.browser_context.get_current_page()
                await page.reload(wait_until="domcontentloaded", timeout=15000)
                self.heuristics.inject_message("SYSTEM: A timeout occurred. I have reloaded the page to recover. Please re-assess the current state.")
        except Exception as reload_e:

            logger.error(f"Failed to reload page after timeout: {reload_e}")
            

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
                if not self.controller or not hasattr(self.controller, "webui_manager"): # type: ignore
                    continue
                manager = getattr(self.controller, "webui_manager", None)
                if not manager or not hasattr(manager, "bu_plan") or not manager.bu_plan:
                    continue

                # --- 1. Gather Context for Planner ---
                last_step = self.state.history.history[-1]
                plan_str = json.dumps(manager.bu_plan, indent=2)
                # BrowserStateHistory doesn't have element_tree directly, 
                # it's usually in the state object of the history item
                page_summary = ""
                # BrowserStateHistory stores the state in a way that might require 
                # checking for element_tree in the underlying state object if available

                last_thought = getattr(last_step.model_output, "thought", "N/A") if last_step.model_output else "N/A"

                # --- 2. Construct Prompt ---
                # Fallback prompt if PLANNER_PROMPT is not found in src.utils.prompts
                DEFAULT_PLANNER_PROMPT = "Goal: {goal}\nPlan: {plan}\nLast Thought: {last_thought}\nPage Summary: {page_summary}\nUpdate the plan."
                from src.utils import prompts # type: ignore
                PLANNER_PROMPT = getattr(prompts, 'PLANNER_PROMPT', DEFAULT_PLANNER_PROMPT) # type: ignore

                prompt = PLANNER_PROMPT.format(
                    goal=self.task,
                    plan=plan_str,
                    last_thought=last_thought,
                    page_summary=page_summary
                )

                # --- 3. Call Planner LLM & Parse -type: ignore---
                response = await self.planner_llm.ainvoke([HumanMessage(content=prompt)])
                plan_update = json_repair.loads(response.content.strip())

                # --- 4. Apply Plan Update ---
                if isinstance(plan_update, dict) and "action" in plan_update:
                    action = plan_update.get("action")
                    logger.info(f"🧠 Planner decided to '{action}' the plan.")

                    if action == "add" and "step_description" in plan_update:
                        desc = plan_update["step_description"] # type: ignore
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
        # Executes initial, forced actions using the controller to bypass the agent loop
        """
        execute initial actions directly using the controller.
        This bypasses the standard agent loop for forced plan steps.
        Returns a list of result strings.
        """
        if not self.initial_actions:
            return []

        results = []
        for action_data in self.initial_actions:
            # Handle Pydantic models (ActionModel) if browser_use converted them
            if hasattr(action_data, 'model_dump'):
                action_data = action_data.model_dump(exclude_none=True)
            elif hasattr(action_data, 'dict'):
                action_data = action_data.dict(exclude_none=True)

            if isinstance(action_data, dict):
                for action_name, params in action_data.items():
                    try:
                        logger.info(f"🚀 Executing initial action: {action_name} with params: {params}")
                        execute_func = getattr(self.controller, "execute_action_by_name", None)
                        if self.controller and callable(execute_func):
                            result = execute_func(action_name, params, self.browser_context)
                            if asyncio.iscoroutine(result) or inspect.isawaitable(result): # type: ignore
                                result = await result
                                
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
        Validates the agent's output using a confirmer LLM, if configured, to ensure quality and correctness.
        Validate the output of the agent using the confirmer LLM if available.
        """
        if not self.confirmer_llm:
            return await super()._validate_output()

        try:
            logger.info(f"🔍 Validating output with Confirmer LLM (Strictness: {self.confirmer_strictness})")

            system_prompt = self._get_confirmer_prompt()
            messages = self._build_confirmer_messages(system_prompt)

            async def _call_llm():
                if self.confirmer_llm is None:
                    raise ValueError("Confirmer LLM is not initialized.")
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

            if self.validation_callback:
                try:
                    await self.validation_callback(think_content, clean_response, is_confirmed)
                except Exception as e:
                    logger.error(f"Validation callback failed: {e}")

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

            if not is_confirmed:
                reason_upper = reason.upper()

                new_step_desc = None
                if "CAPTCHA" in reason_upper:
                    new_step_desc = "Solve CAPTCHA detected on page"
                elif "LOGIN" in reason_upper or "SIGN IN" in reason_upper:
                    new_step_desc = "Handle login requirement"

                step_index = getattr(self.state, 'current_step_index', 0)
                if new_step_desc:
                    self.state.stopped = True
                    return False
                return True

            self._handle_confirmer_rejection(reason)
            return False
        except Exception as e:
            logger.error(f"Error during confirmation: {e}")
            return True
        finally:
            self.is_validating = False

    def _get_confirmer_prompt(self) -> str:
        # Prompt helper to select the correct prompt based on strictness settings

        if self.confirmer_strictness and self.confirmer_strictness <= 3:
            return CONFIRMER_PROMPT_FAST.format(task=self.task)
        return CONFIRMER_PROMPT_STANDARD.format(task=self.task, strictness=self.confirmer_strictness or 5)


    def _build_confirmer_messages(self, system_prompt: str) -> List[BaseMessage]:

        history = self.state.history
        last_item = history.history[-1] if history.history else None

        if last_item:
            state_desc = f"Current URL: {last_item.state.url}\nPage Title: {last_item.state.title}\n"
            if last_item.model_output:
                thought = getattr(last_item.model_output, "thought", None)
                if thought:
                    state_desc += f"Agent Thought: {thought}\n"
            content = list(content) + [{"type": "text", "text": state_desc}]

            if last_item.state.screenshot and getattr(self.settings, 'use_vision_for_confirmer', True):
                image_url = f"data:image/jpeg;base64,{last_item.state.screenshot}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        return [SystemMessage(content=system_prompt), HumanMessage(content=content)]

    def _handle_confirmer_rejection(self, response_text: str):
        # Handles confirmer rejection injecting strategic hints to prevent the agent from repeating the same mistake.
        """Handles the logic when the confirmer rejects the completion."""
        msg = f"Confirmer rejected completion: {response_text}"

        # Inject strategy hint to force re-assessment
        strategy_hint = " STRATEGY UPDATE: The previous action did NOT complete the task. You MUST change your approach. 1. Do NOT repeat the same action. 2. Use `assess_page_section` or `get_page_structure` to re-orient. 3. Check for popups or errors."
        full_msg = msg + strategy_hint
  
        self.heuristics.inject_message(full_msg)


        """
        # Inject system notification message in cases where it is needed for heuristics
        Injects a system notification into the message manager as a HumanMessage.
        This is used by heuristics to guide the agent.
        """
        if not hasattr(self, "message_manager"):
            return

        msg = HumanMessage(content=f"System Notification: {content}")

        try:
            # Use getattr to avoid static analysis errors for unknown attributes on MessageManager
            add_msg_func = getattr(self.message_manager, "add_message", None)
            # print(f"trying add messafe", add_msg_func)
            if callable(add_msg_func):
                add_msg_func(msg)
            else:
                add_new_task_func = getattr(self.message_manager, "add_new_task", None)
                if callable(add_new_task_func):
                    add_new_task_func(msg.content)
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

        if self.step_callback and callable(self.step_callback):
            try:
                if self.state.history.history:

                    last_state = self.state.history.history[-1].state
                    last_output = self.state.history.history[-1].model_output
                    if inspect.iscoroutinefunction(self.step_callback):
                        await self.step_callback(last_state, last_output, step + 1)
                    elif callable(self.step_callback):
                        self.step_callback(last_state, last_output, step + 1)

            except Exception as e:
                logger.error(f"Error in step_callback: {e}")

    async def _execute_done_callback(self):
        """Executes registered done callbacks."""
        if self.done_callback and callable(self.done_callback):
            try:
                if inspect.iscoroutinefunction(self.done_callback):
                    await self.done_callback(self.state.history)
                elif callable(self.done_callback): # type: ignore
                    self.done_callback(self.state.history)
            except Exception as e:
                logger.error(f"Error in done_callback: {e}")

    async def _handle_pre_step(self) -> bool:
        # Handles pre-step operations such as loading cookies, managing tabs, and injecting site knowledge.
        """
        Handles pre-step checks: pause, stop, model switching, failures.

        Also manages cookie persistence across sessions

        Returns True if the agent should stop/break the loop.
        """
        # Await self._process_control_queue()
        await self.control_queue_handler._process_control_queue()

        # Suggest alternative strategies on failure
        # Manage Model Switching (Cost Saver / Smart Retry)
        self.heuristics.manage_model_switching()

                # Suggest alternative strategies on failure
        self._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self.tab_handler._manage_tabs()

        # Custom Memory Injection
        # if self.use_custom_memory and self.memory_manager:
        await self.site_knowledge_injector.inject_site_knowledge()


        # Check if we should stop due to too many failures
        # Note: self.heuristics is a direct attribute of BrowserUseAgent
        # if await self.heuristics.check_max_failures():
        #    return True

        # Check if MFA is required and wait for approval

        if self.heuristics.is_waiting_for_mfa:
            logger.info("MFA required, waiting for approval from another device...")
        if self.send_agent_message_callback:
            if self.send_agent_message_callback:
                await self.send_agent_message_callback({ # type: ignore
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

            if self.browser_context:   

                try:
                    with open(self.cookie_path, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                    await self.browser_context.add_cookies(cookies)
                    logger.info(f"Loaded cookies from {self.cookie_path}")
                except Exception as e:
                    logger.error(f"Failed to load cookies: {e}")
        return False

class PostStepHandler:
    """Handles post-step logic: callbacks, persistence, completion check."""
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def handle_post_step(self, step: int, max_steps: int) -> bool:
        await self.agent._execute_step_callbacks(step)

        # Persistence: Auto-save history after each step
        if self.agent.save_history_path and hasattr(self.agent.cleanup_handler, 'save_history_async'):
            try:
                await self.agent.cleanup_handler.save_history_async(self.agent.save_history_path)
            except Exception as e:
                logger.warning(f"Failed to auto-save history: {e}")
        await self.agent.heuristics.check_completion_heuristics()
        # Check for loops
        self.agent.heuristics.detect_loop()
  
        # Check for progress indicators
        await self.agent.heuristics.detect_progress()

        # Check for blocking elements
        await self.agent.heuristics.check_blocking_elements()
  
        # Check navigation recovery
        await self.agent.heuristics.check_navigation_recovery()

  
        # Check for complex page structure and add subtasks
        await self.agent.heuristics.check_and_add_subtasks()

        # completion_check_handler = CompletionCheckHandler(self)
        # await completion_check_handler.check_completion()
        completion_check_handler = CompletionCheckHandler(self.agent)
        await completion_check_handler.check_completion()

  
        # Check login status (New)
        await self.agent.heuristics.check_login_status()

        if self.agent.cookie_path:
            await self.action_handler.save_cookies(self.cookie_path)

        # --- User Interaction Dialog ---
        if await self._handle_user_interaction():
            return True # Stop requested
        else:
            if await completion_check_handler.check_completion():
                return False # Validation failed, continue loop
            if self.agent.send_agent_message_callback:
                await self.agent.send_agent_message_callback({
                    "type": "agent_status",
                    "status": "Completed ✅",
                    "message": "Task completed successfully."
                })
            await self.agent.cleanup_handler.cleanup()
            return True # Task done, break loop

        return False
    
    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        """
        return await self.user_interaction_handler.handle_user_interaction()

    async def _request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Internal wrapper for requesting user decision.
        """
        return await self.agent.llm_manager.request_user_decision(intel, next_task)   

    async def _handle_shutdown(self) -> None:
        """Handles saving cookies on shutdown if a browser context exists."""
        if self.agent.browser_context:
            await self.agent.cookie_handler.save_cookies(self.agent.browser_context, self.agent.cookie_path)

class BrowserActionHandler:
    """Handles browser-specific actions like cookie management."""
    def __init__(self, browser_context: Any, controller: Any, heuristics: Any):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics

    async def save_cookies(self, cookie_path: str) -> None:
        """Saves cookies from the browser context to a file."""
        cookies = await self.browser_context.get_cookies()
        with open(cookie_path, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, indent=2)

class CompletionCheckHandler:
    """Handles the logic for checking if a task is completed and validating the output."""
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def check_completion(self) -> bool:
        """
        Checks if the agent has marked the task as done and performs validation.
        """
        if self.agent.state.history.is_done():
            if self.agent.confirmer_llm:
                if not await self.agent._validate_output(): # type: ignore
                    return False  # Validation failed, continue loop
            
            await self.agent.log_completion()
            return True
        return False


class ComponentInitializer:
    """Initializes and configures agent components."""
    def __init__(self, agent: BrowserUseAgent, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None):
        self.agent = agent
        self._setup_system_prompt(system_prompt)

    def _setup_system_prompt(self, system_prompt: Optional[str]):
        if system_prompt:
            self.agent.message_manager.system_prompt = SystemMessage(content=system_prompt)

class AgentCleanupHandler: # type: ignore
    """Handles cleanup tasks for the agent."""
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def _manage_tabs(self):

        """Closes extraneous tabs to keep the agent focused."""
        nav_controller = getattr(self.agent.controller, 'nav_controller', None)
        if nav_controller:
            await nav_controller.manage_tabs(self.agent.browser_context)

    async def _inject_site_knowledge(self):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        await self.agent.site_knowledge_injector.inject_site_knowledge()

    async def cleanup(self):
        """Performs cleanup of tasks and resources."""
        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel()
            try:
                await self.agent.planner_task
            except asyncio.CancelledError:   
                pass

        if not self.agent.inhibit_close:            
            if self.agent.browser_context:
                try:
                    await self.agent.browser_context.close()
                except Exception as e:
                    logger.warning(f"Failed to close browser context: {e}")

        if self.agent.settings and getattr(self.agent.settings, 'generate_gif', False):
            output_path = 'agent_history.gif'
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(task=self.agent.task, history=self.agent.state.history, output_path=output_path)

        # Execute Done Callback
        if hasattr(self.agent, '_execute_done_callback'):
            await self.agent._execute_done_callback()

        await self.agent._execute_done_callback()
    
    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = self.agent.state.history.history[-1] if self.agent.state.history.history else None
        intel = "No specific new information from this step."
        if last_step_info and last_step_info.model_output:
            # Use getattr to safely access 'thought' as it might not be in the schema 
            thought = getattr(last_step_info.model_output, "thought", None)
            if thought:
                intel = thought
        
        next_action_desc = "Agent is considering its next move." # type: ignore
        if (last_step_info and last_step_info.model_output and
                last_step_info.model_output.action):
            action_strings = []
            actions = last_step_info.model_output.action if isinstance(last_step_info.model_output.action, list) else [last_step_info.model_output.action]
            for action_model in actions:
                if hasattr(action_model, "model_dump"):
                    action_strings.append(str(action_model.model_dump()))
                else:
                    action_strings.append(str(action_model))
            next_action_desc = ", ".join(action_strings)
        if not next_action_desc:
            next_action_desc = "Agent is considering its next move."
        handler = UserInteractionHandler(self.agent)
        user_decision = await self.agent.llm_manager.request_user_decision(intel, next_action_desc)

        if user_decision is False: # type: ignore # User chose No/Abort
            self.agent.state.stopped = True
            return True # Break the loop
        elif isinstance(user_decision, str): # User provided a custom task
            self.agent.task = user_decision # Update the main task for the agent to re-evaluate
            self.agent.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")
            return False # Continue the loop, agent will now work on the new task
        
        return False

    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Requests confirmation/input from the user via WebSocket.
        """
        if not self.agent.send_agent_message_callback or not self.agent.confirmation_response_queue:
            return True
        await self.agent.send_agent_message_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review.",
            "intel": intel,
            "next_task": next_task
        })

        while True:
            # Create tasks for both queues
            confirmation_task = asyncio.create_task(self.agent.confirmation_response_queue.get())
            control_task = asyncio.create_task(self.agent.agent_control_queue.get()) if self.agent.agent_control_queue else None

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
                    self.agent.pause()
                    logger.info("Agent paused via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Paused ⏸️"})
                elif command == "resume":
                    self.agent.resume()
                    logger.info("Agent resumed via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Participating"})

                elif command == 'stop':
                    self.agent.stop()
                    return False

    async def _handle_max_steps_reached(self) -> None:
        """Handles the scenario where the agent reaches the maximum number of steps."""
        logger.info("Reached maximum number of steps.")
        if self.agent.send_agent_message_callback:
            await self.agent.send_agent_message_callback({
                "type": "agent_status",
                "status": "Max steps reached 🏁",
                "message": "The agent has reached the maximum number of steps allowed for this task."
            })

    async def save_history_async(self, path: str):
        """Saves the agent history to a file asynchronously."""
        try:
            await IOManager.write_file(path, self.agent.state.history.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save history asynchronously: {e}")

    async def _handle_shutdown(self) -> None:
        """Handle cleanup after run."""
        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel()
            logger.info("🚫 Planner task cancelled.")

        if self.agent.settings and getattr(self.agent.settings, 'generate_gif', False):
            output_path: str = 'agent_history.gif'
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(task=self.agent.task, history=self.agent.state.history, output_path=output_path)

        # Execute Done Callback
        await self.agent._execute_done_callback()