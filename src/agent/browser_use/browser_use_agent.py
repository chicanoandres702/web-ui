from __future__ import annotations
import abc
import asyncio
from asyncio.log import logger
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast
import json
import json_repair
import logging
import os
import re
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[3]))
from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc # AIMessage is from langchain_core.messages
from browser_use.agent.views import (
    ActionResult,
    AgentHistoryList,
    BrowserStateHistory,
    AgentOutput,
    AgentStepInfo,
    BrowserStateHistory)
# from browser_use.agent.message_manager.utils import is_model_without_tool_support
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableLambda # Import RunnableLambda
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD

from src.agent.browser_use import agent_utils  # Import the new module
from src.utils.utils import retry_async
from src.agent.browser_use.components.site_knowledge_injector import SiteKnowledgeInjector
from src.utils.io_manager import IOManager

class InstructionHandlerInterface(abc.ABC):
    @abc.abstractmethod

    async def handle_instructions(self, task: Any, decision: Dict[str, str]) -> str:
        pass


def _convert_settings_to_dict(settings: Any) -> Dict[str, Any]:
    """
    Converts settings to a dictionary, handling RunnableSequence objects.
    """
    if isinstance(settings, dict):
        return settings.copy()
    elif hasattr(settings, "dict") and callable(settings.dict):
        return cast(Dict[str, Any], settings.dict())
    else:
        logger.warning(f"Settings object is not a dictionary.  Returning empty dictionary.")
        return {}
    
load_dotenv()

class AgentLLMManager:
    """Manages LLM-related operations for the agent, including model switching and patching."""
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
        """Wraps the LLM with a RunnableLambda that applies Qwen/Ollama JSON fix logic."""
        llm_with_aimessage_output = llm | RunnableLambda(self._convert_agent_output_to_aimessage)
        return llm_with_aimessage_output | RunnableLambda(self._apply_qwen_fix_logic)

    def _convert_agent_output_to_aimessage(self, agent_output: AgentOutput) -> BaseMessage:
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
        """Applies the Qwen/Ollama JSON fix logic to an LLM response (BaseMessage)."""
        if not hasattr(response, 'content') or not isinstance(response.content, str):
            return response

        return response

    def _set_tool_calling_method(self) -> str | None:
        """Sets the tool calling method based on model and configuration."""
        settings = getattr(self.agent, 'settings', None)
        if not settings:
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
                # Handle control commands (pause/resume/stop) if needed
                pass

def _convert_settings_to_dict(settings: Any) -> Dict[str, Any]:
    """
    Converts settings to a dictionary, handling RunnableSequence objects.
    """
    if isinstance(settings, dict):
        return settings.copy()
    elif hasattr(settings, "dict") and callable(settings.dict):
        return cast(Dict[str, Any], settings.dict())
    else:
        #If settings is a RunnableSequence or other type, return an empty dictionary or handle appropriately
        logger.warning(f"Settings object is not a dictionary.  Returning empty dictionary.")

def _safe_get_attribute(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default)

ToolCallingMethod = str

def _prepare_agent_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares and validates the agent settings before passing them to AgentSettings.
    """
    if isinstance(settings, dict):
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
        settings,
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

        self.last_domain = None
        self.state = state
        self.browser_context = browser_context # type: ignore
        self.send_agent_message_callback = send_agent_message_callback
        self.confirmation_response_queue = confirmation_response_queue
        self.successful_steps_since_switch = 0
        self.switched_to_retry_model = False
        self.planner_task: Optional[asyncio.Task] = None


        agent_kwargs = kwargs.copy()
        self.use_custom_memory = agent_kwargs.pop('use_memory', True) #type: ignore
        # Extract agent-specific args that should not be passed to the base Agent classs
        self.save_history_path = agent_kwargs.pop('save_history_path', None)
        self.planner_llm = agent_kwargs.pop('planner_llm', None)
        self.planner_interval = agent_kwargs.pop('planner_interval', 5.0)
        self.cookie_path = agent_kwargs.pop('cookie_path', "./cookies")
        self.initial_actions = agent_kwargs.pop('initial_actions', None)

        self.agent_control_queue = agent_kwargs.pop('agent_control_queue', None)
        if confirmer_llm:
            agent_kwargs['validate_output'] = True

        self.validation_callback = validation_callback
        self.step_callback: Optional[Callable] = kwargs.get("step_callback")

        self.done_callback: Optional[Callable] = kwargs.get("done_callback")
        self.source = agent_kwargs.pop('source', None)
        self.inhibit_close = agent_kwargs.pop('inhibit_close', False)
        self.enable_smart_retry = enable_smart_retry
        self.enable_cost_saver = enable_cost_saver
        self.enable_user_interaction_dialog = agent_kwargs.pop('enable_user_interaction_dialog', True)
        self.auto_save_on_stuck = auto_save_on_stuck
        self.instruction_handler = instruction_handler

        #Safely get the llm
        original_llm = agent_kwargs.get('llm')
        if not original_llm:
            raise ValueError("LLM must be provided in agent_kwargs or model_priority_list")
                        
        # This instructs the LLM to produce output conforming to AgentOutput -type: ignore
        if hasattr(original_llm, "with_structured_output"):
            structured_llm = original_llm.with_structured_output(AgentOutput)
            logger.info("Applied AgentOutput schema to LLM using with_structured_output.") #type: ignore
        else:
            logger.warning("LLM does not support with_structured_output. Proceeding without structured output enforcement.")
            structured_llm = original_llm
        
        # Patch LLM if it's a model that needs JSON fixes (e.g., Qwen/Ollama)
        self.llm_manager = AgentLLMManager(self, model_priority_list)
        chat_model_library = original_llm.__class__.__name__ if original_llm else "" #type: ignore
        model_name = getattr(original_llm, "model_name", "").lower() or getattr(original_llm, "model", "").lower()
        if any(m in model_name for m in ['qwen', 'ollama']):
            structured_llm = self.llm_manager._patch_llm_with_qwen_fix(structured_llm)
            logger.info(f"Applied Qwen/Ollama JSON fix patch to LLM: {model_name}")
        else:
            logger.info(f"Using standard LLM output for: {model_name}")
        agent_kwargs['llm'] = structured_llm
        self.max_consecutive_failures = kwargs.get("max_consecutive_failures", 500)

        # super().__init__(**agent_kwargs)

        from src.agent.browser_use.agent_components import AgentHeuristics as ComponentHeuristics
        self.heuristics = ComponentHeuristics(self)
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness = confirmer_strictness
        self.controller = controller

        if model_priority_list and model_priority_list[0].get('llm'):
            self.llm_manager.set_llm(model_priority_list[0]['llm'])
        else:
            self.llm_manager.set_llm(agent_kwargs['llm'])

        from src.agent.browser_use.agent_components import BrowserActionHandler
        # self._initialize_action_handler()
        self.action_handler = BrowserActionHandler(self.browser_context, controller=self.controller, heuristics=self.heuristics)

    def _initialize_action_handler(self):
        """d
        Initializes the BrowserActionHandler.
        """
        if not self.browser_context:
            return # type: ignore
        from src.agent.browser_use.control_queue_handler import ControlQueueHandler #type: ignore

        # self.control_queue_handler = ControlQueueHandler(self) # type: ignore
        self.cleanup_handler = AgentCleanupHandler(self) #type: ignore
        self.action_handler = BrowserActionHandler(self.browser_context, controller=self.controller, heuristics=self.heuristics)
        self.post_step_handler = PostStepHandler(self)

    def _set_tool_calling_method(self) -> str:
        return self.llm_manager._set_tool_calling_method() or "auto"
    
    @property
    def _tool_calling_supported(self) -> bool:
        """Whether tool calling is supported by the agent."""
        return self._set_tool_calling_method() == "function_calling"
    def _initialize_components(self, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None) -> None:

        """
        Initializes agent components and applies system prompt override.
        """
        from src.agent.browser_use.agent_components import ComponentInitializer
        from src.agent.browser_use.agent_components import AgentCleanupHandler
        # ComponentInitializer(self, agent_kwargs, system_prompt)

        # Initialize SiteKnowledgeInjector
        self.site_knowledge_injector = SiteKnowledgeInjector(self)

        # Initialize AgentLoopHandler
        self.planner_task = None

    async def run(self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None) -> AgentHistoryList:


        loop = asyncio.get_event_loop()
        if self.send_agent_message_callback:

            await self.send_agent_message_callback({"type": "agent_status", "status": "Participating"})

        try:
            self._log_agent_run()

            # Start Planner Loop if enabled
            if self.planner_llm:
                  self.planner_task = asyncio.create_task(self._planner_loop())

            # Execute initial actions if provided
            if hasattr(self, 'initial_actions') and self.initial_actions:
                # Use custom execution to ensure actions are run reliably via controller
                results = await self._execute_initial_actions()
                if results:
                    # Inject into message history so the agent knows what happened
                    summary = "\n".join(results)
                    msg = f"SYSTEM NOTE: The following actions were already executed for this step:\n{summary}\n\nCheck the current state. If the goal is achieved, output {{'finish': ...}}. Do NOT repeat the above actions."
                    if hasattr(self, "message_manager") and self.message_manager:
                        add_msg_func = getattr(self.message_manager, "add_message", None)
                        if callable(add_msg_func):

                            add_msg_func(HumanMessage(content=msg))
                    # Also set last_result for good measure if the parent class uses it
                    self.state.last_result = [ActionResult(extracted_content=summary, include_in_memory=True)]

            for step in range(max_steps):
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
                    self.state.consecutive_failures += 1
                    if self.state.consecutive_failures >= self.max_consecutive_failures:
                        break
                        
                    # If a custom task was provided, the agent.task would have been updated.
                    # We should continue the loop to process the new task.
            else:
                await self.cleanup_handler._handle_max_steps_reached()

            return self.state.history

        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history
        except asyncio.CancelledError:

            logger.info('Agent execution cancelled')
            return self.state.history
        finally:
            await self.cleanup_handler.cleanup()

    async def _execute_agent_step(self, step_info: AgentStepInfo):
        """Executes a single step of the agent."""
        try:
            await self.step(step_info)
        except Exception as e:
            if "Timeout" in str(e) and "exceeded" in str(e):
                logger.warning(f"⚠️ Step {step_info.step_number + 1} timed out: {e}. Attempting to recover by reloading page.")
                self.state.consecutive_failures += 1
                await self._recover_from_timeout()
            else:
                logger.error(f"An unexpected error occurred during step {step_info.step_number + 1}: {e}")
                self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step_info.step_number + 1}: {e}")
                raise e

    async def _recover_from_timeout(self):
        """Attempts to recover from a timeout error by reloading the page."""
        try:
            logger.info("Attempting to reload the page to recover...")
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
                from src.utils import prompts
                PLANNER_PROMPT = getattr(prompts, 'PLANNER_PROMPT', DEFAULT_PLANNER_PROMPT)

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
                            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
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
                manager = getattr(self.controller, "webui_manager", None)
                if manager and hasattr(manager, "add_plan_step"):
                    new_step_desc = None
                    if "CAPTCHA" in reason_upper:
                        new_step_desc = "Solve CAPTCHA detected on page"
                    elif "LOGIN" in reason_upper or "SIGN IN" in reason_upper:
                        new_step_desc = "Handle login requirement"

                    step_index = getattr(self.state, 'current_step_index', 0)
                    if new_step_desc:
                        insert_idx = step_index + 1
                        manager.add_plan_step(new_step_desc, index=insert_idx)
                        manager.update_plan_step_status(step_index, "failed", result=f"Blocked: {reason}")
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

        if self.confirmer_strictness and self.confirmer_strictness <= 3:
            return CONFIRMER_PROMPT_FAST.format(task=self.task)
        return CONFIRMER_PROMPT_STANDARD.format(task=self.task, strictness=self.confirmer_strictness or 5)

    def _build_confirmer_messages(self, system_prompt: str) -> List[BaseMessage]:

        history = self.state.history
        last_item = history.history[-1] if history.history else None
        content: Sequence[Any] = [{"type": "text", "text": f"User Task: {self.task}"}]

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
            # Use getattr to avoid static analysis errors for unknown attributes on MessageManager
            add_msg_func = getattr(self.message_manager, "add_message", None)
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
        """
        Handles pre-step checks: pause, stop, model switching, failures.

        Also manages cookie persistence across sessions

        Returns True if the agent should stop/break the loop.
        """
        # Await self._process_control_queue()

        # Suggest alternative strategies on failure
        # Manage Model Switching (Cost Saver / Smart Retry)
        self.heuristics.manage_model_switching()

                # Suggest alternative strategies on failure
        self._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self.cleanup_handler._manage_tabs()

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

            if self.browser_context:   

                try:
                    with open(self.cookie_path, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                    await self.browser_context.add_cookies(cookies)
                    logger.info(f"Loaded cookies from {self.cookie_path}")
                except Exception as e:
                    logger.error(f"Failed to load cookies: {e}")



        return False
class UserInteractionHandler:
    """Handles user interaction requests and decision logic."""
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = self.agent.state.history.history[-1] if self.agent.state.history.history else None
        intel = "No specific new information from this step."
        
        if last_step_info and last_step_info.model_output:
            thought = getattr(last_step_info.model_output, "thought", None)
            if thought:
                intel = thought
        
        next_action_desc = "Agent is considering its next move."
        if last_step_info and last_step_info.model_output and last_step_info.model_output.action:
            action_strings = []
            actions = last_step_info.model_output.action if isinstance(last_step_info.model_output.action, list) else [last_step_info.model_output.action]
            for action_model in actions:
                if hasattr(action_model, "model_dump"):
                    action_dict = action_model.model_dump()
                    action_strings.append(str(action_dict))
                else:
                    action_strings.append(str(action_model))
            next_action_desc = ", ".join(action_strings)
                
        user_decision = await self.agent.llm_manager.request_user_decision(intel, next_action_desc)

        if user_decision is False:
            self.agent.state.stopped = True
            return True
        elif isinstance(user_decision, str):
            self.agent.task = user_decision
            self.agent.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")
            return False
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
            await self.agent.action_handler.save_cookies(self.agent.cookie_path)



        # --- User Interaction Dialog ---
        if await self._handle_user_interaction():
            return True # Stop requested

        if self.agent.state.history.is_done():
            if not await self.agent._validate_output():

                return False # Validation failed, continue loop

            await self.agent.log_completion()
            # Finalize the run
            await self.agent.cleanup_handler.cleanup()
            return True # Task done, break loop

        return False
    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        """
        handler = UserInteractionHandler(self.agent)
        return await handler.handle_user_interaction()

    async def _request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Internal wrapper for requesting user decision.
        """
        return await self.agent.llm_manager.request_user_decision(intel, next_task)

class BrowserActionHandler:
    """Handles browser-specific actions like cookie management."""
    def __init__(self, browser_context: Any, controller: Any, heuristics: Any):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics

    async def save_cookies(self, cookie_path: str) -> None:
        """Saves cookies from the browser context to a file."""
        if self.browser_context and cookie_path:
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
                if not await self.agent._validate_output():
                    return False  # Validation failed, continue loop
            
            await self.agent.log_completion()
            return True
        return False

class AgentHeuristics:
    """Encapsulates heuristic logic for agent behavior, loop detection, and strategy."""
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent
        self.is_waiting_for_mfa = False

    def inject_message(self, content: str):
        self.agent.inject_notification(content) # type: ignore

    async def check_completion_heuristics(self):
        """Heuristic checks to see if the task is actually finished even if LLM didn't say so."""
        pass

    def detect_loop(self):
        """Detects if the agent is stuck in an action loop."""
        history = self.agent.state.history.history
        if len(history) < 3:
            return

        # Simple heuristic: check if the last 3 actions are identical
        last_actions = []
        for item in history[-3:]:
            if item.model_output and item.model_output.action:
                last_actions.append(str(item.model_output.action))

        if len(last_actions) == 3 and len(set(last_actions)) == 1:
            logger.warning("🔄 Loop detected: Agent is repeating the same action.")
            self.inject_message("SYSTEM: Loop detected. You have performed the same action 3 times. Please try a different approach or inspect the page more closely.")
            return True
    
        return False
    
    async def detect_progress(self):
        """Detects if progress is being made."""
        # Check if the URL has changed or if the state has been updated
        if len(self.agent.state.history.history) >= 2:
            last_state = self.agent.state.history.history[-1].state
            prev_state = self.agent.state.history.history[-2].state
            
            if last_state.url != prev_state.url:
                logger.debug("Progress detected: URL changed.")
                return True
            
            if last_state.title != prev_state.title:
                logger.debug("Progress detected: Page title changed.")
                return True
                

    async def check_blocking_elements(self):
        """Checks for overlays, cookie banners, or popups."""
        if not self.agent.browser_context:
            return

        try:
            page = await self.agent.browser_context.get_current_page()
            # Check for common overlay/popup selectors
            blocking_selectors = [
                '.modal', '.popup', '#cookie-banner', '.overlay', 
                '[id*="cookie"]', '[class*="modal"]', '.sp-fancybox-wrap'
            ]
            
            for selector in blocking_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if await element.is_visible():
                        logger.info(f"Potential blocking element detected: {selector}")
                        # We don't auto-close here to avoid breaking site flow, 
                        # but we notify the agent to handle it.
                        self.inject_message(f"SYSTEM: A potential popup or overlay ({selector}) is visible. If it blocks your actions, try to close it first.")
                        return
        except Exception as e:
            logger.debug(f"Error checking blocking elements: {e}")
            
        pass

    async def check_navigation_recovery(self):
        """Checks if navigation is stuck and attempts recovery."""
        if len(self.agent.state.history.history) > 5:
            last_actions = [
                str(h.model_output.action) 
                for h in self.agent.state.history.history[-5:] 
                if h.model_output and h.model_output.action
            ]
            if len(set(last_actions)) == 1:
                logger.warning("🔄 Navigation stuck: Attempting recovery via page reload.")
                if self.agent.browser_context:
                    try:
                        page = await self.agent.browser_context.get_current_page()
                        await page.reload(wait_until="domcontentloaded", timeout=10000)
                        self.inject_message("SYSTEM: Navigation appeared stuck. I have reloaded the page to attempt recovery.")
                    except Exception as e:
                        logger.error(f"Navigation recovery failed: {e}")


    async def check_login_status(self):
        """Checks if the agent has been logged out or needs to log in."""
        if not self.agent.browser_context:
            return

        try:
            page = await self.agent.browser_context.get_current_page()
            url = page.url.lower()
            
            # Common indicators of being logged out or needing login
            login_indicators = ['login', 'signin', 'sign-in', 'accounts.google.com', 'auth']
            
            if any(indicator in url for indicator in login_indicators):
                logger.info(f"Login page detected: {url}")
                self.inject_message("SYSTEM: It appears you are on a login or authentication page. If you have credentials, please log in. If you are stuck, notify the user.")
                
        except Exception as e:
            logger.debug(f"Error checking login status: {e}")
            
        pass

    def manage_model_switching(self):
        """Handles logic for switching between primary and retry models."""
        if not self.agent.enable_smart_retry and not self.agent.enable_cost_saver:
            return

        consecutive_failures = self.agent.state.consecutive_failures
        model_priority_list = self.agent.llm_manager.model_priority_list

        if not model_priority_list or len(model_priority_list) < 2:
            return

        # Smart Retry: Switch to a more capable model if failing
        if self.agent.enable_smart_retry and consecutive_failures >= 2 and not self.agent.switched_to_retry_model:
            retry_model = model_priority_list[1].get('llm')
            if retry_model:
                logger.info(f"🔄 Smart Retry: Switching to backup model due to {consecutive_failures} failures.")
                self.agent.llm_manager.set_llm(retry_model)
                self.agent.switched_to_retry_model = True
                self.inject_message("SYSTEM: I've switched to a more capable model to help resolve the current issue.")

        # Cost Saver: Switch back to primary model after successful steps
        elif self.agent.enable_cost_saver and self.agent.switched_to_retry_model:
            if consecutive_failures == 0:
                self.agent.successful_steps_since_switch += 1
            else:
                self.agent.successful_steps_since_switch = 0

            if self.agent.successful_steps_since_switch >= 3:
                primary_model = model_priority_list[0].get('llm')
                if primary_model:
                    logger.info("💰 Cost Saver: Switching back to primary model after successful steps.")
                    self.agent.llm_manager.set_llm(primary_model)
                    self.agent.switched_to_retry_model = False
                    self.agent.successful_steps_since_switch = 0
                    

    def suggest_alternative_strategy(self):
        """Injects strategy hints into the message history on failure."""
        if self.agent.state.consecutive_failures >= 2:
            logger.info("💡 Injecting alternative strategy hint due to consecutive failures.")
            
            hints = [
                "If you are stuck in a loop, try scrolling the page or looking for hidden elements.",
                "Check if there are any error messages or popups blocking your view.",
                "Try to use a different search query or navigation path.",
                "Verify if the element you are looking for is inside an iframe or shadow DOM.",
                "If the page hasn't loaded fully, try waiting or reloading."
            ]
            import random
            hint = random.choice(hints)
            self.inject_message(f"STRATEGY HINT: {hint}")

    async def check_and_add_subtasks(self):
        """Analyzes page complexity and adds subtasks to the plan if needed."""
        if not self.agent.browser_context:
            return

        try:
            history = self.agent.state.history.history
            if len(history) < 1:
                return

            # Only analyze if we haven't added a subtask recently to avoid plan bloating
            manager = getattr(self.agent.controller, "webui_manager", None)
            if not manager or not hasattr(manager, "bu_plan"):
                return

            # Heuristic: If the page has many interactive elements or long text, 
            # and the current plan step is generic, suggest breaking it down.
            last_state = history[-1].state
            element_count = len(last_state.interacted_element) if hasattr(last_state, 'interacted_element') else 0
            
            if element_count > 50:
                logger.info(f"Complex page detected ({element_count} elements).")
                # This is a hook for the Planner Loop to take action
                # The actual LLM-based breakdown happens in _planner_loop
        except Exception as e:
            logger.debug(f"Error in check_and_add_subtasks: {e}")
            
        # This can be expanded to use the LLM to break down complex pages
        pass


class ComponentInitializer:
    """Initializes and configures agent components."""
    def __init__(self, agent: BrowserUseAgent, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None):
        self.agent = agent
        self._setup_system_prompt(system_prompt)

    def _setup_system_prompt(self, system_prompt: Optional[str]):
        if system_prompt:
            self.agent.message_manager.system_prompt = SystemMessage(content=system_prompt)


class AgentCleanupHandler:
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
            # MessageManager does not have a close() method in browser-use.
            # If there are specific cleanup tasks for message_manager, they should be handled here.
            # Otherwise, we rely on the parent Agent class or garbage collection.
            pass



        if self.agent.settings and getattr(self.agent.settings, 'generate_gif', False):
            output_path = 'agent_history.gif'
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(task=self.agent.task, history=self.agent.state.history, output_path=output_path)

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
        
        next_action_desc = "Agent is considering its next move."
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


    async def _handle_max_steps_reached(self):

        """Handles the case where the agent reaches maximum steps without completion."""
        error_message = 'Failed to complete task in maximum steps'
        self.agent.state.history.history.append(
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
            await IOManager.write_file(path, self.agent.state.history.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save history asynchronously: {e}")



        """Handle cleanup after run."""

        if self.agent.planner_task and not self.agent.planner_task.done():
            self.agent.planner_task.cancel() # type: ignore
            logger.info("🚫 Cleanup inhibited: Browser will remain open.") # type: ignore
        else:
            try:
                await self.agent.close()
            except Exception as e:
                # Log as debug to avoid noise during forced shutdowns/interrupts
                logger.debug(f"Error during agent cleanup (likely benign): {e}")

        if self.agent.settings and getattr(self.agent.settings, 'generate_gif', False):
            output_path: str = 'agent_history.gif'
            if isinstance(self.agent.settings.generate_gif, str):
                output_path = self.agent.settings.generate_gif
            create_history_gif(task=self.agent.task, history=self.agent.state.history, output_path=output_path)

        # Execute Done Callback
        await self.agent._execute_done_callback()