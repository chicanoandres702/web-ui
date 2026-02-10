from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
import json
from pathlib import Path
import json
import json_repair
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from dotenv import load_dotenv

import sys

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc # AIMessage is from langchain_core.messages
from browser_use.agent.views import (

    AgentOutput,
    ActionResult,
    AgentStepInfo,
    AgentHistoryList,
    AgentHistory,

    BrowserStateHistory
)
from src.agent.browser_use.control_queue_handler import ControlQueueHandler
from src.agent.browser_use.tab_management_handler import TabManagementHandler
from src.agent.browser_use.user_interaction_handler import UserInteractionHandler
from src.agent.browser_use.agent_cleanup_handler import AgentCleanupHandler
from src.agent.browser_use.agent_heuristics import AgentHeuristics
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD


class ComponentInitializer:
    """Initializes and configures agent components."""
    def __init__(self, agent: Any, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None):
        self.agent = agent
        if system_prompt:
            self.agent.message_manager.system_prompt = SystemMessage(content=system_prompt)

import abc
class InstructionHandlerInterface(abc.ABC): # type: ignore
    @abc.abstractmethod
    async def handle_instructions(self, task: Any, decision: Dict[str, str]) -> str:
        pass

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from src.agent.browser_use import agent_utils  # Import the new module

from src.agent.browser_use.components.site_knowledge_injector import SiteKnowledgeInjector

from src.agent.browser_use.components.cookie_manager import CookieManager
from src.utils.io_manager import IOManager
from src.utils.utils import parse_json_safe, retry_async

class AgentLLMManager:
    """Manages LLM-related operations for the agent, including model switching and patching."""
    def __init__(self, agent):
        self.agent = agent

    def set_llm(self, llm):
        """Sets the current LLM for the agent."""
                #Before setting, log the previous and new LLM
        old_llm_name = getattr(self.agent, 'llm', None)
        if old_llm_name:
            old_llm_name = old_llm_name.__class__.__name__

        new_llm_name = llm.__class__.__name__
        logger.info(f"Switching LLM from {old_llm_name} to {new_llm_name}")
        self.agent.llm = llm


    def _patch_llm_with_qwen_fix(self, llm: Runnable) -> Runnable:
        """Wraps the LLM with a RunnableLambda that applies Qwen/Ollama JSON fix logic."""
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
        except Exception:
            logger.debug("QwenFixWrapper: Initial json_repair failed. Attempting more aggressive repair.")
            # Fallback to more aggressive repair or default to empty JSON
            try:
                fixed_content = content + '}' * (content.count('{') - content.count('}'))
                data = json_repair.loads(fixed_content)
                response.content = json.dumps(data)
            except Exception as inner_e:
                logger.warning(f"QwenFixWrapper: Inner repair attempt failed: {inner_e}. Original: '{content}'")
                response.content = "{}"
            
        return response

    def _set_tool_calling_method(self) -> str | None:
        """Sets the tool calling method based on model and configuration."""
        tool_calling_method = self.agent.settings.tool_calling_method
        chat_model_library = self.agent.chat_model_library
        model_name = self.agent.model_name.lower()
    
        if tool_calling_method == 'auto':
            if 'Ollama' in chat_model_library:
                return None
            elif any(m in model_name for m in ['qwen', 'deepseek', 'gpt', 'claude']) and 'Ollama' not in chat_model_library:
                return 'function_calling'
            elif chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """Requests user decision via WebSocket."""
        """
        Requests confirmation/input from the user via WebSocket, allowing indefinite waiting.
        """
        send_callback = getattr(self.agent, 'send_agent_message_callback', None)
        if not send_callback or not self.agent.confirmation_response_queue:
            return True

        await send_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review. Awaiting external device approval.",
            "intel": intel,
            "next_task": next_task
        })

        response_data = await self.agent.confirmation_response_queue.get()
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
    """Handles browser-specific actions.

    This class decouples browser interactions from the agent's core logic,
    providing a clear abstraction for performing actions in the browser.
    """
    def __init__(self, browser_context, controller, heuristics):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics

    async def save_cookies(self, cookie_path: str) -> None:
        """Saves cookies from the browser context to a file."""
        try:
            cookies = await self.browser_context.get_cookies()
            # Serialize the cookies to a JSON string before writing to the file
            cookies_json = json.dumps(cookies, indent=2)  # Pretty print the JSON

            # Write the JSON string to the file
            with open(cookie_path, 'w', encoding='utf-8') as f:
                f.write(cookies_json)
            logger.info(f"Saved cookies to {cookie_path}")
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")


    def detect_loop(self) -> bool:
        """Delegates loop detection to heuristics."""
        return self.heuristics.detect_loop()

    async def detect_progress(self) -> bool:
        """Delegates progress detection to heuristics."""
        return await self.heuristics.detect_progress()

    async def check_blocking_elements(self) -> None:
        """Delegates blocking element checks to heuristics."""
        await self.heuristics.check_blocking_elements()

    async def check_navigation_recovery(self) -> None:
        """Delegates navigation recovery to heuristics."""
        await self.heuristics.check_navigation_recovery()

    async def check_login_status(self) -> None:
        """Delegates login status checks to heuristics."""
        await self.heuristics.check_login_status()

    def manage_model_switching(self) -> None:
        """Delegates model switching logic to heuristics."""
        self.heuristics.manage_model_switching()

    def suggest_alternative_strategy(self) -> None:
        """Delegates strategy hints to heuristics."""
        self.heuristics.suggest_alternative_strategy()

    async def check_and_add_subtasks(self) -> None:
        """Delegates subtask analysis to heuristics."""
        await self.heuristics.check_and_add_subtasks()

ToolCallingMethod = str

class BrowserUseAgentBase:
    def __init__(
        self,
        state,
        settings,
        message_manager,
        confirmer_llm=None,
        confirmer_strictness=7,
        model_priority_list: Optional[List[Dict]] = None,
        enable_cost_saver=False,
        enable_smart_retry=False,
        auto_save_on_stuck=True,
        system_prompt: Optional[str] = None,
        validation_callback=None,
        task: str = " ",
        browser_context=None,
        send_agent_message_callback: Optional[Callable] = None,
        confirmation_event: Optional[asyncio.Event] = None,
        confirmation_response_queue: Optional[asyncio.Queue] = None,
        instruction_handler: Optional[InstructionHandlerInterface] = None,
        **kwargs
    ):
        self.send_agent_message_callback = send_agent_message_callback
        self.last_domain = None
        self.task = task
        self.state = state
        self.settings = settings
        self.confirmation_response_queue = confirmation_response_queue
        self.successful_steps_since_switch = 0
        self.switched_to_retry_model = False
        self.planner_task: Optional[asyncio.Task] = None

        # Extract use_memory to avoid passing it to parent Agent which doesn't support it
        # We intercept 'use_memory' to use our own SimpleMemoryManager instead of mem0

        
        # Extract agent-specific args that should not be passed to the base Agent class
        self.save_history_path = kwargs.pop('save_history_path', None)
        self.planner_llm = kwargs.pop('planner_llm', None)
        self.planner_interval = kwargs.pop('planner_interval', 5.0)
        self.cookie_path = kwargs.pop('cookie_path', "./tmp/cookies.json")
        self.enable_user_interaction_dialog = kwargs.pop('enable_user_interaction_dialog', True)
        self.initial_actions = kwargs.pop('initial_actions', None)
        self.agent_control_queue = kwargs.pop('agent_control_queue', None)
        self.done_callback = kwargs.pop('done_callback', None)

        if confirmer_llm:
            kwargs['validate_output'] = True

        self.validation_callback = validation_callback
        self.step_callback: Optional[Callable] = kwargs.get("step_callback")

        # Extract source from kwargs if provided, default to None
        self.source = kwargs.pop('source', None)
        self.inhibit_close = False
        self.enable_smart_retry = enable_smart_retry
        self.enable_cost_saver = enable_cost_saver
        self.auto_save_on_stuck = auto_save_on_stuck
        self.instruction_handler = instruction_handler
        if 'llm' in kwargs:
            agent_kwargs = kwargs.copy() # type: ignore

            original_llm = agent_kwargs['llm']
            
            # Apply structured output using AgentOutput schema
            # This instructs the LLM to produce output conforming to AgentOutput
            if hasattr(original_llm, "with_structured_output"):
                structured_llm = original_llm.with_structured_output(AgentOutput)
                logger.info("Applied AgentOutput schema to LLM using with_structured_output.")
            else:
                logger.warning("LLM does not support with_structured_output. Proceeding without structured output enforcement.")
                structured_llm = original_llm
        self.max_consecutive_failures = kwargs.get("max_consecutive_failures", 500)
        super().__init__(task=task, **agent_kwargs) # type: ignore

        self.llm_manager = AgentLLMManager(self)
        self.heuristics = AgentHeuristics(self)
        self.confirmer_llm = confirmer_llm

        self.confirmer_strictness = confirmer_strictness

        if model_priority_list and model_priority_list[0].get('llm'):
            self.llm_manager.set_llm(model_priority_list[0]['llm'])
        else:
            self.llm_manager.set_llm(agent_kwargs['llm'])

        from src.agent.browser_use.browser_use_agent import BrowserActionHandler
        self.action_handler = BrowserActionHandler(self.browser_context, self.controller, self.heuristics) # type: ignore
        self._initialize_action_handler()
        self.control_queue_handler = ControlQueueHandler(self)

    def _initialize_action_handler(self):
        """d
        Initializes the BrowserActionHandler.
        """

        return self.llm_manager._set_tool_calling_method() or "auto"
        
    def _initialize_components(self, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None) -> None:
        """
        Initializes agent components and applies system prompt override.
        """
        ComponentInitializer(self, agent_kwargs, system_prompt) # type: ignore
        self.site_knowledge_injector = SiteKnowledgeInjector(self) # type: ignore

        self.user_interaction_handler = UserInteractionHandler(self) # type: ignore
        self.pre_step_handler = PreStepHandler(self) # type: ignore
        self.cleanup_handler = AgentCleanupHandler(self) # type: ignore
        self.tab_handler = TabManagementHandler(self) # type: ignore
        self.cookie_handler = CookieHandler(self) # type: ignore


        from src.agent.browser_use.browser_use_agent import PostStepHandler
        self.post_step_handler = PostStepHandler(self) # type: ignore

class CookieHandler:
    """Handles cookie loading and saving for the agent."""
    def __init__(self, agent: Any):
        """Initializes the CookieHandler with the agent instance."""
        self.agent = agent

    async def load_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Loads cookies from a file into the browser context."""
        pass

class BrowserUseAgent(Agent, BrowserUseAgentBase): # Ensure inheritance if intended, or keep as is if wrapper
    # ... (existing __init__ and other methods)

    async def run(self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None, on_step_end: AgentHookFunc | None = None) -> AgentHistoryList:
        loop = asyncio.get_event_loop()
        try:
            if hasattr(self, '_log_agent_run'):
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
                    message_manager = getattr(self, "message_manager", None)
                    if message_manager:
                        # MessageManager in browser-use uses add_message for HumanMessages
                        add_msg_func = getattr(message_manager, "add_message", None)
                        if callable(add_msg_func):
                            add_msg_func(HumanMessage(content=msg))
                    # Also set last_result for good measure if the parent class uses it
                    self.state.last_result = [ActionResult(extracted_content=summary, include_in_memory=True)]

            for step in range(max_steps):
                try:
                    if await self.pre_step_handler.handle_pre_step():
                        break

                    if on_step_start is not None:
                        await on_step_start(self) # type: ignore

                    step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                    await self._execute_agent_step(step_info)

                    if on_step_end is not None:
                        await on_step_end(self) # type: ignore

                    if await self.post_step_handler.handle_post_step(step, max_steps):
                        break

                    # If a custom task was provided, the agent.task would have been updated.
                    # We should continue the loop to process the new task.
                except Exception as e:
                    logger.error(f"Error in step {step}: {e}")
                    self.state.consecutive_failures += 1
                    if self.state.consecutive_failures >= self.max_consecutive_failures:
                        break
            else:
                completion_handler = CompletionCheckHandler(self)
                completion_handler.handle_max_steps_reached()
            return self.state.history
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            raise e
        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history
        except asyncio.CancelledError:
            logger.info('Agent execution cancelled')
            return self.state.history
        finally:
            if hasattr(self, 'cleanup_handler'):
                await self.cleanup_handler.handle_cleanup()

    async def save_history_async(self, path: str):
        """Saves the agent history to a file asynchronously."""
        try:
            await IOManager.write_file(path, self.state.history.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save history asynchronously: {e}")

    async def _execute_agent_step(self, step_info: AgentStepInfo):
        """Executes a single step of the agent."""
        try:
            await super().step(step_info)
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
                current_page = await self.browser_context.get_current_page()
                await current_page.reload(wait_until="domcontentloaded", timeout=15000)
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

                if hasattr(last_step, 'state') and hasattr(last_step.state, 'element_tree'):
                    page_summary = (getattr(last_step.state, 'element_tree', "") or "")[:4000]
                
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
            # Extract reason
            if is_confirmed:
                reason = clean_response[3:].strip(" .:-")
            elif clean_response.upper().startswith("NO"):
                reason = clean_response[2:].strip(" .:-")


            if self.validation_callback:
                try:
                    if inspect.iscoroutinefunction(self.validation_callback):
                                
                        await self.validation_callback(think_content, reason, is_confirmed)
                    else:
                        self.validation_callback(think_content, reason, is_confirmed)
                except Exception as e:
                    logger.error(f"Validation callback failed: {e}")
            reason_upper = reason.upper()
            current_step_index = getattr(self, 'current_step_index', None)
            if current_step_index is not None and self.controller:
                    manager = getattr(self.controller, "webui_manager", None)
                    if manager and hasattr(manager, "update_plan_step_status"):

                        # 1. Check for Critical Blockers to update Plan (Dynamic Planning)
                        if not is_confirmed:
                            if "CAPTCHA" in reason_upper:
                                new_step_desc = "Solve CAPTCHA detected on page"
                            elif "LOGIN" in reason_upper or "SIGN IN" in reason_upper:
                                new_step_desc = "Navigate to login page and authenticate"
                            else:
                                new_step_desc = None
                                                            
                                
                            

                            if new_step_desc:
                                insert_idx = current_step_index + 1
                                manager.add_plan_step(new_step_desc, index=insert_idx)
                                manager.update_plan_step_status(current_step_index, "failed", result=f"Blocked: {reason}")

                                logger.info(f"🚨 Confirmer detected blocker. Added recovery step: '{new_step_desc}' at index {insert_idx}")
                                self.state.stopped = True # Stop current agent to switch to new step
                                return False

                        # 2. Normal Status Update
                        status = "completed" if is_confirmed else "in_progress"
                        display_result = reason
                        if not is_confirmed:
                            display_result = f"⚠️ Rejection: {reason}"
                        
                        manager.update_plan_step_status(current_step_index,status, result=display_result)
                        logger.info(f"✅ Confirmer updated plan step {current_step_index + 1}. Status: {status}. Reason: {reason}")

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

    def _build_confirmer_messages(self, system_prompt: str) -> List[BaseMessage]:
        """Constructs the message history for the confirmer LLM."""
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
                else:
                    self.done_callback(self.state.history)
            except Exception as e:
                logger.error(f"Error in done_callback: {e}")

class PreStepHandler:
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def _process_control_queue(self) -> None:
        """Processes commands from the control queue."""
        await self.agent.control_queue_handler.process_control_queue()

    async def handle_pre_step(self) -> bool:
        """
        Handles pre-step checks: pause, stop, model switching, failures.

        Also manages cookie persistence across sessions

        Returns True if the agent should stop/break the loop.
        """
        # Process control queue
        await self._process_control_queue()


        # Manage Model Switching (Cost Saver / Smart Retry)
        self.agent.heuristics.manage_model_switching()

        # Suggest alternative strategies on failure
        if hasattr(self.agent, '_suggest_alternative_strategy'):
            self.agent._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self._manage_tabs()

        # Custom Memory Injection
        # if self.use_custom_memory and self.memory_manager:
        await self.agent.site_knowledge_injector.inject_site_knowledge()


        # Check if we should stop due to too many failures
        # Note: self.heuristics is a direct attribute of BrowserUseAgent

        # if await self.heuristics.check_max_failures():
        #    return True


        # Check if MFA is required and wait for approval

        if self.agent.heuristics.is_waiting_for_mfa:
            logger.info("MFA required, waiting for approval from another device, after setting pre step...")
            if getattr(self.agent, 'send_agent_message_callback', None):
                await self.agent.send_agent_message_callback({ # type: ignore
                    "type": "agent_status", 
                    "status": "Waiting for MFA 🔐"
                })
                
            while self.agent.heuristics.is_waiting_for_mfa and not self.agent.state.stopped:
                await asyncio.sleep(15)  # Check every second
            logger.info("MFA approved or agent stopped.")
            self.agent.heuristics.is_waiting_for_mfa = False  # Reset flag

            # Verify there is still a valid page and browser context
            if not self.agent.browser_context or not hasattr(self.agent.browser_context, "get_current_page"):
                logger.warning("Browser context is no longer valid after MFA, stopping agent.")
                self.agent.state.stopped = True
                return True

            return True

        if self.agent.state.stopped:

            logger.info('Agent stopped')
            return True

        while self.agent.state.paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self.agent.state.stopped:  # Allow stopping while paused

                return True
        # Load cookies at the beginning of each session
        if self.agent.cookie_path and os.path.exists(self.agent.cookie_path):
            if self.agent.browser_context:   
                try:
                    with open(self.agent.cookie_path, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                        
                        # browser-use BrowserContext objects store the Playwright context in .context
                        playwright_context = getattr(self.agent.browser_context, 'context', None)
                        if playwright_context and hasattr(playwright_context, 'add_cookies'):
                            await playwright_context.add_cookies(cookies)
                        else:
                            logger.warning("Could not access Playwright context to add cookies.")
                            
                        logger.info(f"Loaded cookies from {self.agent.cookie_path}")
                except Exception as e:
                    logger.error(f"Failed to load cookies: {e}")
        return False

    async def _manage_tabs(self):
        """Closes extraneous tabs to keep the agent focused."""
        nav_controller = getattr(self.agent.controller, 'nav_controller', None)
        if nav_controller and self.agent.browser_context:
            await nav_controller.manage_tabs(self.agent.browser_context)

    async def _inject_site_knowledge(self):
        """Checks if we have entered a new domain and injects relevant knowledge."""
        await self.agent.site_knowledge_injector.inject_site_knowledge()


    async def _handle_post_step(self, step: int, max_steps: int) -> bool:
        """
        Handles post-step logic: callbacks, persistence, completion check.
        Returns True if the agent should stop/break the loop (task done).
        """
        # Execute WebUI Callbacks
        await self.agent._execute_step_callbacks(step)

        # Persistence: Auto-save history after each step
        if self.agent.save_history_path:
            try:
                await self.agent.save_history_async(self.agent.save_history_path)

            except Exception as e:
                logger.warning(f"Failed to auto-save history: {e}")

        await self.agent.heuristics.check_completion_heuristics()

        # Check for loops
        self.agent.heuristics.detect_loop()
  
        # Check for progress indicators (if implemented in heuristics)
        if hasattr(self.agent.heuristics, 'detect_progress'):
            await self.agent.heuristics.detect_progress()

        if hasattr(self.agent.action_handler, 'check_blocking_elements'):
            await self.agent.action_handler.check_blocking_elements()
  
        if hasattr(self.agent.action_handler, 'check_navigation_recovery'):
            await self.agent.action_handler.check_navigation_recovery()
        

        # Check for complex page structure and add subtasks
        await self.agent.heuristics.check_and_add_subtasks()

        completion_check_handler = CompletionCheckHandler(self.agent)
        await completion_check_handler.check_completion()

        if hasattr(self.agent.action_handler, 'check_login_status'):
            await self.agent.action_handler.check_login_status()


        if self.agent.cookie_path and hasattr(self.agent.action_handler, 'save_cookies'):
            await self.agent.action_handler.save_cookies(self.agent.cookie_path)
        
        # --- User Interaction Dialog ---
        if await self.agent.user_interaction_handler.handle_user_interaction():
            return True # Stop requested


        if self.agent.state.history.is_done():
            if getattr(self.agent.settings, 'validate_output', False) and step < max_steps - 1:
                if not await self.agent._validate_output():
                    return False # Validation failed, continue loop

            await self.agent.log_completion()
            return True # Task done, break loop

        return False

    async def _handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = self.agent.state.history.history[-1] if self.agent.state.history.history else None
        intel = "No specific new information from this step."
        thought = None
        if last_step_info and last_step_info.model_output:
            # Use getattr to safely access 'thought' as it might not be in the schema
            # or might be in a nested structure depending on the specific AgentOutput implementation
            thought = getattr(last_step_info.model_output, "thought", None)
            if thought:
                intel = str(thought) if thought is not None else intel
        
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
                next_action_desc = ", ".join(action_strings) if action_strings else next_action_desc

        user_decision = await self.agent.llm_manager.request_user_decision(intel, next_action_desc)

        if user_decision is False: # User chose No/Abort
            self.agent.state.stopped = True
            return True # Break the loop
        elif isinstance(user_decision, str): # User provided a custom task
            self.agent.task = user_decision 
            self.agent.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")
            return False # Continue the loop, agent will now work on the new task
        return False

    async def request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Requests confirmation/input from the user via WebSocket.
        """
        send_callback = getattr(self.agent, 'send_agent_message_callback', None)
        if not send_callback or not self.agent.confirmation_response_queue:
            return True

        await send_callback({
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
            
            done_tasks, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if confirmation_task in done_tasks:
                response_data = confirmation_task.result()
                response = response_data.get("response")
                custom_task = response_data.get("custom_task")

                if response == "yes": return True
                if response == "no": return False
                if custom_task: return custom_task
            
            if control_task and control_task in done_tasks:
                control_data = control_task.result()
                command = control_data.get("command")
                
                if command == "pause":
                    self.agent.pause()
                    logger.info("Agent paused via user request during confirmation.")
                    if send_callback:
                        await send_callback({"type": "agent_status", "status": "Paused ⏸️"})
                elif command == "resume":
                    self.agent.resume()
                    logger.info("Agent resumed via user request during confirmation.")
                    if send_callback:
                        await send_callback({"type": "agent_status", "status": "Participating"})

                elif command == 'stop':
                    self.agent.stop()
                    return False

class CompletionCheckHandler:
    def __init__(self, agent: Any):
        self.agent = agent

    def handle_max_steps_reached(self):
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

    async def check_completion(self) -> bool:
        return self.agent.state.history.is_done()

    async def _handle_cleanup(self):
        """Handles agent cleanup using the AgentCleanupHandler."""

        if self.agent.browser_context:
                try:
                    if hasattr(self.agent, 'cookie_manager'):
                        await self.agent.cookie_manager.save_cookies(self.agent.browser_context)
                        logger.info("Cookies saved on shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to save cookies on agent shutdown: {e}")

        
        
        if hasattr(self.agent, 'cleanup_handler') and self.agent.cleanup_handler:
            await self.agent.cleanup_handler.handle_cleanup()
        await self.agent._execute_done_callback()
