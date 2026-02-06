from __future__ import annotations

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

# Ensure project root is in sys.path so 'src' imports work
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

# from lmnr.sdk.decorators import observe
from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc # AIMessage is from langchain_core.messages
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentOutput,
    AgentHistoryList,
    AgentStepInfo,
    ToolCallingMethod,
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async
from dotenv import load_dotenv

from browser_use.agent.message_manager.utils import is_model_without_tool_support
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from src.utils.utils import retry_async, parse_json_safe # Ensure parse_json_safe is imported
from langchain_core.runnables import Runnable, RunnableLambda # Import RunnableLambda
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD
from src.utils.memory_utils import get_memory_manager
from src.agent.browser_use.agent_heuristics import AgentHeuristics

load_dotenv()
logger = logging.getLogger(__name__)

from src.utils.io_manager import IOManager

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)

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
        **kwargs
    ):
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness = confirmer_strictness
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
        self.current_model_index = -1 # -1 indicates main_llm
        self.current_step_index = current_step_index
        self.validation_callback = validation_callback

        # Prepare arguments for the parent Agent class
        agent_kwargs = kwargs.copy()
        
        # Extract use_memory to avoid passing it to parent Agent which doesn't support it
        # We intercept 'use_memory' to use our own SimpleMemoryManager instead of mem0
        self.use_custom_memory = agent_kwargs.pop('use_memory', False)
        
        # Extract planner args
        self.planner_llm = agent_kwargs.pop('planner_llm', None)
        self.planner_interval = agent_kwargs.pop('planner_interval', 5.0)
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

            # Then apply the Qwen fix to this (potentially structured) LLM
            # agent_kwargs['llm'] = self._patch_llm_with_qwen_fix(structured_llm)
            agent_kwargs['llm'] = original_llm

        super().__init__(**agent_kwargs)
        
        # Apply system prompt override if provided
        if system_prompt:
            if hasattr(self, "message_manager"):
                self.message_manager.system_message = SystemMessage(content=system_prompt)

        # Link controller to this agent for HUD control
        if self.controller and hasattr(self.controller, "set_agent"):
            self.controller.set_agent(self)

        self.main_llm = self.llm  # Store the primary (high-quality) LLM
        self.last_domain = None
        self.successful_steps_since_switch = 0
        self.memory_manager = get_memory_manager() if self.use_custom_memory else None
        self.heuristics = AgentHeuristics(self)

    def _convert_agent_output_to_aimessage(self, agent_output: AgentOutput) -> BaseMessage:
        """
        Converts an AgentOutput Pydantic model to an AIMessage.
        The AIMessage's content will be the JSON string of the AgentOutput.
        The AIMessage's tool_calls will be derived from AgentOutput.action.
        """
        # The content of the AIMessage will be the JSON string of the AgentOutput
        # This ensures that the agent's message_manager and other parsing logic
        # receive a JSON string, which they expect.
        json_content = agent_output.model_dump_json()
        
        tool_calls = []
        if agent_output.action:
            actions_list = agent_output.action if isinstance(agent_output.action, list) else [agent_output.action]
            for action_model in actions_list:
                if isinstance(action_model, ActionModel): # Ensure it's an ActionModel
                    # LangChain's ToolCall format: {"name": "tool_name", "args": {"arg1": "value1"}}
                    # We need to ensure 'name' is not duplicated in 'args'
                    action_args = action_model.model_dump(exclude_unset=True)
                    tool_name = action_args.pop("name", None) # Extract 'name' and remove from args
                    
                    if tool_name:
                        tool_calls.append({
                            "name": tool_name,
                            "args": action_args
                        })
                    else:
                        logger.warning(f"ActionModel found without a 'name' attribute: {action_model}. Skipping tool call conversion.")
                else:
                    logger.warning(f"Unexpected action type in AgentOutput: {type(action_model)}. Skipping tool call conversion.")

        return AIMessage(content=json_content, tool_calls=tool_calls)

    def _apply_qwen_fix_logic(self, response: BaseMessage) -> BaseMessage:
        """
        Applies the Qwen/Ollama JSON fix logic to an LLM response (BaseMessage).
        This function is designed to be used within a RunnableLambda.
        """
        # Ensure the response is mutable if we need to change its content
        # For AIMessage, content is usually mutable, but for other BaseMessage types, it might not be.
        # We'll assume it's an AIMessage or similar where content can be set.
        if not hasattr(response, 'content') or not isinstance(response.content, str):
            return response

        content = response.content.strip()
        if not content:
            return response

        try:
            logger.debug(f"QwenFixWrapper: Attempting to parse raw LLM output: '{content}'")
            # Attempt to parse with json_repair.loads
            data = parse_json_safe(content) # Use parse_json_safe from utils
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
                     # Attempt to close unclosed JSON object
                     fixed_content = content + '}' * (content.count('{') - content.count('}'))
                     data = json_repair.loads(fixed_content)
                     response.content = json.dumps(data)
                # If all repair attempts fail, the outer exception will catch it,
                # or the content will remain as is if no repair was possible.
                # We don't want to default to "{}" here unless absolutely necessary,
                # as it might mask the actual problematic output.
            except Exception as inner_e:
                logger.warning(f"QwenFixWrapper: Inner repair attempt failed: {inner_e}. Original: '{content}'")
                response.content = "{}"
        return response

    def _patch_llm_with_qwen_fix(self, llm: Runnable) -> Runnable:
        """
        Wraps the LLM with a RunnableLambda that applies the Qwen/Ollama JSON fix logic.
        This avoids direct patching of ainvoke and allows chaining.
        """
        # First, ensure the output is converted to an AIMessage with JSON content
        # if it's an AgentOutput Pydantic model.
        # This RunnableLambda will be applied *after* the structured_llm produces AgentOutput.
        llm_with_aimessage_output = llm | RunnableLambda(self._convert_agent_output_to_aimessage)

        # Then, apply the Qwen fix logic to this AIMessage (which now has JSON string content)
        # The _apply_qwen_fix_logic expects a BaseMessage and will process its string content.
        return llm_with_aimessage_output | RunnableLambda(self._apply_qwen_fix_logic)
    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if 'Ollama' in self.chat_model_library:
                return 'raw'
            # Optimization: Explicitly enable function calling for known capable models
            # This prevents them from falling back to 'raw' mode which causes parsing errors with empty content
            elif any(m in self.model_name.lower() for m in ['qwen', 'deepseek', 'gpt', 'claude']) and 'Ollama' not in self.chat_model_library:
                return 'function_calling'
            elif is_model_without_tool_support(self.model_name):
                return 'raw'
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

    @time_execution_async("--run (agent)")
    async def run(
            self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None,
            on_step_end: AgentHookFunc | None = None
    ) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()

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
                except Exception as e:
                    if "Timeout" in str(e) and "exceeded" in str(e):
                        logger.warning(f"⚠️ Step {step + 1} timed out: {e}. Attempting to recover by reloading page.")
                        self.state.consecutive_failures += 1
                        try:
                            page = await self.heuristics._get_current_page()
                            if page and not page.is_closed():
                                await page.reload(wait_until="domcontentloaded", timeout=15000)
                                self.heuristics.inject_message("SYSTEM: A timeout occurred. I have reloaded the page to recover. Please re-assess the current state.")
                        except Exception as reload_e:
                            logger.error(f"Failed to reload page after timeout: {reload_e}")
                        continue
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

    async def _handle_pre_step(self) -> bool:
        """
        Handles pre-step checks: pause, stop, model switching, failures.
        Returns True if the agent should stop/break the loop.
        """

        # Manage Model Switching (Cost Saver / Smart Retry)
        self._manage_model_switching()

        # Suggest alternative strategies on failure
        self._suggest_alternative_strategy()

        # Close extraneous tabs (e.g. ads, social shares)
        await self._manage_tabs()

        # Custom Memory Injection
        if self.use_custom_memory and self.memory_manager:
            await self._inject_site_knowledge()

        # Check if we should stop due to too many failures
        # Note: self.heuristics is a direct attribute of BrowserUseAgent
        if await self.heuristics.check_max_failures():
            return True

        # Check control flags before each step
        if self.state.stopped:
            logger.info('Agent stopped')
            return True

        while self.state.paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self.state.stopped:  # Allow stopping while paused
                return True
        
        return False

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

        # Check for complex page structure and add subtasks
        await self.heuristics.check_and_add_subtasks()

        # Check login status (New)
        await self.heuristics.check_login_status()

        if self.state.history.is_done():
            if self.settings.validate_output and step < max_steps - 1:
                if not await self._validate_output():
                    return False # Validation failed, continue loop

            await self.log_completion()
            return True # Task done, break loop
        
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
