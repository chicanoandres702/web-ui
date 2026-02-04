from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime

# from lmnr.sdk.decorators import observe
from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc
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
from src.utils.utils import retry_async
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

        # Prepare arguments for the parent Agent class
        agent_kwargs = kwargs.copy()
        
        # Extract use_memory to avoid passing it to parent Agent which doesn't support it
        # We intercept 'use_memory' to use our own SimpleMemoryManager instead of mem0
        self.use_custom_memory = agent_kwargs.pop('use_memory', False)
        
        # Note: We do NOT pass 'use_memory' to super().__init__ to avoid TypeError if the parent class doesn't support it.

        if confirmer_llm:
            agent_kwargs['validate_output'] = True
            
        if history:
            agent_kwargs['history'] = history

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

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if is_model_without_tool_support(self.model_name):
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

            # Execute initial actions if provided
            if self.initial_actions:
                logger.info(f"🚀 Executing initial actions: {self.initial_actions}")
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
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

            is_confirmed = response_text.upper().startswith("YES")
            
            # Extract reason
            reason = response_text[3:].strip(" .:-")
            if not reason:
                reason = "Confirmed by LLM." if is_confirmed else "Rejected by LLM."

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
            
            self._handle_confirmer_rejection(response_text)
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
        strategy_hint = " STRATEGY UPDATE: Task not confirmed. STOP and RE-ASSESS. 1. Use `assess_page_section` or `get_page_structure` to understand the current state. 2. Use `find_navigation_options` to find the correct path. 3. Scroll if needed."
        full_msg = msg + strategy_hint

        self.heuristics.inject_message(full_msg)

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
                    self._inject_message(msg)
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

        # Refresh HUD
        if self.controller and hasattr(self.controller, "refresh_hud"):
            last_action_str = ""
            if self.state.history.history:
                last_model_output = self.state.history.history[-1].model_output
                if last_model_output:
                    try:
                        actions = last_model_output.action
                        if isinstance(actions, list):
                            last_action_str = " | ".join([str(a) for a in actions])
                        else:
                            last_action_str = str(actions)
                    except: pass
            await self.controller.refresh_hud(self.browser_context, last_action=last_action_str[:100])

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
