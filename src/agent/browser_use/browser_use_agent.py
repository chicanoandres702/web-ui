from __future__ import annotations

import asyncio
import logging
import os
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
from src.utils.utils import save_text_to_file, retry_async
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD
from src.utils.memory_utils import configure_mem0

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


class BrowserUseAgent(Agent):
    def __init__(
        self,
        confirmer_llm=None,
        confirmer_strictness=None,
        use_vision_for_confirmer=True,
        smart_retry_llm=None,
        cheap_llm=None,
        enable_cost_saver=False,
        auto_save_on_stuck=False,
        source=None,
        history: AgentHistoryList = None,
        save_history_path: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness = confirmer_strictness
        self.use_vision_for_confirmer = use_vision_for_confirmer
        self.smart_retry_llm = smart_retry_llm
        self.cheap_llm = cheap_llm
        self.enable_cost_saver = enable_cost_saver
        self.auto_save_on_stuck = auto_save_on_stuck
        self.source = source
        self.save_history_path = save_history_path
        self.is_validating = False
        self.switched_to_retry_model = False
        self.using_cheap_model = False

        # Prepare arguments for the parent Agent class
        agent_kwargs = kwargs.copy()
        
        # Extract use_memory to avoid passing it to parent Agent which doesn't support it
        use_memory = agent_kwargs.pop('use_memory', False)

        if confirmer_llm:
            agent_kwargs['validate_output'] = True
            
        if history:
            agent_kwargs['history'] = history

        if use_memory:
            configure_mem0()

        super().__init__(**agent_kwargs)
        
        # Apply system prompt override if provided
        if system_prompt:
            if hasattr(self, "message_manager"):
                self.message_manager.system_message = SystemMessage(content=system_prompt)

        self.main_llm = self.llm  # Store the primary (high-quality) LLM

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

        # Set up the Ctrl+C signal handler with callbacks specific to this agent
        from browser_use.utils import SignalHandler

        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=None,  # No special cleanup needed on forced exit
            exit_on_second_int=True,
        )
        signal_handler.register()

        try:
            self._log_agent_run()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                if await self._handle_pre_step(signal_handler):
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

        finally:
            # Unregister signal handlers before cleanup
            signal_handler.unregister()
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

            if response_text.upper().startswith("YES"):
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
        if hasattr(self, "message_manager"):
            history = self.state.history
            last_item = history.history[-1] if history.history else None
            if last_item:
                self.message_manager.add_state_message(
                    state=last_item.state,
                    result=[ActionResult(error=msg, include_in_memory=True)],
                    step_info=AgentStepInfo(step_number=len(history.history), max_steps=100),
                )

    def _manage_model_switching(self):
        """Handles Cost Saver and Smart Retry logic."""
        # Cost Saver Logic
        if self.enable_cost_saver and self.cheap_llm and not self.switched_to_retry_model:
            if self.state.consecutive_failures == 0:
                if self.llm != self.cheap_llm:
                    logger.info('💰 Cost Saver: Switching to Cheap LLM.')
                    self.llm = self.cheap_llm
                    self.using_cheap_model = True
            else:
                # If we have failures, switch back to main LLM to recover
                if self.llm != self.main_llm:
                    logger.info(f'⚠️ Cost Saver: Detected {self.state.consecutive_failures} failures. Switching to Main LLM.')
                    self.llm = self.main_llm
                    self.using_cheap_model = False

        # Smart Retry Logic
        if self.smart_retry_llm and not self.switched_to_retry_model and self.state.consecutive_failures >= 2:
            logger.warning(f'⚠️ Detected {self.state.consecutive_failures} consecutive failures. Switching to Smart Retry Model.')
            self.llm = self.smart_retry_llm
            self.switched_to_retry_model = True
            # Add a system message to inform the new model of the situation
            if hasattr(self, "message_manager"):
                try:
                    self.message_manager.add_state_message(self.state, result=[ActionResult(error=f"Previous model failed {self.state.consecutive_failures} times. You are now in control. Please analyze the situation carefully and try a different approach.", include_in_memory=True)])
                except Exception as e:
                    logger.warning(f"Failed to add retry state message: {e}")
            
            self.using_cheap_model = False # Ensure we don't show cost saver badge
            self.state.consecutive_failures = 0  # Reset counter to give new model a chance

    async def _check_max_failures(self) -> bool:
        """Checks if max failures reached and handles auto-save."""
        if self.state.consecutive_failures >= self.settings.max_failures:
            logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
            if self.auto_save_on_stuck:
                logger.info("💾 Auto-saving page content due to failure...")
                try:
                    page = await self.browser_context.get_current_page()
                    content = await page.evaluate("document.body.innerText")
                    title = await page.title()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stuck_{timestamp}.txt"
                    path = os.path.join("./tmp/agent_history", self.state.agent_id, filename)
                    save_text_to_file(path, f"URL: {page.url}\nTitle: {title}\n\n{content}")
                    logger.info(f"Saved stuck page content to {path}")
                except Exception as e:
                    logger.error(f"Failed to auto-save stuck page: {e}")
            return True
        return False

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

    async def _handle_pre_step(self, signal_handler) -> bool:
        """
        Handles pre-step checks: pause, stop, model switching, failures.
        Returns True if the agent should stop/break the loop.
        """
        # Check if waiting for user input after Ctrl+C
        if self.state.paused:
            signal_handler.wait_for_resume()
            signal_handler.reset()

        # Manage Model Switching (Cost Saver / Smart Retry)
        self._manage_model_switching()

        # Check if we should stop due to too many failures
        if await self._check_max_failures():
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
                self.save_history(self.save_history_path)
            except Exception as e:
                logger.warning(f"Failed to auto-save history: {e}")

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

    async def _handle_cleanup(self):
        """Handles cleanup after run."""
        await self.close()

        if self.settings.generate_gif:
            output_path: str = 'agent_history.gif'
            if isinstance(self.settings.generate_gif, str):
                output_path = self.settings.generate_gif
            create_history_gif(task=self.task, history=self.state.history, output_path=output_path)

        # Execute Done Callback
        await self._execute_done_callback()
