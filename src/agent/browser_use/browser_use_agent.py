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
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils.utils import save_text_to_file

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


class BrowserUseAgent(Agent):
    def __init__(self, confirmer_llm=None, confirmer_strictness=None, use_vision_for_confirmer=True, smart_retry_llm=None, cheap_llm=None, enable_cost_saver=False, auto_save_on_stuck=False, source=None, history: AgentHistoryList = None, **kwargs):
        self.confirmer_llm = confirmer_llm
        self.confirmer_strictness = confirmer_strictness
        self.use_vision_for_confirmer = use_vision_for_confirmer
        self.smart_retry_llm = smart_retry_llm
        self.cheap_llm = cheap_llm
        self.enable_cost_saver = enable_cost_saver
        self.auto_save_on_stuck = auto_save_on_stuck
        self.source = source
        self.is_validating = False
        self.switched_to_retry_model = False
        self.using_cheap_model = False
        if confirmer_llm:
            kwargs['validate_output'] = True
        
        # Ensure these are not passed to parent Agent if they somehow ended up in kwargs
        kwargs.pop('confirmer_llm', None)
        kwargs.pop('confirmer_strictness', None)
        kwargs.pop('use_vision_for_confirmer', None)
        kwargs.pop('smart_retry_llm', None)
        kwargs.pop('cheap_llm', None)
        kwargs.pop('enable_cost_saver', None)
        kwargs.pop('auto_save_on_stuck', None)
        kwargs.pop('source', None)

        super().__init__(**kwargs)
        self.main_llm = self.llm  # Store the primary (high-quality) LLM
        if history:
            self.history = history

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
                # Check if waiting for user input after Ctrl+C
                if self.state.paused:
                    signal_handler.wait_for_resume()
                    signal_handler.reset()

                # Cost Saver Logic: Use cheap model by default, switch to main on failure
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

                # Smart Retry Logic: Switch model if failures persist
                if self.smart_retry_llm and not self.switched_to_retry_model and self.state.consecutive_failures >= 2:
                    logger.warning(f'⚠️ Detected {self.state.consecutive_failures} consecutive failures. Switching to Smart Retry Model.')
                    self.llm = self.smart_retry_llm
                    self.switched_to_retry_model = True
                    # Add a system message to inform the new model of the situation
                    if hasattr(self, "message_manager"):
                        self.message_manager.add_state_message(self.state, result=[ActionResult(error=f"Previous model failed {self.state.consecutive_failures} times. You are now in control. Please analyze the situation carefully and try a different approach.", include_in_memory=True)])
                    
                    self.using_cheap_model = False # Ensure we don't show cost saver badge
                    self.state.consecutive_failures = 0  # Reset counter to give new model a chance

                # Check if we should stop due to too many failures
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
                    break

                # Check control flags before each step
                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
                    if self.state.stopped:  # Allow stopping while paused
                        break

                if on_step_start is not None:
                    await on_step_start(self)

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                if on_step_end is not None:
                    await on_step_end(self)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
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

            return self.state.history

        except KeyboardInterrupt:
            # Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history

        finally:
            # Unregister signal handlers before cleanup
            signal_handler.unregister()

            if self.settings.save_playwright_script_path:
                logger.info(
                    f'Agent run finished. Attempting to save Playwright script to: {self.settings.save_playwright_script_path}'
                )
                try:
                    # Extract sensitive data keys if sensitive_data is provided
                    keys = list(self.sensitive_data.keys()) if self.sensitive_data else None
                    # Pass browser and context config to the saving method
                    self.state.history.save_as_playwright_script(
                        self.settings.save_playwright_script_path,
                        sensitive_data_keys=keys,
                        browser_config=self.browser.config,
                        context_config=self.browser_context.config,
                    )
                except Exception as script_gen_err:
                    # Log any error during script generation/saving
                    logger.error(f'Failed to save Playwright script: {script_gen_err}', exc_info=True)

            await self.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)

    async def _validate_output(self) -> bool:
        """
        Validate the output of the agent using the confirmer LLM if available.
        """
        if not self.confirmer_llm:
            return await super()._validate_output()

        self.is_validating = True
        try:
            logger.info(f"🔍 Validating output with Confirmer LLM (Strictness: {self.confirmer_strictness})")

            # Optimize prompt for speed if strictness is low (Fast Mode)
            if self.confirmer_strictness and self.confirmer_strictness <= 3:
                system_prompt = (
                    f"Quickly verify if the task '{self.task}' is done based on the screenshot/text.\n"
                    "Respond 'YES' if it looks mostly correct. Respond 'NO' only if clearly wrong.\n"
                    "Be brief."
                )
            else:
                system_prompt = (
                    "You are a quality assurance validator for a browser automation agent.\n"
                    f"Your task is to verify if the agent has successfully completed the user's request: '{self.task}'.\n"
                    f"Strictness Level: {self.confirmer_strictness or 5}/10 (10 being extremely strict, 1 being lenient).\n"
                    "Analyze the agent's last action and the current browser state.\n"
                    "1. VISUAL INSPECTION: Look at the screenshot (if provided). Does the page visually confirm the task is done? Check for specific elements (e.g., 'Order Confirmed', specific data, green checks) implied by the task.\n"
                    "2. CONTEXT CHECK: Do the URL and page title match the expected outcome?\n"
                    "If the task is completed successfully, respond with 'YES'. If incomplete, incorrect, or needs more steps, respond with 'NO' followed by a short reason."
                )

            history = self.state.history
            last_item = history.history[-1] if history.history else None

            content = []
            content.append({"type": "text", "text": f"User Task: {self.task}"})

            if last_item:
                state_desc = f"Current URL: {last_item.state.url}\nPage Title: {last_item.state.title}\n"
                if last_item.model_output:
                    try:
                        thought = getattr(last_item.model_output, "thought", None)
                        if thought:
                            state_desc += f"Agent Thought: {thought}\n"
                    except Exception:
                        pass

                content.append({"type": "text", "text": state_desc})

                if last_item.state.screenshot and self.use_vision_for_confirmer:
                    image_url = f"data:image/jpeg;base64,{last_item.state.screenshot}"
                    content.append({"type": "image_url", "image_url": {"url": image_url}})

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]

            try:
                response = await self.confirmer_llm.ainvoke(messages)
                response_text = response.content.strip()
                logger.info(f"✅ Confirmer response: {response_text}")

                if response_text.upper().startswith("YES"):
                    return True
                else:
                    msg = f"Confirmer rejected completion: {response_text}"
                    if hasattr(self, "message_manager"):
                        self.message_manager.add_state_message(
                            state=last_item.state,
                            result=[ActionResult(error=msg, include_in_memory=True)],
                            step_info=AgentStepInfo(step_number=len(history.history), max_steps=100),
                        )
                    return False

            except Exception as e:
                logger.error(f"Error during confirmation: {e}")
                return True
        finally:
            self.is_validating = False
