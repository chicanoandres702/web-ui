from __future__ import annotations

from pathlib import Path
import logging
from queue import Empty
import sys
import os

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.agent.browser_use.control_queue_handler import ControlQueueHandler
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    from browser_use.agent.service import Agent
except ImportError:
    Agent = object  # Fallback if browser_use is not installed


import abc

class LocalModelParser:
    """
    A reusable component for local AI-based parsing and conversion.
    Encapsulates the logic for using lightweight local models (like TinyLlama)
    to process or fix LLM outputs.
    """
    def __init__(self, model_name: str = "tiny-llama-q4_k_m.gguf"):
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), "models", model_name)
        self._llm_instance = None

    def _ensure_model_downloaded(self):
        """Downloads the model if it doesn't exist."""
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Downloading local model: {self.model_name}...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                import urllib.request
                url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                urllib.request.urlretrieve(url, self.model_path)
                logger.info(f"Downloaded local model: {self.model_name} to {self.model_path}")
            else:
                logger.info(f"Local model already exists at {self.model_path}")
        except Exception as e:  # type: ignore
            logger.error(f"Error downloading or verifying local model: {e}")
            
    def get_instance(self):
        if not self._llm_instance:
            try:
                from llama_cpp import Llama # type: ignore
            except ImportError:
                logger.error("llama-cpp-python not installed. Please install it to use LocalModelParser.")
                return None
            self._ensure_model_downloaded()
            self._llm_instance = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
            logger.info(f"Loaded local model: {self.model_name} from {self.model_path}")
        return self._llm_instance

    async def parse_text_to_json(self, text: str) -> str:
        """Converts raw text to a JSON string using the local model."""
        try:
            llm = self.get_instance()
            if not llm:
                return "{}"
            prompt = f"<|system|>\nConvert this text to valid JSON tool calls:\n{text}\n<|assistant|>\n"
            output = llm(prompt, max_tokens=512, stop=["<|endoftext|>"], echo=False)
            return output['choices'][0]['text'].strip()
        except:
            return "{}"
        """
        Initializes agent components and applies system prompt override.
        """
        ComponentInitializer(self, agent_kwargs, system_prompt) # type: ignore
        self.browser_factory = BrowserFactory(self)

        self.site_knowledge_injector = SiteKnowledgeInjector(self) # type: ignore

        self.user_interaction_handler = UserInteractionHandler(self) # type: ignore
        self.cleanup_handler = AgentCleanupHandler(self) # type: ignore
        self.tab_handler = TabManagementHandler(self) # type: ignore
        self.cookie_handler = CookieHandler(self) # type: ignore


        self.post_step_handler = PostStepHandler(self) # type: ignore

class CookieHandler:
    def __init__(self, agent: Any):
        """Initializes the CookieHandler with the agent instance."""
        self.agent = agent

    async def load_cookies(self, browser_context: Any, cookie_path: str) -> None:
        """Loads cookies from a file into the browser context."""
        pass

class BrowserUseAgent(Agent): # Ensure inheritance if intended, or keep as is if wrapper

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
            if step_info.step_number == 0:
                logger.warning(f"⚠️ Step {step_info.step_number + 1} timed out: {e}. Attempting to recover by reloading page.")
                self.state.consecutive_failures += 1
                await self._recover_from_timeout()
            else:
                logger.error(f"An unexpected error occurred during step {step_info.step_number + 1}: {e}")
                self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step_info.step_number + 1}: {e}")
                raise e            
            try:
                await self._handle_pre_step()
            except asyncio.TimeoutError as e:
                logger.warning(f"⚠️ Step {step_info.step_number + 1} timed out: {e}. Attempting to recover by reloading page.")
                self.state.consecutive_failures += 1
                await self._recover_from_timeout()
            except Exception as e:
                logger.error(f"An unexpected error occurred during step {step_info.step_number + 1}: {e}")
                self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step_info.step_number + 1}: {e}")
                raise e
                logger.warning(f"⚠️ Step {step_info.step_number + 1} timed out: {e}. Attempting to recover by reloading page.")
                self.consecutive_failures += 1
                await self._recover_from_timeout()
            except Exception as e:
                logger.error(f"An unexpected error occurred during step {step_info.step_number + 1}: {e}")
                self.heuristics.inject_message(f"SYSTEM: An error occurred during step {step_info.step_number + 1}: {e}")
                raise e
    
    async def _handle_pre_step(self) -> bool:
        """Handles pre-step logic including control queue and model switching."""
        return await self.pre_step_handler.handle_pre_step()

    async def _log_agent_run(self):
        """Logs the start of the agent run."""
        logger.info(f"🚀 Starting Agent Run - Task: {self.task}")

    async def log_completion(self):
        """Logs the completion of the task."""
        logger.info(f"✅ Task completed successfully: {self.task}")
        

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
        async def _load_cookies(self):
            """Loads cookies from a file into the browser context."""
            pass

        async def _manage_tabs(self):
            """Closes extraneous tabs to keep the agent focused."""
            pass

        async def _inject_site_knowledge(self):
            """Checks if we have entered a new domain and injects relevant knowledge."""
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


class PostStepHandler:
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent
    async def handle_post_step(self, step: int, max_steps: int) -> bool:

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

    
class AgentCleanupHandler:
    """Handles cleanup tasks when the agent stops or finishes."""

    def __init__(self, agent: Any):
        self.agent = agent

    async def handle_cleanup(self) -> None:
        """Performs cleanup: stops planner, saves cookies, and closes browser if needed."""
        logger.info("🧹 Agent cleanup initiated...")
        self.agent.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")
        return False # Continue the loop, agent will now work on the new task

    async def request_user_confirmation(self, intel: str, next_task: str) -> bool | str:
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

class BrowserFactory:
    @staticmethod
    def get_default_chrome_path() -> str:
        """
        Returns the default Chrome user data path based on the operating system.
        """
        if sys.platform == "win32":
            return os.path.join(os.environ["LOCALAPPDATA"], "Google\\Chrome\\User Data")
        elif sys.platform == "darwin":
            return os.path.expanduser("~/Library/Application Support/Google/Chrome")
        else:  # Linux
            return os.path.expanduser("~/.config/google-chrome")

    async def create_native_browser(self, headless: bool = False) -> Any:
        """
        Creates a browser instance using the native default Chrome installation and user data.
        """
        from browser_use.browser.browser import Browser, BrowserConfig
        
        chrome_path = self.get_default_chrome_path()

        logger.info(f"🚀 Launching native browser with user data from: {chrome_path}")
        
        config = BrowserConfig(
        headless=headless,
        extra_chromium_args=[
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={chrome_path}"
        ]
        )
        self.browser = Browser(config=config)
        return self.browser

class BrowserManager:
    """
    Handles browser creation and context management
    """
    def __init__(self, agent):
        self.agent = agent
        self.browser = None
        self.context = None

    async def create_browser(self, config: Dict[str, Any]) -> Any:
        """
        Creates and configures a browser instance based on provided settings.
        """
        from browser_use.browser.browser import Browser, BrowserConfig
        
        browser_config = BrowserConfig(
            headless=config.get('headless', False),
            disable_security=config.get('disable_security', True),
            extra_chromium_args=config.get('extra_chromium_args', []),
            wss_url=config.get('wss_url'),
            chrome_instance_path=config.get('chrome_instance_path')
        )

        self.browser = Browser(config=browser_config)
        return self.browser

    async def create_native_browser(self, config: Dict[str, Any]) -> Any:
        """
        Creates and configures a browser instance based on provided settings.
        """
        from browser_use.browser.browser import Browser, BrowserConfig

        browser_config = BrowserConfig(
            headless=config.get('headless', False),
            disable_security=config.get('disable_security', True),
            extra_chromium_args=config.get('extra_chromium_args', []),
            wss_url=config.get('wss_url'),
            chrome_instance_path=config.get('chrome_instance_path')
        )

        self.browser = Browser(config=browser_config)
        return self.browser

    async def close_browser(self):
        self.browser = None
        
class PostStepHandler:
    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def handle_post_step(self, step: int, max_steps: int) -> bool:
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
    
        # Check for progress indicators
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

class AgentCleanupHandler:
    """Handles cleanup tasks when the agent stops or finishes."""
    def __init__(self, agent: Any):
        self.agent = agent

    async def handle_cleanup(self):
        """Performs cleanup: stops planner, saves cookies, and closes browser if needed."""
        logger.info("🧹 Agent cleanup initiated...")
        
        # Stop background planner
        if hasattr(self.agent, 'planner_task') and self.agent.planner_task:
            self.agent.planner_task.cancel()
            try:
                await self.agent.planner_task
            except asyncio.CancelledError:
                pass

        # Save cookies before closing
        if hasattr(self.agent, 'cookie_handler') and self.agent.cookie_path:
            await self.agent.action_handler.save_cookies(self.agent.cookie_path)

        # Close browser context if not inhibited
        if not getattr(self.agent, 'inhibit_close', False):
            if self.agent.browser_context:
                await self.agent.browser_context.close()
            if hasattr(self.agent, 'browser') and self.agent.browser:
                await self.agent.browser.close()
        
        # Execute final callback
        if hasattr(self.agent, '_execute_done_callback'):
            await self.agent._execute_done_callback()
        
        logger.info("✅ Cleanup complete.")

class ControlQueueProcessor:
    def __init__(self, agent: Any):
        self.agent = agent

    async def process(self) -> None: 
        if not self.agent.agent_control_queue:
            return

        try:
            while True:
                try:
                    control_data = self.agent.agent_control_queue.get_nowait()
                    command = control_data.get("command")

                    if command == "pause":
                        self.agent.pause()
                        logger.info("Agent paused via control queue.")
                        if self.agent.send_agent_message_callback:
                            await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Paused ⏸️"})
                    elif command == "resume":
                        self.agent.resume()
                        logger.info("Agent resumed via control queue.")
                        if self.agent.send_agent_message_callback:
                            await self.agent.send_agent_message_callback({"type": "agent_status", "status": "Participating"})
                    elif command == "stop":
                        self.agent.stop()
                        logger.info("Agent stopped via control queue.")
                except Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing control queue: {e}")
            
    async def handle_command(self, command: str) -> bool:
        """Handles specific control commands."""
        if command == 'stop':
            self.agent.stop()
            return True
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


load_dotenv()
class BrowserActionHandler:
    """Handles browser-specific actions and state checks."""
    def __init__(self, browser_context: Any, controller: Any, heuristics: Any):
        self.browser_context = browser_context
        self.controller = controller
        self.heuristics = heuristics

    async def save_cookies(self, cookie_path: str) -> None:
        """Saves current session cookies to a file."""
        if not self.browser_context:
            return
        try:
            playwright_context = getattr(self.browser_context, 'context', None)
            if playwright_context and hasattr(playwright_context, 'cookies'):
                cookies = await playwright_context.cookies()
                os.makedirs(os.path.dirname(cookie_path), exist_ok=True)
                with open(cookie_path, 'w', encoding='utf-8') as f:
                    json.dump(cookies, f, indent=2)
                logger.info(f"Saved cookies to {cookie_path}")
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")

    async def check_blocking_elements(self) -> None:
        """Checks for overlays or popups that might block interaction."""
        # Implementation for detecting and handling overlays
        pass

    async def check_navigation_recovery(self) -> None:
        """Checks if the agent is stuck on a blank or error page."""
        pass

    async def check_login_status(self) -> None:
        """Heuristic check to see if the agent was unexpectedly logged out."""
        pass


class TabManagementHandler:
    """Manages browser tabs to maintain focus and performance."""
    def __init__(self, agent: Any):
        self.agent = agent

    async def manage_tabs(self) -> None:
        """Closes extraneous tabs like ads or popups."""
        nav_controller = getattr(self.agent.controller, 'nav_controller', None)
        if nav_controller and self.agent.browser_context:
            await nav_controller.manage_tabs(self.agent.browser_context)


class ComponentInitializer:
    """Initializes and configures agent components and system prompts."""
    def __init__(self, agent: Any):
        self.agent = agent

    def initialize_components(self, agent_kwargs: Dict[str, Any], system_prompt: Optional[str] = None):
            """
            Initializes modular components for the agent and applies system prompt overrides.
            """
            self.agent.browser_factory = BrowserFactory(self.agent)
            self.agent.site_knowledge_injector = SiteKnowledgeInjector(self.agent)
            self.agent.user_interaction_handler = UserInteractionHandler(self.agent)
            self.agent.cleanup_handler = AgentCleanupHandler(self.agent)
            self.agent.tab_handler = TabManagementHandler(self.agent)
            self.agent.cookie_handler = CookieHandler(self.agent)
            self.agent.post_step_handler = PostStepHandler(self.agent)
            self.agent.pre_step_handler = PreStepHandler(self.agent)
            self.agent.control_queue_handler = ControlQueueHandler(self.agent)
    
            if system_prompt:
                self._apply_system_prompt_override(system_prompt)
    
    def _apply_system_prompt_override(self, system_prompt: str):
        """
        Overrides the default system prompt in the message manager.
        """
        try:
            if hasattr(self.agent, 'message_manager'):
                # browser-use MessageManager typically stores the system prompt
                # We update it to ensure the LLM receives the custom instructions
                self.agent.message_manager.system_prompt = system_prompt
                logger.info("Custom system prompt applied to MessageManager.")
        except Exception as e:
            logger.error(f"Failed to apply system prompt override: {e}")
    
    async def retry_async(coro_func, retries=3, delay=2, logger=None, error_message="Operation failed"):
        """
        Utility function to retry asynchronous operations.
        """
        for i in range(retries):
            try:
                return await coro_func()
            except Exception as e:
                if logger:
                    logger.warning(f"{error_message} (Attempt {i+1}/{retries}): {e}")
                if i == retries - 1:
                    raise
