import logging
import re
import os
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage
from browser_use.agent.views import ActionResult, AgentStepInfo
from src.utils.utils import save_text_to_file
from src.utils.browser_scripts import JS_DETECT_BLOCKING_ELEMENTS, JS_DETECT_NAVIGATION_CONTROLS
from src.agent.browser_use.navigation_recovery import evaluate_site_state

logger = logging.getLogger(__name__)

class AgentHeuristics:
    """
    Encapsulates heuristic checks and behaviors for the BrowserUseAgent.
    This includes detecting blocking elements, loops, progress, and managing model switching.
    """
    def __init__(self, agent):
        self.agent = agent
        self.cheap_model_probation = 0  # Steps to wait before retrying cheap model after failure
        self.processed_urls_for_subtasks = set()

    def inject_message(self, content: str):
        """Injects a system message or state update into the agent's message manager."""
        if hasattr(self.agent, "message_manager"):
             # Attempt to add state message only if state is fully initialized
             has_element_tree = False
             if hasattr(self.agent.state, "element_tree") and self.agent.state.element_tree:
                 has_element_tree = True
             elif hasattr(self.agent.state, "browser_state") and hasattr(self.agent.state.browser_state, "element_tree") and self.agent.state.browser_state.element_tree:
                 has_element_tree = True

             if has_element_tree:
                 try:
                    step = len(self.agent.state.history.history) + 1
                    self.agent.message_manager.add_state_message(
                        self.agent.state, 
                        result=[ActionResult(error=content, include_in_memory=True)],
                        step_info=AgentStepInfo(step_number=step, max_steps=100)
                    )
                    return
                 except Exception as e:
                     logger.warning(f"Failed to inject strategy hint via state message: {e}")
             
             # Fallback: Try adding as a HumanMessage
             try:
                 msg = HumanMessage(content=f"System Notification: {content}")
                 if hasattr(self.agent.message_manager, "add_message"):
                    self.agent.message_manager.add_message(msg)
                 elif hasattr(self.agent.message_manager, "add_new_task"):
                    self.agent.message_manager.add_new_task(f"System Notification: {content}")
                 elif hasattr(self.agent.message_manager, "messages") and isinstance(self.agent.message_manager.messages, list):
                    self.agent.message_manager.messages.append(msg)
                 elif hasattr(self.agent.message_manager, "_messages") and isinstance(self.agent.message_manager._messages, list):
                    self.agent.message_manager._messages.append(msg)
                 else:
                     logger.warning(f"MessageManager missing add_message method and messages list for fallback injection.")
             except Exception as inner_e:
                 logger.warning(f"Failed to inject strategy hint via fallback: {inner_e}")

    def manage_model_switching(self):
        """Handles Cost Saver and Smart Retry logic."""
        
        # Check if we just failed with the cheap model
        if self.agent.using_cheap_model and self.agent.state.consecutive_failures > 0:
            self.cheap_model_probation = 3  # Penalty: Stay on main model for 3 successful steps
            logger.info(f"📉 Cheap model failed. Applying probation for {self.cheap_model_probation} steps.")

        # Decrement probation if we are succeeding
        if self.cheap_model_probation > 0 and self.agent.state.consecutive_failures == 0:
            self.cheap_model_probation -= 1

        # Cost Saver Logic
        if self.agent.enable_cost_saver and self.agent.cheap_llm and not self.agent.switched_to_retry_model:
            # Only switch to cheap if no failures AND no probation
            if self.agent.state.consecutive_failures == 0 and self.cheap_model_probation == 0:
                if self.agent.llm != self.agent.cheap_llm:
                    logger.info('💰 Cost Saver: Switching to Cheap LLM.')
                    self.agent.llm = self.agent.cheap_llm
                    self.agent.using_cheap_model = True
            else:
                # If we have failures OR probation, ensure we are on Main LLM (unless already on Smart Retry)
                if self.agent.llm != self.agent.main_llm:
                    reason = "Failures detected" if self.agent.state.consecutive_failures > 0 else "Cheap model probation"
                    logger.info(f'⚠️ Cost Saver: {reason}. Switching to Main LLM.')
                    self.agent.llm = self.agent.main_llm
                    self.agent.using_cheap_model = False

        # Smart Retry Logic
        if self.agent.smart_retry_llm and not self.agent.switched_to_retry_model and self.agent.state.consecutive_failures >= 2:
            logger.warning(f'⚠️ Detected {self.agent.state.consecutive_failures} consecutive failures. Switching to Smart Retry Model.')
            self.agent.llm = self.agent.smart_retry_llm
            self.agent.switched_to_retry_model = True
            # Add a system message to inform the new model of the situation
            msg = f"Previous model failed {self.agent.state.consecutive_failures} times. You are now in control. Please analyze the situation carefully and try a different approach."
            self.inject_message(msg)
            
            self.agent.using_cheap_model = False # Ensure we don't show cost saver badge
            self.agent.state.consecutive_failures = 0  # Reset counter to give new model a chance

    def suggest_alternative_strategy(self):
        """Injects strategy hints if the agent is failing."""
        if self.agent.state.consecutive_failures == 0:
            return

        if self.agent.state.consecutive_failures == 1:
            msg = "⚠️ Action failed. Strategy Hint: Check if the element is covered by a popup or ad. Try using 'clear_view' to clean the page, scroll to make it visible, or use a different selector."
            self.inject_message(msg)
        elif self.agent.state.consecutive_failures > 1:
             msg = f"⚠️ Action failed {self.agent.state.consecutive_failures} times. STOP repeating the same action. You MUST try a different strategy (e.g. search instead of click, go back, or use a different tool)."
             self.inject_message(msg)

    async def check_completion_heuristics(self):
        """Checks for common completion signals in the page content."""
        try:
            if self.agent.state.history.is_done():
                return

            if "quiz" in self.agent.task.lower() or "test" in self.agent.task.lower():
                page = await self.agent.browser_context.get_current_page()
                content = await page.evaluate("document.body.innerText")
                content_lower = content.lower()
                indicators = ["quiz complete", "your score", "results:", "you scored", "thank you for playing", "100%", "completed"]
                if any(ind in content_lower for ind in indicators):
                     self.inject_message("System Notification: The page seems to indicate the quiz is complete (found score/results). If you have answered all questions, please mark the task as Done.")
        except Exception:
            pass

    def detect_loop(self):
        """Detects if the agent is performing repetitive actions without state change."""
        try:
            history = self.agent.state.history.history
            if len(history) < 2:
                return

            last_step = history[-1]
            prev_step = history[-2]

            # Check if URL is the same and actions are identical
            if last_step.state.url == prev_step.state.url:
                last_actions = getattr(last_step.model_output, "action", [])
                prev_actions = getattr(prev_step.model_output, "action", [])
                
                if last_actions and prev_actions and str(last_actions) == str(prev_actions):
                    # Ignore scrolling actions as they are often repeated validly
                    action_str = str(last_actions).lower()
                    if "scroll_down" in action_str or "scroll_up" in action_str or "wait" in action_str or "sleep" in action_str:
                        return

                    logger.warning("🔄 Loop detected: Identical action on same URL.")
                    self.inject_message("SYSTEM ALERT: You are repeating the exact same action on the same page. This suggests the previous action did not have the desired effect. DO NOT repeat it. Try scrolling, using a different element, or refreshing.")
        except Exception as e:
            logger.warning(f"Error in loop detection: {e}")

    async def check_blocking_elements(self):
        """Checks for blocking elements (ads/popups) and injects a warning."""
        try:
            page = await self.agent.browser_context.get_current_page()
            is_blocked = await page.evaluate(JS_DETECT_BLOCKING_ELEMENTS)
            if is_blocked:
                msg = "SYSTEM ALERT: A large overlay, ad, or popup seems to be blocking the screen. This may prevent you from interacting with the page. Try using `clear_view` or `close_difficult_popup` to remove it."
                self.inject_message(msg)
                
                # Auto-save screenshot of the blocking element
                try:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    agent_id = getattr(self.agent.state, "agent_id", "unknown_agent")
                    filename = f"blocked_{timestamp}.png"
                    save_path = os.path.join("./tmp/agent_history", agent_id, "debug_screenshots")
                    os.makedirs(save_path, exist_ok=True)
                    full_path = os.path.join(save_path, filename)
                    await page.screenshot(path=full_path)
                    logger.info(f"📸 Saved screenshot of blocking element: {full_path}")
                except Exception as e:
                    logger.warning(f"Failed to save blocking element screenshot: {e}")
        except Exception:
            pass

    async def check_login_status(self):
        """Checks if the user is logged in and injects a notification."""
        try:
            page = await self.agent.browser_context.get_current_page()
            status_msg = await page.evaluate(JS_DETECT_NAVIGATION_CONTROLS)
            
            if "User appears to be Logged In" in status_msg:
                 self.inject_message(f"System Notification: {status_msg} STOP attempting to log in. Proceed to the dashboard or next task.")
            elif "Auth Detected" in status_msg and self.agent.state.consecutive_failures >= 2:
                 self.inject_message("System Notification: You seem stuck on a login page. If you believe you are logged in, try navigating to the target URL directly.")
        except Exception:
            pass

    async def check_navigation_recovery(self):
        """Checks if navigation recovery is needed using specialized assessment logic."""
        task_lower = self.agent.task.lower()
        if not any(kw in task_lower for kw in ["quiz", "test", "module", "class", "course", "assignment"]):
            return

        try:
            page = await self.agent.browser_context.get_current_page()
            content = await page.evaluate("document.body.innerText")
            full_context = f"{page.url}\n{content}"
            
            step = len(self.agent.state.history.history)
            max_steps = getattr(self.agent.settings, 'max_steps', 100)
            
            assessment = evaluate_site_state(full_context, step, max_steps)
            
            if assessment["status"] != "on_track":
                msg = f"SYSTEM ALERT: Navigation Assessment: {assessment['status']}. Reason: {assessment['reason']}"
                if assessment["action_required"]:
                    if assessment["action_required"] == "RENAVIGATE":
                        msg += " Recommended Action: Use 'go_to_url' to try the URL again."
                    msg += f" Recommended Action: {assessment['action_required']}"
                self.inject_message(msg)
        except Exception as e:
            logger.warning(f"Error in navigation recovery check: {e}")

    async def check_and_add_subtasks(self):
        """
        Checks if the current page contains a list of subtasks relevant to the goal.
        If found, automatically adds them to the plan via the controller.
        """
        if not self.agent.controller or not hasattr(self.agent.controller, "webui_manager"):
            return

        manager = self.agent.controller.webui_manager
        if not hasattr(manager, "bu_plan") or not manager.bu_plan:
            return # No plan to add to

        try:
            page = await self.agent.browser_context.get_current_page()
            url = page.url
            
            if url in self.processed_urls_for_subtasks:
                return
            
            self.processed_urls_for_subtasks.add(url)
            
            text = await page.evaluate("document.body.innerText")
            if len(text) < 200: return
            
            # Heuristic keywords to identify structured content
            keywords = ["module", "chapter", "week", "unit", "lesson", "assignment", "quiz", "task list", "steps to complete"]
            if not any(kw in text.lower() for kw in keywords):
                return
                
            from src.utils.prompts import SUBTASK_EXTRACTION_PROMPT
            from src.utils.utils import parse_json_safe
            
            content_sample = text[:6000]
            prompt = SUBTASK_EXTRACTION_PROMPT.format(goal=self.agent.task, content_sample=content_sample)
            
            response = await self.agent.llm.ainvoke([HumanMessage(content=prompt)])
            subtasks = parse_json_safe(response.content)
            
            if isinstance(subtasks, list) and len(subtasks) > 0:
                # Find insertion point (after current pending step)
                current_idx = next((i for i, s in enumerate(manager.bu_plan) if s["status"] == "pending"), -1)
                insert_pos = current_idx + 1 if current_idx != -1 else len(manager.bu_plan)
                
                for task_desc in reversed(subtasks):
                    manager.bu_plan.insert(insert_pos, {"step": task_desc, "status": "pending"})
                
                manager.bu_plan_updated = True
                self.inject_message(f"System Notification: I detected a complex structure and automatically added {len(subtasks)} subtasks to your plan based on the page content.")
                logger.info(f"Automatically added {len(subtasks)} subtasks from {url}")
        except Exception as e:
            logger.warning(f"Error in check_and_add_subtasks: {e}")

    async def detect_progress(self):
        """Detects progress indicators (e.g. 'Question 3 of 10') and injects them."""
        try:
            page = await self.agent.browser_context.get_current_page()
            text = await page.evaluate("document.body.innerText")
            
            matches = re.findall(r'(?:Question|Step|Page)\s*(\d+)\s*(?:of|/)\s*(\d+)', text, re.IGNORECASE)
            
            if matches:
                current, total = matches[-1]
                try:
                    curr_int = int(current)
                    tot_int = int(total)
                    if curr_int <= tot_int:
                        msg = f"System Notification: Progress detected - Step {current} of {total}. "
                        if curr_int == tot_int:
                            msg += "This is the FINAL step. Complete it and look for a Submit/Finish button."
                        else:
                            msg += "You are moving forward. Continue to the next step."
                        self.inject_message(msg)
                except ValueError:
                    pass
        except Exception:
            pass

    async def check_max_failures(self) -> bool:
        """Checks if max failures reached and handles auto-save."""
        if self.agent.state.consecutive_failures >= self.agent.settings.max_failures:
            logger.error(f'❌ Stopping due to {self.agent.settings.max_failures} consecutive failures')
            if self.agent.auto_save_on_stuck:
                logger.info("💾 Auto-saving page content due to failure...")
                try:
                    page = await self.agent.browser_context.get_current_page()
                    content = await page.evaluate("document.body.innerText")
                    title = await page.title()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stuck_{timestamp}.txt"
                    path = os.path.join("./tmp/agent_history", self.agent.state.agent_id, filename)
                    save_text_to_file(path, f"URL: {page.url}\nTitle: {title}\n\n{content}")
                    logger.info(f"Saved stuck page content to {path}")
                except Exception as e:
                    logger.error(f"Failed to auto-save stuck page: {e}")
            return True
        return False
