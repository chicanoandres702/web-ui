
import asyncio
import logging
import re
import os
import asyncio
import sys
from pathlib import Path

from typing import Any, Optional, Set

# Ensure project root is in sys.path so 'src' imports work
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from datetime import datetime
from langchain_core.messages import HumanMessage
from browser_use.agent.views import ActionResult, AgentStepInfo

from src.utils.utils import save_text_to_file
from src.utils.browser_scripts import JS_DETECT_BLOCKING_ELEMENTS, JS_DETECT_NAVIGATION_CONTROLS
from src.agent.browser_use.navigation_recovery import evaluate_site_state
logger = logging.getLogger(__name__)

class BlockingElementChecker:

    """Checks for blocking elements (ads/popups) and injects a warning."""

    def __init__(self, agent):
        self.agent = agent

    async def check_blocking_elements(self):
        """Checks for blocking elements (ads/popups) and injects a warning."""
        try:
            page = await self._get_current_page()
            if not page:
                return

            # 1. Check for CAPTCHA (Priority)
            content = await page.evaluate("document.body.innerText.toLowerCase()")
            if "captcha" in content or "verify you are human" in content or "security check" in content:
                self.agent.inject_message(
                    "SYSTEM ALERT: CAPTCHA detected. You MUST use the `solve_captcha` tool immediately. Do NOT use `clear_view` or `close_difficult_popup` as they will fail.")
                return

            # 2. Check for Chrome Promotion (Auto-Dismiss)
                try:
                    if page is None:
                        return
                    for btn_text in ["No thanks", "Not now", "No, thanks", "Later", "Dismiss"]:
                        # XPath for case-insensitive text match on buttons or links
                        xpath = f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{btn_text.lower()}')] | //a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{btn_text.lower()}')]"
                        element = page.locator(xpath).first
                        if await element.is_visible():
                            await element.click()
                            logger.info(f"🛡️ Heuristics: Auto-dismissed Chrome promotion via '{btn_text}'")
                            self.agent.inject_message(
                                f"System Notification: I automatically dismissed a Chrome promotion popup by clicking '{btn_text}'.")
                            await page.wait_for_timeout(500)

                            return
                except Exception as e:
                    logger.warning(f"Failed to auto-dismiss Chrome popup: {e}")

            is_blocked = await page.evaluate(JS_DETECT_BLOCKING_ELEMENTS)
            if is_blocked:
                msg = "SYSTEM ALERT: A large overlay, ad, or popup seems to be blocking the screen. This may prevent you from interacting with the page. Try using `clear_view` or `close_difficult_popup` to remove it."
                self.agent.inject_message(msg)

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
                    logger.warning(f"Failed to save screenshot of a blocking element: {e}")
        except Exception:
            pass


    async def _get_current_page(self):
        """Safely retrieves the current page from the browser context."""
        try:
            return await self.agent.browser_context.get_current_page()
        except Exception:
            return None


class HeuristicInterface:
    def __init__(self, agent):
        self.agent = agent

class CompletionHeuristicsChecker:
    """Checks for common completion signals in the page content."""

    def __init__(self, agent):
        self.agent = agent

    async def check_completion_heuristics(self):
        """Checks for common completion signals in the page content."""
        try:
            if self.agent.state.history.is_done():
                return

            page = await self._get_current_page()
            if not page:
                return
            content = await page.evaluate("document.body.innerText")
            content_lower = content.lower()

            # 1. Quiz/Test/Course Completion
            if any(kw in self.agent.task.lower() for kw in ["quiz", "test", "exam", "course", "module"]):
                indicators = ["quiz complete", "your score", "results:", "you scored", "thank you for playing",
                              "100%", "completed", "congratulations", "certificate"]
                if any(ind in content_lower for ind in indicators):
                    self.agent.inject_message(
                        "System Notification: The page indicates the task/quiz is complete. If you have the results, please mark the task as Done.")
                    return

            # 2. General Success/Submission
            success_indicators = ["thank you for your order", "order confirmed", "submission received",
                                 "successfully", "message sent", "you are now logged in", "payment successful",
                                 "setup complete"]
            if any(ind in content_lower for ind in success_indicators):
                self.agent.inject_message(
                    "System Notification: I detected a success message on the page. If the task was to submit or order, it appears complete. Verify and finish.")
                return

        except Exception:
            pass

    async def _get_current_page(self):
        """Safely retrieves the current page from the browser context."""
        try:
            return await self.agent.browser_context.get_current_page()
        except Exception:
            return None


class LoginStatusChecker:
    """Checks the login status of the user."""

    def __init__(self, agent):
        self.agent = agent

    async def check_login_status(self):
        """Checks if the user is logged in and injects a notification."""
        try:
            page = await self._get_current_page()
            if not page:
                return
            status_msg = await page.evaluate(JS_DETECT_NAVIGATION_CONTROLS)

            # if "User appears to be Logged In" in status_msg:
            #     self._inject_login_status_message(status_msg)
            # elif "Auth Detected" in status_msg and self.agent.state.consecutive_failures >= 2:
            #     self._inject_login_status_message(status_msg)

        except Exception:
            pass


    async def _get_current_page(self):
        """Safely retrieves the current page from the browser context."""
        try:
            return await self.agent.browser_context.get_current_page()
        except Exception:
            return None
async def _safe_evaluate(page:Any, js_code: str) -> str | None:
    """Safely evaluates JavaScript code on a page and returns the result."""
    try:
        result = await page.evaluate(js_code)
        return str(result) if result is not None else None
    except Exception as e:
        logger.warning(f"Error evaluating JavaScript: {e}")
        return None

async def _safe_take_screenshot(page:Any, path: str):

     try:
          await page.screenshot(path=path)
     except Exception as e: # type: ignore
          logger.warning(f"Failed to save screenshot: {e}")

class AgentHeuristics: # type: ignore
    """
    Encapsulates heuristic checks and behaviors for the BrowserUseAgent.
    This includes detecting blocking elements, loops, progress, and managing model switching.
    """
    def __init__(self, agent):
        self.agent = agent
        self.is_waiting_for_mfa = False # Flag to track if the agent is waiting for MFA approval
        self.processed_urls_for_subtasks = set()

        self.completion_checker = CompletionHeuristicsChecker(self.agent)
        self.blocking_element_checker = BlockingElementChecker(self.agent)
        self.login_status_checker = LoginStatusChecker(self.agent)
    async def _get_current_page(self):
        """Safely retrieves the current page from the browser context."""
        try:
            return await self.agent.browser_context.get_current_page()
        except Exception:
            return None # type: ignore

    def inject_message(self, content: str):

             # 2. Fallback: Attempt to add state message only if state is fully initialized
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

                     logger.warning(f"Failed to inject message via state message: {e}")

    def manage_model_switching(self):
        """Handles Cost Saver and Smart Retry logic."""
        agent = self.agent
        if not getattr(agent, 'model_priority_list', None) or not (getattr(agent, 'enable_smart_retry', False) or getattr(agent, 'enable_cost_saver', False)):
            # Reset to main LLM if not already there
            agent.current_model_index = -1
            agent.switched_to_retry_model = False
            agent.using_cheap_model = False
            return

        current_priority = agent.model_priority_list[agent.current_model_index]['priority'] if agent.current_model_index != -1 else 0

        # --- Smart Retry: Upgrade on failure ---
        if agent.enable_smart_retry and agent.state.consecutive_failures >= 2:
            # Find the next model with a lower priority number (higher strength)
            candidates = [m for m in agent.model_priority_list if m['priority'] < current_priority]
            if not candidates and current_priority > 0: # If we are not on main, we can always go back to main
                pass # Will fall through to the reset logic below
            elif candidates:
                best_fallback = max(candidates, key=lambda x: x['priority'])
                new_index = agent.model_priority_list.index(next(m for m in agent.model_priority_list if m['priority'] == best_fallback['priority']))
                
                if new_index != agent.current_model_index: # type: ignore
                    logger.warning(f'⚠️ Smart Retry: Switching to stronger model (Priority {best_fallback["priority"]}).')
                    agent.llm = best_fallback['llm']

                    agent.current_model_index = new_index
                    agent.switched_to_retry_model = True
                    agent.using_cheap_model = False
                    agent.state.consecutive_failures = 0
                    agent.successful_steps_since_switch = 0 
                    self.inject_message("Switched to a stronger model to resolve issue.")
                    return

        # --- Cost Saver: Downgrade on success ---
        if agent.enable_cost_saver and agent.state.consecutive_failures == 0 and agent.successful_steps_since_switch >= 3:
            # Find the next model with a higher priority number (lower cost/strength)
            candidates = [m for m in agent.model_priority_list if m['priority'] > current_priority]
            if candidates:
                cheapest_option = min(candidates, key=lambda x: x['priority'])
                new_index = agent.model_priority_list.index(next(m for m in agent.model_priority_list if m['priority'] == cheapest_option['priority']))

                if new_index != agent.current_model_index: # type: ignore
                    logger.info(f'💰 Cost Saver: Downgrading to cheaper model (Priority {cheapest_option["priority"]}).')
                    agent.llm = cheapest_option['llm']
                    agent.current_model_index = new_index
                    agent.switched_to_retry_model = False
                    agent.using_cheap_model = True
                    agent.successful_steps_since_switch = 0
                    return

        # --- Fallback: Revert to main model on any failure ---
        if agent.state.consecutive_failures > 0 and agent.current_model_index != -1: # type: ignore
            logger.warning(f"Failure on model (Priority {current_priority}). Reverting to main model: {agent.llm_manager.main_llm}.")
            agent.llm = agent.main_llm
            agent.current_model_index = -1
            agent.switched_to_retry_model = False
            agent.using_cheap_model = False
            agent.successful_steps_since_switch = 0
            return
        
        # --- Track successful steps for cost-saving ---
        if agent.state.consecutive_failures == 0:
            agent.successful_steps_since_switch += 1
        else:
            agent.successful_steps_since_switch = 0

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
        await self.completion_checker.check_completion_heuristics()

    def detect_loop(self):
        """Detects if the agent is performing repetitive actions without state change."""
        try:
            history = self.agent.state.history.history
            if len(history) < 3:
                return False

            # Check if the last 3 actions are identical
            last_actions = []
            for item in history[-3:]:
                if item.model_output and item.model_output.action:
                    last_actions.append(str(item.model_output.action))

            if len(last_actions) == 3 and len(set(last_actions)) == 1:
                logger.warning("🔄 Loop detected: Agent is repeating the same action.")
                self.inject_message("SYSTEM: Loop detected. You have performed the same action 3 times. Please try a different approach or inspect the page more closely.")
                return True
            return False
        except Exception as e:
            logger.debug(f"Error in loop detection: {e}")
            return False

    async def check_navigation_recovery(self):
        """Checks if navigation recovery is needed using specialized assessment logic."""
        task_lower = self.agent.task.lower()
        if not any(kw in task_lower for kw in ["quiz", "test", "module", "class", "course", "assignment"]):
            return

        try:
            if not self.agent.browser_context or not hasattr(self.agent.browser_context, "get_current_page"):
                return  # No browser context or get_current_page method available

            page = await self._get_current_page()
            if not page:
                return
            content = await _safe_evaluate(page, "document.body.innerText")
            full_context = f"{page.url}\n{content}"
            
            step = len(self.agent.state.history.history)
            max_steps = getattr(self.agent.settings, 'max_steps', 100)
            
            # Check for Timeouts in recent history
            history = self.agent.state.history.history
            if history:
                last_result = history[-1].result
                if last_result:
                    error_msg = "".join([r.error for r in last_result if r.error])
                    if "Timeout" in error_msg and "exceeded" in error_msg:
                        self.inject_message("SYSTEM ALERT: The previous action timed out. The page might be loaded but blocked by ads or slow scripts. Try using 'refresh' or 'scroll_down' to trigger a render. Do not retry the exact same navigation immediately.")

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
            page = await self._get_current_page()
            if not page:
                return
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
            page = await self._get_current_page()
            if not page:
                return
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
        if self.agent.state.consecutive_failures >= self.agent.max_consecutive_failures:
            logger.error(f'❌ Stopping due to {self.agent.max_consecutive_failures} consecutive failures')
            if self.agent.auto_save_on_stuck:
                logger.info("💾 Auto-saving page content due to failure...")
                try:
                    page = await self._get_current_page()
                    if not page:
                        return True
                    content = await _safe_evaluate(page, "document.body.innerText")
                    title = await page.title()
                    url = page.url
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stuck_{timestamp}.txt"
                    path = os.path.join("./tmp/agent_history", self.agent.state.agent_id, filename)
                    save_text_to_file(path, f"URL: {url}\nTitle: {title}\n\n{content}")
                    logger.info(f"Saved stuck page content to {path}")
                except Exception as e:
                    logger.error(f"Failed to auto-save stuck page: {e}")
            return True

        return False