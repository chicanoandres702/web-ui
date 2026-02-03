import asyncio
import json
import logging
from typing import List, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.browser.context import BrowserContext
from src.controller.custom_controller import CustomController
from src.agent.browser_use.quiz_state import QuizStateManager
from src.agent.browser_use.smart_focus import focus_on_content
from src.agent.browser_use.dom_purification import purify_page
from src.utils import browser_scripts as bs
from src.utils.utils import parse_json_safe, extract_quoted_text
from src.utils.prompts import (
    ENHANCED_AGENT_FOCUS_PROMPT,
    ENHANCED_AGENT_ACTION_SYSTEM_PROMPT,
    ENHANCED_AGENT_ACTION_USER_PROMPT,
    ENHANCED_AGENT_DISCOVERY_PROMPT,
    ENHANCED_AGENT_CHECK_LINK_PROMPT,
    SUBTASK_EXTRACTION_PROMPT,
    RUBRIC_EXTRACTION_PROMPT
)

logger = logging.getLogger(__name__)

class EnhancedDeepResearchAgent:
    """
    An agent that uses an LLM to make smart decisions about navigation, 
    focus, and interaction to achieve a goal on a webpage.
    """
    def __init__(self, llm: BaseChatModel, browser_context: BrowserContext, controller: CustomController):
        self.llm = llm
        self.browser_context = browser_context
        self.controller = controller
        self.state_manager = QuizStateManager()
        self.history: List[Dict[str, Any]] = []
        self._stop_requested = False
        self.discovery_completed = False
        self.consecutive_failures = 0
        self.rubric_constraints = None

    async def stop(self):
        """Signals the agent to stop its execution loop."""
        self._stop_requested = True
        logger.info("Stop requested for EnhancedDeepResearchAgent.")

    async def run(self, goal: str, initial_url: str, max_steps: int = 15) -> List[Dict[str, Any]]:
        """
        Runs the agent to achieve a goal.

        Args:
            goal: The high-level objective.
            initial_url: The starting URL.
            max_steps: The maximum number of steps to execute.

        Returns:
            A list of dictionaries detailing the execution history.
        """
        self._stop_requested = False
        self.history = []
        self.consecutive_failures = 0
        self.rubric_constraints = None

        try:
            page = await self.browser_context.get_current_page()
            await self._execute_action("go_to_url", {"url": initial_url})
            self._add_history_step("Navigate", "completed", f"Navigated to {initial_url}")
            last_result = "Navigation successful."
            await self.wait_for_page_load(page)

            # --- PHASE 1: DISCOVERY & PREREQUISITES ---
            if not self.discovery_completed:
                logger.info("Phase 1: Discovering prerequisite tasks...")
                pre_tasks = await self.discover_tasks(page, goal)
                
                if pre_tasks:
                    self._add_history_step("Discovery", "info", f"Found {len(pre_tasks)} setup tasks: {pre_tasks}")
                    for task in pre_tasks:
                        logger.info(f"Handling prerequisite: {task}")
                        await self._handle_prerequisite(page, task)
                
                self.discovery_completed = True
                logger.info("Discovery complete. Moving to Phase 2.")
                
                # --- PHASE 1.5: DYNAMIC PLANNING ---
                # Check if the page itself offers a plan (e.g. syllabus)
                page_content = await self._get_page_text(page)
                subtasks = await self._extract_subtasks_from_page(page_content, goal)
                if subtasks:
                    self._add_history_step("Planning", "info", f"Extracted {len(subtasks)} subtasks from page: {subtasks}")
                    await self._add_subtasks_to_plan(subtasks)
                
                # --- PHASE 1.6: RUBRIC ANALYSIS ---
                # Check for assignment rubrics to guide quality
                rubric_data = await self._extract_rubric_constraints(page_content, goal)
                if rubric_data and rubric_data.get("is_assignment"):
                    self.rubric_constraints = rubric_data
                    self._add_history_step("Rubric Analysis", "info", f"Extracted rubric: {len(rubric_data.get('scoring_criteria', []))} criteria found.")

            for step in range(max_steps):
                if self._stop_requested:
                    logger.info("Agent run stopped.")
                    self._add_history_step("Stop", "completed", "Agent stopped by user.")
                    break

                # 1. Analyze & Focus
                page_content, pagination_info, analyzed_url = await self._analyze_page(page)
                if step % 3 == 0:
                    await self._handle_focus(goal, page_content, page)

                # 2. Decide
                action_decision = await self._decide_next_action(goal, page_content, last_result, pagination_info, analyzed_url)
                
                # 3. Execute
                last_result = await self._execute_decision(action_decision, goal, analyzed_url)
                
                # 4. Check for finish
                if action_decision.get("action") == "finish":
                    break

        except Exception as e:
            logger.error(f"Error during enhanced agent run: {e}", exc_info=True)
            self._add_history_step("Error", "failed", str(e))

        return self.history

    async def _analyze_page(self, page):
        """Gathers page content and state."""
        current_url = page.url
        page_content = await self._get_page_text(page)
        self.state_manager.sync_from_page(page_content)
        
        # Combine navigation detection with scroll info for wizards/infinite scroll
        nav_signal = await self._detect_navigation_controls(page)
        scroll_info = await self._check_scroll_status(page)
        pagination_info = f"{nav_signal} {scroll_info}".strip()
        
        return page_content, pagination_info, current_url

    async def _handle_focus(self, goal, page_content, page):
        """Determines and applies focus."""
        focus_target = await self._decide_focus(goal, page_content)
        if focus_target:
            await focus_on_content(page, focus_target)
            self._add_history_step("Focus", "completed", f"Focused on '{focus_target}'")

    async def _execute_decision(self, decision: Dict[str, str], goal: str, expected_url: str) -> str:
        """Executes the decided action and returns the result."""
        # Safety Check: Ensure URL hasn't changed since analysis
        page = await self.browser_context.get_current_page()
        if page.url != expected_url:
            self._add_history_step("Safety", "skipped", f"URL changed from {expected_url} to {page.url}")
            return "Action aborted: URL changed during planning. Re-assessing page."

        action = decision.get("action")
        
        # Special handlers that require custom logic
        if action == "go_to_url":
            return await self._handle_go_to_url(decision, goal)
        if action == "finish":
            return await self._handle_finish(decision, goal)
        if action == "wait":
            return await self._handle_wait(decision, goal)
        if action == "handle_dialog":
            return await self._handle_dialog(decision, goal)
        
        # Generic handlers based on configuration
        action_config = self._get_action_config(action)
        if action_config:
            return await self._handle_generic_action(decision, action_config)

        # Default case for unknown actions
        result = f"Action '{action}' is not a recognized action."
        self._add_history_step("Error", "failed", result)
        self.consecutive_failures += 1
        return result

    async def _handle_go_to_url(self, decision: Dict[str, str], goal: str) -> str:
        url = decision.get("url")
        if url and not await self.check_link(url, goal):
            self._add_history_step("Navigation", "skipped", f"Co-Coordinator blocked distraction: {url}")
            return "Navigation blocked by Co-Coordinator."
        res = await self._execute_action("go_to_url", {"url": url})
        self._check_failure(res)
        return res

    def _get_action_config(self, action: str) -> Dict[str, Any]:
        """Returns configuration for generic actions."""
        configs = {
            "click": {"controller_action": "click_element_by_text", "params": {"target": "text"}, "log": "Click"},
            "type": {"controller_action": "type_into_element_by_text", "params": {"target": "text", "value": "value"}, "log": "Type"},
            "upload": {"controller_action": "upload_file_by_text", "params": {"target": "text", "filename": "filename"}, "log": "Upload"},
            "drag_upload": {"controller_action": "upload_file_drag_and_drop_by_text", "params": {"target": "text", "filename": "filename"}, "log": "Drag Upload"},
            "hover_click": {"controller_action": "hover_and_click_by_text", "params": {"hover_target": "hover_text", "click_target": "click_text"}, "log": "Hover & Click"},
            "set_date": {"controller_action": "set_date_by_label", "params": {"target": "label", "value": "date_iso"}, "log": "Set Date"},
            "set_range": {"controller_action": "set_range_by_label", "params": {"target": "label", "value": "value"}, "log": "Set Range"},
            "set_color": {"controller_action": "set_color_by_label", "params": {"target": "label", "value": "color"}, "log": "Set Color"},
            "dismiss_notification_prompt": {"controller_action": "dismiss_notification_prompt", "params": {}, "log": "Dismiss Notification"},
            "dismiss_age_gate": {"controller_action": "dismiss_age_gate", "params": {}, "log": "Dismiss Age Gate"},
        }
        return configs.get(action)

    async def _handle_generic_action(self, decision: Dict[str, str], config: Dict[str, Any]) -> str:
        """Handles generic actions based on configuration."""
        params = {}
        # Map decision keys to controller param keys
        for decision_key, param_key in config["params"].items():
            val = decision.get(decision_key)
            # Special handling for set_range value which needs to be string
            if decision_key == "value" and config["log"] == "Set Range":
                val = str(val)
            
            if val is None:
                return f"Failed: Missing required parameter '{decision_key}' for {config['log']}."
            params[param_key] = val

        res = await self._execute_action(config["controller_action"], params)
        
        # Format log message
        log_detail = ", ".join([f"{k}='{v}'" for k, v in params.items()])
        self._add_history_step(config["log"], "completed", f"{config['log']} ({log_detail}). Result: {res}")
        self._check_failure(res)
        return res

    async def _handle_dialog(self, decision: Dict[str, str], goal: str) -> str:
        dialog_action = decision.get("action", "accept")
        res = await self._execute_action("handle_next_dialog", {"action": dialog_action})
        self._add_history_step("Handle Dialog", "completed", f"Prepared to {dialog_action} next dialog.")
        return res

    async def _handle_finish(self, decision: Dict[str, str], goal: str) -> str:
        reason = decision.get("reason", "Goal achieved.")
        self._add_history_step("Finish", "completed", reason)
        return f"Finished: {reason}"

    async def _handle_wait(self, decision: Dict[str, str], goal: str) -> str:
        result = "Decided to wait and re-evaluate."
        self._add_history_step("Wait", "completed", result)
        await asyncio.sleep(3)
        return result

    def _check_failure(self, result: str):
        if "Error" in result or "Failed" in result or "Could not" in result:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    async def wait_for_page_load(self, page, timeout=5000):
        """Ensures the page is stable and clean before making a move."""
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception:
            pass
        await asyncio.sleep(1)
        try:
            await purify_page(page)
            logger.info("Page stabilized and purified of noise.")
        except Exception as e:
            logger.warning(f"Purification failed: {e}")

    async def discover_tasks(self, page, goal: str) -> List[str]:
        """Scans the page to find mandatory setup tasks."""
        try:
            # Get text content to avoid HTML verbosity
            page_text = await self._get_page_text(page)
            content_sample = page_text[:5000]
            prompt = ENHANCED_AGENT_DISCOVERY_PROMPT.format(goal=goal, content_sample=content_sample)
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            found_tasks = parse_json_safe(response.content)
            return found_tasks if isinstance(found_tasks, list) else []
        except Exception as e:
            logger.warning(f"Discovery phase failed: {e}")
            return []

    async def _extract_subtasks_from_page(self, page_content: str, goal: str) -> List[str]:
        """Analyzes page content to find a structured breakdown of the task."""
        try:
            content_sample = page_content[:6000] # Look at a good chunk of the page
            prompt = SUBTASK_EXTRACTION_PROMPT.format(goal=goal, content_sample=content_sample)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            subtasks = parse_json_safe(response.content)
            return subtasks if isinstance(subtasks, list) else []
        except Exception as e:
            logger.warning(f"Subtask extraction failed: {e}")
            return []

    async def _extract_rubric_constraints(self, page_content: str, goal: str) -> Dict[str, Any]:
        """Analyzes page content to find grading rubrics and requirements."""
        try:
            content_sample = page_content[:8000] # Rubrics can be verbose
            prompt = RUBRIC_EXTRACTION_PROMPT.format(goal=goal, content_sample=content_sample)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            data = parse_json_safe(response.content)
            return data if isinstance(data, dict) and data.get("is_assignment") else None
        except Exception as e:
            logger.warning(f"Rubric extraction failed: {e}")
            return None

    async def _add_subtasks_to_plan(self, subtasks: List[str]):
        """Adds extracted subtasks to the webui manager's plan."""
        if not subtasks: return
        for task in subtasks:
            # Use the controller action to add it properly to the UI and overlay
            await self._execute_action("add_plan_step", {"step_description": task})

    async def _detect_navigation_controls(self, page) -> str:
        """Checks for pagination, wizard steps, and navigation controls."""
        try:
            return await page.evaluate(bs.JS_DETECT_NAVIGATION_CONTROLS)
        except Exception:
            return ""

    async def _check_scroll_status(self, page) -> str:
        """Checks scroll position to detect infinite scroll potential."""
        try:
            info = await page.evaluate(bs.JS_GET_SCROLL_INFO)
            if not info['isAtBottom']:
                return f"Scroll Status: {info['percent']}% viewed. You can 'scroll_down' to see more."
            return "Scroll Status: At bottom."
        except Exception:
            return ""

    async def _handle_prerequisite(self, page, task_desc: str):
        """Handles a single prerequisite task."""
        # Extract potential target text (e.g., "Click 'Start'" -> "Start")
        target_text = extract_quoted_text(task_desc) or task_desc

        # Wait for content to ensure it's loaded (useful for lazy elements)
        await self._execute_action("wait_for_content", {"text": target_text, "timeout": 4})

        success = await focus_on_content(page, target_text)
        if success:
            # If the task description implies clicking, try to click
            if "click" in task_desc.lower() or "accept" in task_desc.lower() or "start" in task_desc.lower():
                await self._execute_action("click_element_by_text", {"text": target_text})
            await asyncio.sleep(2)
            await self.wait_for_page_load(page)

    async def check_link(self, url: str, goal: str) -> bool:
        """Co-coordinator check to distinguish research from social media/ads."""
        prompt = ENHANCED_AGENT_CHECK_LINK_PROMPT.format(goal=goal, url=url)
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return "YES" in res.content.upper()

    async def _execute_action(self, action_name: str, params: Dict[str, Any] = None):
        """Executes an action by name using the controller's registry."""
        if params is None: params = {}
        return await self.controller.execute_action_by_name(action_name, params, self.browser_context)

    async def _decide_focus(self, goal: str, page_content: str) -> str:
        """Asks the LLM what the most important element to focus on is."""
        # Smart Positioning Prompt
        prompt = ENHANCED_AGENT_FOCUS_PROMPT.format(
            goal=goal, 
            status_summary=self.state_manager.get_status_summary(),
            page_content=page_content[:2000]
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip().strip('"')
        except Exception as e:
            logger.warning(f"Failed to decide focus: {e}")
            return ""

    async def _decide_next_action(self, goal: str, page_content: str, last_result: str, pagination_info: str, current_url: str = "") -> Dict[str, str]:
        """Asks the LLM to decide the next action to take."""
        
        # Inject strategy hint if multiple failures occurred
        strategy_hint = ""
        
        # Context-specific hints for academic/social flows
        if "yellowdig" in current_url.lower():
             strategy_hint += "\n\n[Yellowdig Mode]: Use 'post_to_yellowdig' for posts/comments. Ensure content is 40+ words, formal, and professional."
        elif "scholar.google" in current_url.lower():
             strategy_hint += "\n\n[Scholar Mode]: Use 'get_google_scholar_citation' for precision. Use 'get_google_scholar_citations' for bulk extraction."
        elif any(x in current_url.lower() for x in ["canvas", "blackboard", "moodle", "brightspace"]):
             strategy_hint += "\n\n[LMS Mode]: Check 'Modules' or 'Assignments' for tasks. Look for 'Submit Assignment' buttons."

        if self.consecutive_failures >= 2:
            strategy_hint = (
                f"\n\n⚠️ CRITICAL: The previous {self.consecutive_failures} actions failed. "
                "Your current strategy is not working. You MUST try a DIFFERENT approach. "
                "Examples: Use a different selector, try searching instead of clicking, scroll to find the element, or check for popups."
            )

        # Format rubric info for the prompt if available
        rubric_info = "None"
        if self.rubric_constraints:
            criteria = self.rubric_constraints.get('scoring_criteria', [])
            rules = self.rubric_constraints.get('formatting_rules', [])
            rubric_info = f"Formatting: {rules}. Criteria: {json.dumps(criteria[:3])}..." # Truncate to save tokens

        user_prompt = ENHANCED_AGENT_ACTION_USER_PROMPT.format(
            goal=goal,
            status_summary=self.state_manager.get_status_summary(),
            last_result=last_result,
            rubric_constraints=rubric_info,
            page_content=page_content[:2000] + (f"\n\n[System Note]: {pagination_info}" if pagination_info else "")
        ) + strategy_hint

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=ENHANCED_AGENT_ACTION_SYSTEM_PROMPT), 
                HumanMessage(content=user_prompt)
            ])
            action_json = parse_json_safe(response.content)
            return action_json
        except Exception as e:
            logger.error(f"Failed to decide next action: {e}")
            return {"action": "wait"}

    def _add_history_step(self, step: str, status: str, result: Any):
        """Adds a step to the execution history."""
        self.history.append({"step": step, "status": status, "result": str(result)})

    async def _get_page_text(self, page) -> str:
        """Helper to get page text content."""
        return await page.evaluate("document.body.innerText")