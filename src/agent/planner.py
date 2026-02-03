import json
import logging
import re
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.prompts import DEFAULT_PLANNER_PROMPT
from src.utils.utils import clean_json_string, parse_json_safe
from src.agent.browser_use.navigation_recovery import evaluate_site_state, generate_recovery_plan
from src.agent.browser_use.dom_purification import purify_page

logger = logging.getLogger(__name__)

def _clean_plan_step(step: str) -> str:
    """Removes leading numbering (e.g., '1. ') from plan steps."""
    return re.sub(r'^\d+\.\s*', '', step).strip()

async def generate_hierarchical_plan(
    llm: BaseChatModel, 
    task: str, 
    system_prompt: Optional[str] = None,
    additional_context: Optional[str] = None
) -> List[str]:
    """
    Generates a hierarchical plan (list of subtasks) for the given task using the provided LLM.
    """
    prompt_content = system_prompt if system_prompt and system_prompt.strip() else DEFAULT_PLANNER_PROMPT
    
    user_content = f"User Task: {task}"
    if additional_context:
        user_content += f"\n\n{additional_context}\n\nIMPORTANT: The agent is currently at the state described above. Adjust the plan to start from this state (e.g., do not navigate to the URL if already there, do not login if already logged in)."

    messages = [
        SystemMessage(content=prompt_content),
        HumanMessage(content=user_content)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        plan = parse_json_safe(response.content)
        if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
            # Clean up steps
            cleaned_plan = [_clean_plan_step(step) for step in plan]
            logger.info(f"Generated Plan: {cleaned_plan}")
            return cleaned_plan
        else:
            logger.warning(f"Planner output was not a list of strings: {response.content}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error in planner: {e}. Content: {response.content[:100]}...")
        return []
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        return []

class Planner:
    """
    Main planning component for browser-based agents.
    Now includes automated reassessment and recovery logic.
    """
    
    def __init__(self, model_provider=None):
        self.model_provider = model_provider
        self.current_plan = []
        self.execution_history = []

    def create_initial_plan(self, goal: str, initial_context: str = "") -> List[Dict[str, str]]:
        """
        Generates the baseline plan with mandatory reassessment anchors.
        """
        # We explicitly inject 'Anchoring' and 'Validation' steps into every plan
        base_plan = [
            {
                "step": f"Navigate to the target site and wait for stable state.",
                "status": "pending",
                "type": "navigation"
            },
            {
                "step": "Identify and dismiss any modal overlays, vignettes, or cookie banners.",
                "status": "pending",
                "type": "anchoring"
            },
            {
                "step": f"Locate primary content container related to goal: {goal}",
                "status": "pending",
                "type": "extraction"
            },
            {
                "step": "Execute core task actions while monitoring progress indicators.",
                "status": "pending",
                "type": "action"
            },
            {
                "step": "Final validation: Confirm task success against the original goal.",
                "status": "pending",
                "type": "validation"
            }
        ]
        self.current_plan = base_plan
        return self.current_plan

    def reassess_state(self, page_content: str, current_step_idx: int) -> Dict[str, Any]:
        """
        Automated reassessment loop. Calls the assessment module to check for 'Confusion'.
        """
        assessment = evaluate_site_state(page_content, current_step_idx, len(self.current_plan))
        
        if assessment["status"] != "on_track":
            logger.warning(f"Planner Reassessment: {assessment['reason']}")
            # If the agent is lost or blocked, we patch the plan with recovery steps
            recovery_steps = generate_recovery_plan(assessment["reason"])
            self._patch_plan(current_step_idx, recovery_steps)
            
        return assessment

    def _patch_plan(self, index: int, new_steps: List[Dict]):
        """
        Inserts recovery steps into the current execution queue.
        """
        for i, step in enumerate(new_steps):
            step["status"] = "pending"
            step["is_recovery"] = True
            self.current_plan.insert(index + i, step)
        
        logger.info(f"Plan patched with {len(new_steps)} recovery steps.")

    def update_step_status(self, index: int, status: str, result: str = ""):
        """Updates plan progress and logs history."""
        if 0 <= index < len(self.current_plan):
            self.current_plan[index]["status"] = status
            self.execution_history.append({
                "step": self.current_plan[index]["step"],
                "status": status,
                "result": result
            })

    def get_next_step(self) -> Optional[Dict[str, str]]:
        """Retrieves the next pending step."""
        for step in self.current_plan:
            if step["status"] == "pending":
                return step
        return None

    async def execute_purification_cycle(self, page) -> bool:
        """Executes the DOM purification script on the given page."""
        result = await purify_page(page)
        return result.get("success", False)