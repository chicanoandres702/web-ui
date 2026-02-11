import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
import json_repair

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent

from src.utils.prompts import PLANNER_PROMPT, INITIAL_PLANNING_PROMPT

logger = logging.getLogger(__name__)


class AgentPlanner:
    """
    A modular planner component that runs in the background to dynamically 
    update the agent's execution plan.
    """
    def __init__(
        self,
        agent: Any,
        llm: BaseChatModel,
        interval: float = 4.0
    ):  # type: ignore
        self.agent = agent
        self.llm = llm
        self.interval = interval
        self.running = False
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("🧠 AgentPlanner started.")

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel() # type: ignore
            try:
                # We don't await here to avoid blocking sync cleanup
                pass 
            except Exception: # type: ignore
                pass
            self._task = None
        logger.info("🧠 AgentPlanner stopped.")

    async def create_initial_plan(self, goal: str):
        """Generates the initial breakdown of the user's goal."""
        logger.info("🧠 Planner: Creating initial plan...")

        try:
            prompt = INITIAL_PLANNING_PROMPT.format(goal=goal)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            content = response.content
            if not isinstance(content, str):
                content = json.dumps(content)
            parsed = json_repair.loads(content.strip())

            
            tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else [] # type: ignore
            
            if tasks and self.agent.controller and hasattr(self.agent.controller, "webui_manager"):
                manager = self.agent.controller.webui_manager
                if hasattr(manager, "bu_plan"):
                    # Clear existing and add new
                    manager.bu_plan = []
                    for task in tasks:
                        manager.add_plan_step(task) # type: ignore
                    manager.bu_plan_updated = True
                    logger.info(f"🧠 Planner: Initialized plan with {len(tasks)} steps.")
        except Exception as e:
            logger.error(f"🧠 Planner: Failed to create initial plan: {e}")

    async def reorganize_plan(self, reorg_mode: bool = False):
        """Forces a reorganization of the current plan."""

        logger.info("🧠 Planner: Force Reorganizing Plan...")
        await self._update_plan_cycle(reorg_mode=True)

    async def _update_plan_cycle(self, reorg_mode: bool = False):
        """Single cycle of plan update."""
        # Check dependencies
        if not self.agent.controller or not hasattr(self.agent.controller, "webui_manager"):
            return
        
        manager = self.agent.controller.webui_manager
        if not manager or not hasattr(manager, "bu_plan") or not manager.bu_plan:
            return

        try:
            # --- 1. Gather Context ---
            last_step = self.agent.state.history.history[-1] if self.agent.state.history.history else None
            plan_str = json.dumps(manager.bu_plan, indent=2)
            page_summary = getattr(last_step.state, 'element_tree', "") if last_step and hasattr(last_step, 'state') and hasattr(last_step.state, 'element_tree') else "N/A" #getattr(self.agent.state, "page_summary", "N/A")
            last_thought = getattr(last_step.model_output, "thought", "N/A") if last_step and last_step.model_output else "N/A"

            # --- 2. Construct Prompt ---
            prompt = PLANNER_PROMPT.format(
                goal=self.agent.task,
                plan=plan_str,
                last_thought=last_thought,
                page_summary=page_summary
            )

            # --- 3. Call Planner LLM ---
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            plan_update = json_repair.loads(response.content.strip())

            # --- 4. Parse & Apply ---
            if isinstance(plan_update, dict) and "action" in plan_update:
                await self._apply_actions(manager, plan_update)
        except Exception as e:
            logger.error(f"🧠 Planner Cycle Error: {e}")

    async def _loop(self):
        pass

    async def _apply_actions(self, manager: Any, actions: List[Dict[str, Any]]):
        pass

def create_agent_planner(agent: Any, llm: BaseChatModel) -> AgentPlanner:
    """Factory function to create an AgentPlanner."""

    return AgentPlanner(agent, llm)