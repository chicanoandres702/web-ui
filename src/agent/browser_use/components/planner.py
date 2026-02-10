import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
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
    update the agent's execution plan based on page context.
    """
    def __init__(
        self, 
        agent: Any, 
        llm: BaseChatModel,
        interval: float = 4.0
    ):
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
            except Exception:
                pass
            self._task = None
        logger.info("🧠 AgentPlanner stopped.")

    async def create_initial_plan(self, goal: str):
        """Generates the initial breakdown of the user's goal."""
        logger.info("🧠 Planner: Creating initial plan...")
        try:
            prompt = INITIAL_PLANNING_PROMPT.format(goal=goal)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)]) # type: ignore
            
            content = response.content
            if not isinstance(content, str):
                content = json.dumps(content)
            parsed = json_repair.loads(content.strip())
            
            tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
            
            if tasks and self.agent.controller and hasattr(self.agent.controller, "webui_manager"):
                manager = self.agent.controller.webui_manager
                if hasattr(manager, "bu_plan"):

                    # Clear existing and add new
                    manager.bu_plan = []
                    for task in tasks:
                        manager.add_plan_step(task)
                    manager.bu_plan_updated = True
                    logger.info(f"🧠 Planner: Initialized plan with {len(tasks)} steps.")
        except Exception as e:
            logger.error(f"🧠 Planner: Failed to create initial plan: {e}")

    async def reorganize_plan(self):
        """Forces a reorganization of the current plan."""
        logger.info("🧠 Planner: Force Reorganizing Plan...")
        await self._update_plan_cycle(reorg_mode=True)

    async def _update_plan_cycle(self, reorg_mode: bool = False):
        """Single cycle of plan update."""
        pass
        #Removed to here

    async def _loop(self):
        """Background task to dynamically update the plan."""
        pass

    async def _apply_actions(self, manager: Any, actions: List[Dict[str, Any]]):
        pass

def create_agent_planner(agent: Any, llm: BaseChatModel) -> AgentPlanner:
    """Factory function to create an AgentPlanner."""

    return AgentPlanner(agent, llm)