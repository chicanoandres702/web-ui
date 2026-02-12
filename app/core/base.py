from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional
from app.models import AgentResponse

class BaseAgent(ABC):
    """
    Abstract base class ensuring all agents follow the same interface.
    """
    
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    async def run(self, task: str, max_steps: int = 10, callback: Optional[Callable] = None) -> AgentResponse:
        """
        Execute the agent task.
        callback: async function(type: str, data: Any) for real-time updates.
        """
        pass