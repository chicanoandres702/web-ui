import asyncio
import logging
from typing import Optional, Any, Coroutine

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages the lifecycle of an asynchronous agent task (start, stop, pause, resume).
    Decouples the UI from the raw asyncio task and agent state.
    """
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._agent: Optional[Any] = None

    @property
    def is_running(self) -> bool:
        """Checks if the task is currently active."""
        return self._task is not None and not self._task.done()

    @property
    def is_paused(self) -> bool:
        """Checks if the agent is currently paused."""
        if self._agent and hasattr(self._agent, "state"):
            return getattr(self._agent.state, "paused", False)
        return False

    async def start(self, coro: Coroutine, agent: Any) -> asyncio.Task:
        """
        Starts a new task with the given coroutine and agent instance.
        Stops any existing task first.
        """
        if self.is_running:
            await self.stop()
        
        self._agent = agent
        self._task = asyncio.create_task(coro)
        return self._task

    async def stop(self):
        """
        Stops the current task.
        Attempts a graceful stop by setting agent flags, then cancels if necessary.
        """
        if self._agent:
            # Signal the agent to stop
            if hasattr(self._agent, "state"):
                self._agent.state.stopped = True
                # If paused, resume so it can process the stop flag
                if self._agent.state.paused:
                    self._agent.state.paused = False
            elif hasattr(self._agent, "stop"):
                 if asyncio.iscoroutinefunction(self._agent.stop):
                     await self._agent.stop()
                 else:
                     self._agent.stop()

        if self._task:
            if not self._task.done():
                try:
                    # Wait briefly for the agent to exit its loop gracefully
                    # Increased timeout to allow for cleanup of current step
                    await asyncio.wait_for(asyncio.shield(self._task), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Task did not stop gracefully, forcing cancellation.")
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        logger.info("Task cancelled successfully.")
                    except Exception as e:
                        logger.error(f"Error during task cancellation: {e}")
                except Exception as e:
                    logger.error(f"Error stopping task: {e}")
            
            self._task = None
        self._agent = None

    def pause(self):
        """Pauses the agent."""
        if self._agent:
            if hasattr(self._agent, "pause"):
                self._agent.pause()
            elif hasattr(self._agent, "state"):
                self._agent.state.paused = True

    def resume(self):
        """Resumes the agent."""
        if self._agent:
            if hasattr(self._agent, "resume"):
                self._agent.resume()
            elif hasattr(self._agent, "state"):
                self._agent.state.paused = False
