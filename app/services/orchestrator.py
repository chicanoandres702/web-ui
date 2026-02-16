import asyncio
import uuid
import logging
from typing import Dict
from app.models import AgentState, AgentTaskRequest, TaskStatus, UserFeedback
from app.browser import BrowserAgentWrapper
from app.llm_manager import create_llm

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    _instance = None
    def __init__(self):
        self.tasks: Dict[str, AgentState] = {}
        self.events: Dict[str, asyncio.Event] = {}
        self.queue = asyncio.Queue()
        self.global_callback = None
        asyncio.create_task(self._worker())

    @classmethod
    def get_instance(cls):
        if not cls._instance: cls._instance = cls()
        return cls._instance

    def set_callback(self, cb):
        self.global_callback = cb

    async def _worker(self):
        while True:
            tid, req = await self.queue.get()
            try:
                await self._process_lifecycle(tid, req)
            except Exception as e:
                logger.error(f"Task {tid} failed: {e}")
            finally:
                self.queue.task_done()

    def add_to_queue(self, request: AgentTaskRequest) -> str:
        tid = str(uuid.uuid4())
        state = AgentState(task_id=tid, task_input=request.task, class_name=request.class_name)
        self.tasks[tid] = state
        self.events[tid] = asyncio.Event()
        self.queue.put_nowait((tid, request))
        if self.global_callback:
            asyncio.create_task(self.global_callback("queue_update", [t.model_dump() for t in self.tasks.values()]))
        return tid

    async def _process_lifecycle(self, tid, req):
        state = self.tasks[tid]
        cb = self.global_callback
        try:
            llm = create_llm(req.model_override)
        except Exception as e:
            state.status = TaskStatus.FAILED
            state.error = str(e)
            if cb: await cb("log", f"LLM Error: {e}")
            return

        agent_wrapper = BrowserAgentWrapper(llm)
        state.status = TaskStatus.PLANNING
        if cb:
            await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
            await cb("log", f"Planning task: {req.task}")

        state.plan = await agent_wrapper.decompose_task(req.task, req.class_name)
        if cb: await cb("plan", {"task_id": tid, "plan": [p.model_dump() for p in state.plan]})

        if req.require_confirmation:
            state.status = TaskStatus.WAITING_FOR_USER
            if cb: await cb("status", {"task_id": tid, "status": "waiting_for_user"})
            await self.events[tid].wait()
            if state.status == TaskStatus.STOPPED:
                if cb: await cb("log", "Task cancelled.")
                return

        state.status = TaskStatus.RUNNING
        if cb:
            await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
            await cb("status", "running")

        try:
            await agent_wrapper.start_session(headless=req.headless)
            for sub in state.plan:
                sub.status = "active"
                if cb: await cb("plan_update", sub.model_dump())
                success = await agent_wrapper.run_step(
                    sub.description, req.class_name, req.headless, req.stream_quality, callback=cb
                )
                sub.status = "completed" if success else "failed"
                if cb: await cb("plan_update", sub.model_dump())
        except Exception as e:
            state.error = str(e)
            if cb: await cb("log", f"Execution Failed: {e}")
        finally:
            try:
                await agent_wrapper.close_session()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

        state.status = TaskStatus.COMPLETED
        if cb:
            await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
            await cb("result", {"task_id": tid, "result": "Workflow Finished"})

    def handle_feedback(self, fb: UserFeedback):
        state = self.tasks.get(fb.task_id)
        if state:
            if fb.approved:
                if fb.modified_plan: state.plan = fb.modified_plan
                state.status = TaskStatus.RUNNING
            else:
                state.status = TaskStatus.STOPPED
            self.events[fb.task_id].set()