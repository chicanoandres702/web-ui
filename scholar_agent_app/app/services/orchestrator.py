import asyncio
import uuid
import logging
from app.models import AgentTaskRequest, AgentState, TaskStatus, UserFeedback, SubTask
from app.core.llm_factory import get_llm
from app.agents.browser_agent import BrowserAgentWrapper

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    _instance = None
    def __init__(self):
        self.tasks: dict[str, AgentState] = {}
        self.events: dict[str, asyncio.Event] = {}
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
                logger.error(f"Queue Failure for {tid}: {e}")
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
        llm = get_llm(req.model_override)
        agent = BrowserAgentWrapper(llm)

        # 1. PLAN
        state.status = TaskStatus.PLANNING
        if cb: await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
        state.plan = await agent.decompose_task(req.task, req.class_name)
        if cb: await cb("plan", {"task_id": tid, "plan": [p.model_dump() for p in state.plan]})

        # 2. FEEDBACK
        if req.require_confirmation:
            state.status = TaskStatus.WAITING_FOR_USER
            if cb: await cb("status", {"task_id": tid, "status": "waiting_for_user"})
            await self.events[tid].wait()
            if state.status == TaskStatus.STOPPED: return

        # 3. RUN
        state.status = TaskStatus.RUNNING
        if cb: await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
        
        for sub in state.plan:
            sub.status = "active"
            if cb: 
                await cb("plan_update", sub.model_dump())
            
            success = await agent.run_step(sub.description, req.class_name, req.headless, req.stream_quality, callback=cb)
            
            sub.status = "completed" if success else "failed"
            if cb: 
                await cb("plan_update", sub.model_dump())

        state.status = TaskStatus.COMPLETED
        if cb: 
            await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
            await cb("result", {"task_id": tid, "result": "Research Completed"})

    def handle_feedback(self, fb: UserFeedback):
        state = self.tasks.get(fb.task_id)
        if state:
            if fb.approved:
                if fb.modified_plan: state.plan = fb.modified_plan
                state.status = TaskStatus.RUNNING
                self.events[fb.task_id].set()
            else:
                state.status = TaskStatus.STOPPED
                self.events[fb.task_id].set()
