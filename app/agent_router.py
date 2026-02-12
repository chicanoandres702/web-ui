from fastapi import APIRouter
from app.models import AgentTaskRequest, UserFeedback
from app.task_orchestrator import TaskOrchestrator
from app.websocket import manager

router = APIRouter()
orc = TaskOrchestrator.get_instance()

@router.post("/agent/run")
async def run_task(req: AgentTaskRequest):
    orc.set_callback(manager.broadcast)
    return {"task_id": orc.add_to_queue(req)}

@router.post("/agent/feedback")
async def feedback(fb: UserFeedback):
    orc.handle_feedback(fb)
    return {"ok": True}
