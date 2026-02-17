from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import requests
import asyncio
from app.models import AgentTaskRequest, UserFeedback, HealthCheck
from app.services.orchestrator import TaskOrchestrator
from app.config import get_settings

router = APIRouter()
orc = TaskOrchestrator.get_instance()
settings = get_settings()

@router.get("/health", response_model=HealthCheck)
async def health():
    return HealthCheck(status="ok", version=settings.VERSION, active_tasks=len(orc.tasks), llm_status="Ready")

@router.get("/ollama/models")
async def get_models():
    try: 
        return await asyncio.to_thread(lambda: requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2).json())
    except: return {"models": []}

@router.post("/agent/run")
async def run_task(req: AgentTaskRequest):
    async def cb(type, data): 
        for c in manager.conns:
            try: await c.send_json({"type": type, "content": data})
            except: pass
    orc.set_callback(cb)
    return {"task_id": orc.add_to_queue(req)}

@router.post("/agent/feedback")
async def feedback(fb: UserFeedback):
    orc.handle_feedback(fb)
    return {"ok": True}

class Manager:
    def __init__(self): self.conns = []
    async def connect(self, ws): await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws): self.conns.remove(ws)
manager = Manager()

@router.websocket("/ws/agent")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)
