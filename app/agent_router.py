from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from app.models import AgentTaskRequest, UserFeedback
from app.task_orchestrator import TaskOrchestrator
from app.websocket import manager
import requests
from app.config import get_settings

router = APIRouter(prefix="/agent", tags=["Agent"])
orc = TaskOrchestrator.get_instance()
settings = get_settings()

@router.get("/ollama/models")
async def get_models():
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.json()
    except: 
        return {"models": []}

@router.post("/run")
async def run_task(req: AgentTaskRequest, request: Request):
    orc.set_callback(manager.broadcast)
    auth_context = request.session.get("google_creds")
    return {"task_id": orc.add_to_queue(req, auth_context)}

@router.post("/feedback")
async def feedback(fb: UserFeedback):
    orc.handle_feedback(fb)
    return {"ok": True}

@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: 
            await ws.receive_text()
    except WebSocketDisconnect: 
        manager.disconnect(ws)
