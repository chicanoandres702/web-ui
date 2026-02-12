"""
This module contains the core backend file definitions for the project setup.
"""
CORE_BACKEND_FILES = {
    "app/models.py": '''from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import uuid
from datetime import datetime

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    WAITING_FOR_USER = "waiting_for_user"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "pending"

class AgentTaskRequest(BaseModel):
    task: str
    class_name: str = "General"
    require_confirmation: bool = True
    model_override: Optional[str] = None
    headless: bool = False
    stream_quality: int = 50

class AgentState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_input: str
    status: TaskStatus = TaskStatus.QUEUED
    plan: List[SubTask] = []
    result: Optional[str] = None
    error: Optional[str] = None
    class_name: str = "General"
    timestamp: datetime = Field(default_factory=datetime.now)

class UserFeedback(BaseModel):
    task_id: str
    approved: bool
    modified_plan: Optional[List[SubTask]] = None
''',
    "app/config.py": '''import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "17.1.1"
    
    GEMINI_API_KEY: str = "" 
    GOOGLE_CLIENT_ID: str = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str = "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"
    SESSION_SECRET: str = "scholar-pro-super-secret-key"
    
    LLM_PROVIDER: str = "ollama" 
    MODEL_NAME: str = "llama3" 
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()
''',
    "app/llm_manager.py": '''import os
import logging
import google.auth
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("ScholarBackend")

def create_llm(model_override: str = None):
    provider = settings.LLM_PROVIDER.lower()
    model = model_override or settings.MODEL_NAME
    if (model and model.startswith("gemini")) or provider == "gemini":
        gemini_model = model if (model and model.startswith("gemini")) else "gemini-flash-latest"
        
        sa_file = "service_account.json"
        if os.path.exists(sa_file):
            try:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_file
                credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
                
                logger.info("Using service account for Gemini.")
                return ChatGoogleGenerativeAI(
                    model=gemini_model,
                    credentials=credentials,
                    temperature=0.1
                )
            except Exception as e:
                logger.error(f"Failed to use service account: {e}. Falling back.")

        # Fallback to API Key
        logger.info("Using API key for Gemini.")
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=model)
''',
    "app/task_orchestrator.py": '''import asyncio
import uuid
import logging
from typing import Dict, List
from app.models import (
    TaskStatus, SubTask, AgentTaskRequest, AgentState, UserFeedback
)
from app.llm_manager import create_llm
from app.browser import BrowserAgentWrapper

logger = logging.getLogger("ScholarBackend")

class TaskOrchestrator:
    _instance = None
    def __init__(self):
        self.tasks: Dict[str, AgentState] = {}
        self.events: Dict[str, asyncio.Event] = {}
        self.queue = asyncio.Queue()
        self.global_callback = None
        self.worker_task = None

    @classmethod
    def get_instance(cls):
        if not cls._instance: cls._instance = cls()
        return cls._instance

    def start_worker(self):
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Task orchestrator worker started.")

    async def stop_worker(self):
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                logger.info("Task orchestrator worker stopped.")
            self.worker_task = None

    def set_callback(self, cb):
        self.global_callback = cb

    async def _worker(self):
        while True:
            try:
                tid, req = await self.queue.get()
                try:
                    await self._process_lifecycle(tid, req)
                except Exception as e:
                    logger.error(f"Task {tid} failed: {e}")
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

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

        try:
            agent_wrapper = BrowserAgentWrapper(llm)
        except Exception as e:
            logger.error(f"Failed to init agent wrapper: {e}")
            return

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
            await agent_wrapper.close_session()

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
''',
    "app/websocket.py": '''from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Any

class ConnectionManager:
    def __init__(self): self.conns = []
    async def connect(self, ws: WebSocket): await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws: WebSocket): self.conns.remove(ws)
    async def broadcast(self, type_: str, content: Any):
        for c in self.conns:
            try: await c.send_json({"type": type_, "content": content})
            except: pass

manager = ConnectionManager()
router = APIRouter()

@router.websocket("/ws/agent")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)
''',
    "app/agent_router.py": '''from fastapi import APIRouter
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
''',
    "app/base_router.py": '''from fastapi import APIRouter
import requests
from app.config import get_settings

settings = get_settings()
router = APIRouter()

@router.get("/health")
async def health(): return {"status": "ok", "version": settings.VERSION}

@router.get("/ollama/models")
async def get_models():
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.json()
    except: return {"models": []}
''',
    "app/lifespan.py": '''from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.task_orchestrator import TaskOrchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    orc = TaskOrchestrator.get_instance()
    orc.start_worker()
    yield
    await orc.stop_worker()
''',
    "app/server.py": '''import os
import sys
import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.config import get_settings
from app.lifespan import lifespan
from app.auth import router as auth_router
from app.agent_router import router as agent_router
from app.base_router import router as base_router
from app.websocket import router as websocket_router

# Force Windows Loop FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

settings = get_settings()
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID)
os.environ.setdefault("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def create_app():
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)
    
    app.include_router(base_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/api/v1")
    app.include_router(auth_router)
    
    static_path = os.path.join(settings.BASE_DIR, "app", "static")
    os.makedirs(static_path, exist_ok=True)
    
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    return app
''',
    "run.py": '''import uvicorn
import sys
import asyncio

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("Starting Scholar Agent Pro...")
    uvicorn.run("app.server:create_app", host="0.0.0.0", port=8000, reload=True, factory=True)
''',
    "app/__init__.py": ""
}
