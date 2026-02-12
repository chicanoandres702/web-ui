import sys
import asyncio

# Force Windows Loop FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import uuid
import json
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict
import requests

from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.browser import BrowserAgentWrapper
from app.models import (
    TaskStatus, SubTask, AgentTaskRequest, AgentState, UserFeedback
)
from app.auth import router as auth_router

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

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScholarBackend")

os.environ.setdefault("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID)
os.environ.setdefault("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def create_llm(model_override: str = None):
    provider = settings.LLM_PROVIDER.lower()
    model = model_override or settings.MODEL_NAME
    if (model and model.startswith("gemini")) or provider == "gemini":
        gemini_model = model if (model and model.startswith("gemini")) else "gemini-flash-latest"
        
        # Check for service account
        sa_file = "service_account.json"
        if os.path.exists(sa_file):
            try:
                credentials = service_account.Credentials.from_service_account_file(sa_file)
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

router = APIRouter()
orc = TaskOrchestrator.get_instance()

class ConnectionManager:
    def __init__(self): self.conns = []
    async def connect(self, ws: WebSocket): await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws: WebSocket): self.conns.remove(ws)
    async def broadcast(self, type_: str, content: Any):
        for c in self.conns:
            try: await c.send_json({"type": type_, "content": content})
            except: pass

manager = ConnectionManager()

@router.get("/health")
async def health(): return {"status": "ok", "version": settings.VERSION}

@router.get("/ollama/models")
async def get_models():
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.json()
    except: return {"models": []}

@router.post("/agent/run")
async def run_task(req: AgentTaskRequest):
    orc.set_callback(manager.broadcast)
    return {"task_id": orc.add_to_queue(req)}

@router.post("/agent/feedback")
async def feedback(fb: UserFeedback):
    orc.handle_feedback(fb)
    return {"ok": True}

@router.websocket("/ws/agent")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)

def create_app():
    app = FastAPI(title=settings.APP_NAME)
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)
    app.include_router(router, prefix="/api/v1")
    app.include_router(auth_router)
    static_path = os.path.join(settings.BASE_DIR, "static")
    os.makedirs(static_path, exist_ok=True)
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    return app