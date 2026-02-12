import os
import subprocess
import sys
import json
import shutil

# ==========================================
#       ENVIRONMENT & DEPENDENCIES
# ==========================================

def sync_dependencies():
    """Sophisticated dependency management with Windows-safe fallbacks."""
    print("--- 🚀 Scholar Agent Pro: Initializing Environment ---")
    
    core_packages = [
        "fastapi", "uvicorn[standard]", "pydantic>=2.0", "pydantic-settings",
        "langchain-google-genai", "langchain-openai", "langchain-community",
        "langchain-ollama", "huggingface_hub", "browser-use", "tqdm", 
        "requests", "python-multipart", "python-dotenv", "google-auth", 
        "google-auth-oauthlib", "playwright"
    ]
    
    try:
        print("Installing core libraries...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", *core_packages])
        
        print("Ensuring Playwright browsers are installed...")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        
        if os.name == 'nt':
            print("Detected Windows: Installing optimized Llama-CPP binaries...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "llama-cpp-python", 
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
                ])
            except subprocess.CalledProcessError:
                print("⚠️ Local Llama-CPP binary skipped. System will use Ollama API.")
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            
        print("--- ✅ Environment Ready ---")
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup Error: {e}")
        sys.exit(1)

# ==========================================
#           PROJECT FILE MAP
# ==========================================

files = {
    "app/config.py": """import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "16.0.0-unified"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Auth & Identity
    API_SECRET_KEY: str = "scholar-nexus-ultra-secret-2024"
    GOOGLE_CLIENT_ID: str = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com" 
    GOOGLE_SERVICE_ACCOUNT_JSON: str = "service_account.json"
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama" 
    MODEL_NAME: str = "llama3" 
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GEMINI_API_KEY: str = "" # Add your key if needed
    
    # Storage
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage")
    USER_DATA_DIR: str = os.path.join(os.getcwd(), "storage", "browser_data")
    KNOWLEDGE_BASE_DIR: str = os.path.join(os.getcwd(), "storage", "knowledge_bases")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def ensure_dirs(self):
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        os.makedirs(self.USER_DATA_DIR, exist_ok=True)
        os.makedirs(self.KNOWLEDGE_BASE_DIR, exist_ok=True)

@lru_cache()
def get_settings():
    s = Settings()
    s.ensure_dirs()
    return s
""",

    "app/models.py": """from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

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
    editable: bool = True

class AgentTaskRequest(BaseModel):
    task: str
    class_name: str = "General"
    require_confirmation: bool = True
    model_override: Optional[str] = None
    headless: bool = False
    stream_quality: int = 50

class UserFeedback(BaseModel):
    task_id: str
    approved: bool
    modified_plan: Optional[List[SubTask]] = None

class AgentState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_input: str
    status: TaskStatus = TaskStatus.QUEUED
    plan: List[SubTask] = []
    result: Optional[str] = None
    error: Optional[str] = None
    class_name: str = "General"
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthCheck(BaseModel):
    status: str
    version: str
    active_tasks: int
    llm_status: str
""",

    "app/core/llm_factory.py": """import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str = None):
        p = settings.LLM_PROVIDER.lower()
        m = model_name or settings.MODEL_NAME
        
        # Priority for Gemini models
        if m and m.startswith("gemini-"):
            return ChatGoogleGenerativeAI(model=m, google_api_key=settings.GEMINI_API_KEY)

        if p == "ollama":
            return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=m)
        
        # Fallback
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)

def get_llm(model_name=None):
    return LLMFactory.create_llm(model_name)
""",

    "app/agents/browser_agent.py": """import json
import logging
import asyncio
import base64
from app.models import SubTask
from langchain.schema import HumanMessage

try:
    from browser_use import Agent as BUAgent
    from browser_use.browser.browser import Browser, BrowserConfig
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

logger = logging.getLogger(__name__)

class BrowserAgentWrapper:
    def __init__(self, llm):
        self.llm = llm

    async def decompose_task(self, task: str, class_name: str):
        prompt = f"Break academic task '{task}' for {class_name} into 3 steps. Return ONLY a JSON list of strings."
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)]).content
            start, end = res.find('['), res.rfind(']') + 1
            steps = json.loads(res[start:end])
            return [SubTask(description=s) for s in steps]
        except:
            return [SubTask(description=task)]

    async def run_step(self, description: str, class_name: str, headless: bool = False, quality: int = 50, callback=None) -> bool:
        if not HAS_LIBS: 
            await asyncio.sleep(1); return True
        
        browser = None
        try:
            browser = Browser(config=BrowserConfig(headless=headless))
            ctx = await browser.new_context()
            agent = BUAgent(task=description, llm=self.llm, browser_context=ctx)
            
            run_task = asyncio.create_task(agent.run(max_steps=15))
            
            while not run_task.done():
                try:
                    page = await ctx.get_current_page()
                    if page:
                        screenshot = await page.screenshot(type='jpeg', quality=quality)
                        encoded = base64.b64encode(screenshot).decode('utf-8')
                        if callback:
                            await callback("browser_stream", {
                                "image": encoded,
                                "url": page.url,
                                "title": await page.title()
                            })
                except:
                    pass
                await asyncio.sleep(0.5)
            
            await run_task
            await ctx.close()
            await browser.close()
            return True
        except Exception as e:
            logger.error(f"Browser Execution Error: {e}")
            if browser: await browser.close()
            return False
""",

    "app/services/orchestrator.py": """import asyncio
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
""",

    "app/api/endpoints.py": """from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import requests
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
    try: return requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2).json()
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
""",

    "app/main.py": """import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router as api_router
from app.config import get_settings

# CRITICAL WINDOWS SUBPROCESS FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix="/api/v1")

static_p = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_p): os.makedirs(static_p)
app.mount("/", StaticFiles(directory=static_p, html=True), name="static")
""",

    "app/server.py": """import uvicorn
import sys, os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
""",

    "app/static/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scholar Pro | Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    <style>
        :root { --bg:#0b0d11; --pnl:#151921; --acc:#3b82f6; --txt:#e0e6ed; --input:#1c232d; --border:#2d3748; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; color: var(--txt); }
        body { background: var(--bg); height: 100vh; overflow: hidden; }
        .shell { display: grid; grid-template-columns: 280px 1fr 300px; height: 100vh; }
        aside { background: var(--pnl); padding: 1.2rem; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; }
        aside#right { border-right: none; border-left: 1px solid var(--border); }
        main { padding: 1.2rem; display: flex; flex-direction: column; gap: 1rem; overflow: hidden; }
        input, select { width: 100%; background: var(--input); border: 1px solid var(--border); color: #fff; padding: 10px; border-radius: 8px; font-size: 13px; }
        button { background: var(--acc); color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: filter 0.2s; }
        button:hover { filter: brightness(1.2); }
        .viewport { flex: 1; background: #000; border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; position: relative; }
        .browser-bar { background: #1a202c; padding: 8px 12px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid var(--border); }
        .url-display { flex: 1; background: #000; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-family: monospace; color: #a0aec0; white-space: nowrap; overflow: hidden; }
        .log-box { height: 140px; background: #000; border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        .item { background: var(--input); padding: 10px; border-radius: 8px; margin-bottom: 8px; font-size: 11px; display: flex; align-items: center; gap: 10px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #4a5568; }
        .active .dot, .running .dot { background: var(--acc); box-shadow: 0 0 10px var(--acc); }
        .completed .dot { background: #10b981; }
        .modal-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); display:none; align-items:center; justify-content:center; z-index: 1000; }
        .modal { background: #1e212b; padding: 30px; border-radius: 12px; width: 400px; border: 1px solid var(--acc); }
    </style>
</head>
<body>
    <div class="shell">
        <aside id="left">
            <h1>Scholar Pro</h1>
            <label>CONTEXT</label><input id="class-name" value="History 101">
            <label>MODE</label><select id="provider-mode"><option value="ollama">Ollama (Local)</option><option value="gemini">Gemini (Cloud)</option></select>
            <label>MODEL</label><select id="model-select"></select>
            <label style="display:flex; align-items:center; gap:8px;"><input type="checkbox" id="headless" style="width:auto;"> Headless Mode</label>
            <div id="g_id_onload" data-client_id="566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com" data-callback="onGoogleAuth"></div>
            <div class="g_id_signin" data-type="standard" style="margin-top:1rem;"></div>
            <div style="margin-top:1rem; border-top:1px solid var(--border); padding-top:1rem;"><h3>QUEUE</h3><div id="queue-box"></div></div>
            <div id="user-info" style="display:none; margin-top: auto; align-items: center; gap: 10px; border-top:1px solid var(--border); padding-top:1rem;"><img id="user-avatar" style="width:32px; height:32px; border-radius:50%;"><span id="user-name"></span></div>
        </aside>
        <main>
            <div style="display:flex; gap:10px;"><input id="task-input" placeholder="Enter objective..." style="margin:0;"><button id="run-btn" style="width:100px;">EXECUTE</button></div>
            <div class="viewport">
                <div class="browser-bar"><div class="url-display" id="url-bar">about:blank</div><div id="browser-status" style="font-size:10px;">STANDBY</div></div>
                <div style="flex:1; display:flex; align-items:center; justify-content:center; background:#111;"><img id="live-stream" style="max-width:100%; max-height:100%;"></div>
            </div>
            <div class="log-box" id="log-box"></div>
        </main>
        <aside id="right">
            <h3>PLAN</h3><div id="plan-box" style="flex:1; overflow-y:auto; margin-top:10px;"></div>
            <div style="margin-top:1rem; border-top:1px solid var(--border); padding-top:1rem;"><h3>KNOWLEDGE</h3><textarea style="width:100%; height:120px; background:transparent; border:none; resize:none; font-size:11px; color:#718096;" readonly>Synthesized data...</textarea></div>
        </aside>
    </div>
    <div class="modal-overlay" id="user-modal"><div class="modal"><h2>Approve Workflow</h2><p style="font-size:14px; opacity:0.7; margin-bottom:20px;">The AI agent has generated a task sequence.</p><div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;"><button id="ok-btn" style="background:#10b981;">APPROVE</button><button id="no-btn" style="background:#ef4444;">CANCEL</button></div></div></div>
    <script>
        let ws, taskId, planState = [], googleUser = null;
        window.onGoogleAuth = (r) => {
            const p = JSON.parse(atob(r.credential.split('.')[1])); googleUser = p;
            document.getElementById('user-info').style.display = 'flex';
            document.getElementById('user-name').innerText = p.name;
            document.getElementById('user-avatar').src = p.picture;
            refreshModelList();
        };
        async function refreshModelList() {
            const mode = document.getElementById('provider-mode').value;
            const select = document.getElementById('model-select');
            if (mode === 'ollama') {
                try {
                    const r = await fetch('/api/v1/ollama/models');
                    const d = await r.json();
                    select.innerHTML = (d.models || []).map(m => `<option value="${m.name}">${m.name}</option>`).join('');
                } catch(e) { select.innerHTML = '<option>Ollama Offline</option>'; }
            } else if (mode === 'gemini') {
                select.innerHTML = googleUser ? '<option value="gemini-flash-latest">gemini-flash-latest</option>' : '<option value="gemini-1.5-flash">gemini-1.5-flash</option>';
            }
        }
        async function init() {
            const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${protocol}://${location.host}/api/v1/ws/agent`);
            ws.onmessage = e => handleMsg(JSON.parse(e.data));
            document.getElementById('run-btn').onclick = async () => {
                await fetch('/api/v1/agent/run', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({
                        task: document.getElementById('task-input').value, 
                        class_name: document.getElementById('class-name').value,
                        model_override: document.getElementById('model-select').value,
                        headless: document.getElementById('headless').checked
                    })
                });
            };
            document.getElementById('ok-btn').onclick = () => feedback(true);
            document.getElementById('no-btn').onclick = () => feedback(false);
            document.getElementById('provider-mode').onchange = refreshModelList;
            refreshModelList();
        }
        async function feedback(approved) {
            document.getElementById('user-modal').style.display = 'none';
            await fetch('/api/v1/agent/feedback', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({task_id: taskId, approved, modified_plan: planState})
            });
        }
        function handleMsg(m) {
            const box = document.getElementById('log-box');
            if(m.type === 'status' && m.content.status === 'waiting_for_user') { taskId = m.content.task_id; document.getElementById('user-modal').style.display = 'flex'; }
            if(m.type === 'browser_stream') {
                document.getElementById('live-stream').src = 'data:image/jpeg;base64,' + m.content.image;
                document.getElementById('url-bar').innerText = m.content.url;
                document.getElementById('browser-status').innerText = 'LIVE: ' + m.content.title.substring(0,12);
            }
            if(m.type === 'log') { box.innerHTML += `<div>> ${m.content}</div>`; box.scrollTop = box.scrollHeight; }
            if(m.type === 'plan') {
                taskId = m.content.task_id; planState = m.content.plan;
                const b = document.getElementById('plan-box'); b.innerHTML = '';
                planState.forEach(s => b.innerHTML += `<div class="item" id="s-${s.id}"><div class="dot"></div><input value="${s.description}" style="margin:0; border:none; background:transparent; font-size:12px; color:white; flex:1;"></div>`);
            }
            if(m.type === 'plan_update') { const el = document.getElementById(`s-${m.content.id}`); if(el) el.className = `item ${m.content.status}`; }
            if(m.type === 'queue_update') {
                const qb = document.getElementById('queue-box'); qb.innerHTML = '';
                m.content.forEach(t => qb.innerHTML += `<div class="item ${t.status}"><div class="dot"></div><div style="font-size:11px;">${t.task_input.substring(0,18)}...<br><small style="opacity:0.5">${t.status}</small></div></div>`);
            }
        }
        window.onload = init;
    </script>
</body>
</html>
"""
}

# ==========================================
#           INSTALLER LOGIC
# ==========================================

def create_project(base_dir="scholar_agent_app"):
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    print(f"Creating project in: {os.path.abspath(base_dir)}")
    for filepath, content in files.items():
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f: f.write(content)
    
    sync_dependencies()
    print("\\n--- 🚀 STARTING ROBUST COMMAND CENTER ---")
    try:
        subprocess.run([sys.executable, "-m", "app.server"], cwd=base_dir)
    except KeyboardInterrupt:
        print("\nShutdown complete.")

if __name__ == "__main__":
    create_project()