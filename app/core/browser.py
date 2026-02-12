import os
import sys
import subprocess
import time
import json

# ==========================================
#        PROJECT FILE DEFINITIONS
# ==========================================

FILES = {
    # ---------------------------------------------------------
    # 1. SHARED MODELS
    # ---------------------------------------------------------
    "app/models.py": """from pydantic import BaseModel, Field
from typing import List, Optional, Any
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

class AgentResponse(BaseModel):
    status: str
    result: str
    steps: List[Any] = []
    errors: List[str] = []

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
""",

    # ---------------------------------------------------------
    # 2. BASE AGENT ABSTRACTION
    # ---------------------------------------------------------
    "app/agents/base.py": """from abc import ABC, abstractmethod
from typing import Optional, Callable
from app.models import AgentResponse

class BaseAgent(ABC):
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    async def run(self, task: str, max_steps: int = 20, callback: Optional[Callable] = None) -> AgentResponse:
        pass
""",

    # ---------------------------------------------------------
    # 3. BROWSER LOGIC (Integrated Wrapper)
    # ---------------------------------------------------------
    "app/browser.py": """import json
import logging
import asyncio
import base64
import os
import sys
import subprocess
import tempfile
import traceback
from typing import List, Optional, Any, Callable

# Force Windows Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)

# --- IMPORT CHECK BLOCK ---
try:
    import aiohttp
    import langchain
    from langchain_core.messages import HumanMessage
    from browser_use import Agent as BUAgent
    from browser_use import Browser, BrowserSession
    HAS_BROWSER_LIB = True
    IMPORT_ERROR = None
except ImportError as e:
    HAS_BROWSER_LIB = False
    IMPORT_ERROR = f"{e}\\n{traceback.format_exc()}"
    class Browser: pass
    class BrowserSession: pass

from app.models import SubTask, AgentResponse
from app.agents.base import BaseAgent

COOKIE_PATH = "./tmp/cookies.json"

async def start_chrome_with_debug_port(port: int = 9222, headless: bool = False):
    \"\"\"Start Chrome with remote debugging enabled.\"\"\"
    user_data_dir = tempfile.mkdtemp(prefix='chrome_cdp_')
    chrome_paths = [
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/usr/bin/google-chrome', 'chrome', 'chromium', 
        r'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        r'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "msedge"
    ]
    chrome_exe = next((p for p in chrome_paths if os.path.exists(p) or p in ['chrome', 'msedge']), 'chrome')

    cmd = [chrome_exe, f'--remote-debugging-port={port}', f'--user-data-dir={user_data_dir}',
           '--no-first-run', '--no-default-browser-check', '--disable-extensions',
           '--disable-popup-blocking', '--disable-blink-features=AutomationControlled']
    if headless: cmd.append('--headless=new')
    else: cmd.append('about:blank')

    process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    for i in range(20):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{port}/json/version', timeout=1) as r:
                    if r.status == 200: return process
        except: pass
        await asyncio.sleep(1)
    raise RuntimeError('Browser CDP failed to start')

class BrowserAgentWrapper(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.browser = None
        self.context = None
        self.chrome_process = None

    async def start_session(self, headless: bool = False):
        if not HAS_BROWSER_LIB: return
        self.chrome_process = await start_chrome_with_debug_port(9222, headless)
        self.browser = Browser(cdp_url="http://localhost:9222", headless=headless)
        self.context = await self.browser.new_context()
        if os.path.exists(COOKIE_PATH):
            try:
                with open(COOKIE_PATH, 'r') as f:
                    cookies = json.load(f)
                await self.context.add_cookies(cookies)
            except: pass

    async def close_session(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.chrome_process:
            try:
                self.chrome_process.terminate()
                await self.chrome_process.wait()
            except: pass
        self.chrome_process = self.browser = self.context = None

    async def decompose_task(self, task: str, class_name: str):
        if not HAS_BROWSER_LIB: return [SubTask(description=f"Analyze {task}")]
        prompt = f"Role: Research Assistant. Context: {class_name}. Task: {task}. Break into 3 raw JSON string steps."
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)]).content
            start, end = res.find('['), res.rfind(']') + 1
            return [SubTask(description=s) for s in json.loads(res[start:end])]
        except: return [SubTask(description=f"Research: {task}")]

    async def run(self, task: str, max_steps: int = 20, callback: Optional[Callable] = None) -> AgentResponse:
        \"\"\"Implemented from user snippet logic\"\"\"
        if not HAS_BROWSER_LIB:
            return await self._run_mock(task, max_steps, callback)

        try:
            # Note: start_session is usually called by the orchestrator before individual steps,
            # but we ensure it exists here.
            if not self.context:
                await self.start_session(headless=False)

            agent = BUAgent(task=task, llm=self.llm, browser_context=self.context)
            
            if callback: await callback("log", "Agent starting execution...")
            
            # Monitoring loop integrated into run logic
            run_task = asyncio.create_task(agent.run(max_steps=max_steps))
            while not run_task.done():
                try:
                    page = await self.context.get_current_page()
                    if page and callback:
                        shot = await page.screenshot(type='jpeg', quality=50)
                        await callback("browser_stream", {"image": base64.b64encode(shot).decode('utf-8'), "url": page.url})
                except: pass
                await asyncio.sleep(0.5)
            
            history = await run_task
            return AgentResponse(status="success", result=str(history.final_result()))

        except Exception as e:
            logger.error(f"Browser agent failed: {e}")
            return AgentResponse(status="failed", result=str(e), errors=[str(e)])

    async def _run_mock(self, task: str, max_steps: int, callback: Optional[Callable]) -> AgentResponse:
        if callback:
            await callback("log", "WARNING: 'browser-use' not found. Running simulation.")
            steps = ["Navigating...", f"Searching '{task}'", "Analyzing results...", "Found info."]
            for i, s in enumerate(steps):
                await callback("step", {"step": i+1, "action": s})
                await asyncio.sleep(1)
        return AgentResponse(status="success", result=f"Simulated: {task}")

    async def run_step(self, description: str, class_name: str, headless: bool = False, quality: int = 50, callback=None) -> bool:
        \"\"\"Helper for legacy orchestrator flow\"\"\"
        resp = await self.run(description, max_steps=10, callback=callback)
        return resp.status == "success"
""",

    # ---------------------------------------------------------
    # 4. AUTH LOGIC
    # ---------------------------------------------------------
    "app/auth.py": """from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

DEFAULT_CLIENT_ID = "566039560183-8iajns5sv7799tn4k2j6h9rsht19bjkp.apps.googleusercontent.com"
DEFAULT_CLIENT_SECRET = "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"

SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid',
    'https://www.googleapis.com/auth/generative-language.peruserquota'
]

@router.get("/login_page")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/auth/login")
async def login(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        redirect_uri = str(request.url_for('auth_callback'))
        client_id = os.getenv("GOOGLE_CLIENT_ID", DEFAULT_CLIENT_ID)
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
        
        client_config = {"web": {"client_id": client_id, "client_secret": client_secret,
                                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                 "token_uri": "https://oauth2.googleapis.com/token"}}
        flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
        auth_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
        request.session['state'] = state
        return RedirectResponse(auth_url)
    except: return HTMLResponse("Auth Init Error")

@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        state = request.session.get('state')
        redirect_uri = str(request.url_for('auth_callback'))
        client_config = {"web": {"client_id": os.getenv("GOOGLE_CLIENT_ID", DEFAULT_CLIENT_ID),
                                 "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", DEFAULT_CLIENT_SECRET),
                                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                 "token_uri": "https://oauth2.googleapis.com/token"}}
        flow = Flow.from_client_config(client_config, scopes=SCOPES, state=state, redirect_uri=redirect_uri)
        flow.fetch_token(authorization_response=str(request.url))
        request.session['google_creds'] = {'token': flow.credentials.token, 'scopes': flow.credentials.scopes}
        return RedirectResponse("/")
    except Exception as e: return HTMLResponse(f"Auth Error: {e}")

@router.get("/auth/status")
async def auth_status(request: Request):
    return {"is_logged_in": "google_creds" in request.session}

@router.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
""",

    # ---------------------------------------------------------
    # 5. BACKEND SERVER
    # ---------------------------------------------------------
    "app/backend.py": """import sys
import asyncio

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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.browser import BrowserAgentWrapper
from app.models import TaskStatus, SubTask, AgentTaskRequest, AgentState, UserFeedback
from app.auth import router as auth_router

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "17.1.2"
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
def get_settings(): return Settings()

settings = get_settings()
os.environ.setdefault("GOOGLE_CLIENT_ID", settings.GOOGLE_CLIENT_ID)
os.environ.setdefault("GOOGLE_CLIENT_SECRET", settings.GOOGLE_CLIENT_SECRET)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def create_llm(model_override: str = None):
    p, m = settings.LLM_PROVIDER.lower(), model_override or settings.MODEL_NAME
    if m.startswith("gemini") or p == "gemini":
        return ChatGoogleGenerativeAI(model=m if m.startswith("gemini") else "gemini-flash-latest", google_api_key=settings.GEMINI_API_KEY)
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=m)

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

    def set_callback(self, cb): self.global_callback = cb

    async def _worker(self):
        while True:
            tid, req = await self.queue.get()
            try: await self._process_lifecycle(tid, req)
            except Exception as e: logging.error(f"Task {tid} failed: {e}")
            finally: self.queue.task_done()

    def add_to_queue(self, request: AgentTaskRequest) -> str:
        tid = str(uuid.uuid4())
        self.tasks[tid] = AgentState(task_id=tid, task_input=request.task, class_name=request.class_name)
        self.events[tid] = asyncio.Event()
        self.queue.put_nowait((tid, request))
        if self.global_callback: asyncio.create_task(self.global_callback("queue_update", [t.model_dump() for t in self.tasks.values()]))
        return tid

    async def _process_lifecycle(self, tid, req):
        state, cb = self.tasks[tid], self.global_callback
        try: llm = create_llm(req.model_override)
        except Exception as e:
            state.status = TaskStatus.FAILED
            if cb: await cb("log", f"LLM Error: {e}")
            return

        agent = BrowserAgentWrapper(llm)
        state.status = TaskStatus.PLANNING
        if cb: await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
        state.plan = await agent.decompose_task(req.task, req.class_name)
        if cb: await cb("plan", {"task_id": tid, "plan": [p.model_dump() for p in state.plan]})

        if req.require_confirmation:
            state.status = TaskStatus.WAITING_FOR_USER
            if cb: await cb("status", {"task_id": tid, "status": "waiting_for_user"})
            await self.events[tid].wait()
            if state.status == TaskStatus.STOPPED: return

        state.status = TaskStatus.RUNNING
        if cb: await cb("queue_update", [t.model_dump() for t in self.tasks.values()])

        try:
            await agent.start_session(headless=req.headless)
            for sub in state.plan:
                sub.status = "active"
                if cb: await cb("plan_update", sub.model_dump())
                success = await agent.run_step(sub.description, req.class_name, req.headless, callback=cb)
                sub.status = "completed" if success else "failed"
                if cb: await cb("plan_update", sub.model_dump())
        finally: await agent.close_session()

        state.status = TaskStatus.COMPLETED
        if cb: await cb("queue_update", [t.model_dump() for t in self.tasks.values()])

    def handle_feedback(self, fb: UserFeedback):
        if s := self.tasks.get(fb.task_id):
            if fb.approved:
                if fb.modified_plan: s.plan = fb.modified_plan
                s.status = TaskStatus.RUNNING
            else: s.status = TaskStatus.STOPPED
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
    try: return requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2).json()
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
    static_p = os.path.join(settings.BASE_DIR, "static")
    os.makedirs(static_p, exist_ok=True)
    app.mount("/", StaticFiles(directory=static_p, html=True), name="static")
    return app
""",

    # ---------------------------------------------------------
    # 6. TEMPLATES & STATIC
    # ---------------------------------------------------------
    "templates/login.html": """<!DOCTYPE html><html><head><title>Login</title><style>body{font-family:sans-serif;background:#0b0d11;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.card{background:#151921;padding:2rem;border:1px solid #2d3748;border-radius:8px;text-align:center}a{display:inline-block;background:#3b82f6;color:white;padding:10px 20px;text-decoration:none;border-radius:6px;margin-top:10px}</style></head><body><div class="card"><h2>Scholar Agent Login</h2><a href="/auth/login">Login with Google</a></div></body></html>""",
    "app/static/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Scholar Pro</title>
    <style>
        :root { --bg:#0b0d11; --pnl:#151921; --acc:#3b82f6; --txt:#e0e6ed; --border:#2d3748; }
        body { background: var(--bg); color: var(--txt); font-family: sans-serif; margin: 0; display: grid; grid-template-columns: 280px 1fr 300px; height: 100vh; overflow: hidden; }
        aside, main { padding: 1.2rem; border-right: 1px solid var(--border); overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
        input, select, button { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid var(--border); background: #1c232d; color: #fff; }
        button { background: var(--acc); font-weight: bold; cursor: pointer; }
        .viewport { flex: 1; background: #000; border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .log-box { height: 160px; background: #050505; padding: 10px; font-family: monospace; font-size: 12px; overflow-y: auto; }
        .item { padding: 8px; background: #1c232d; border-radius: 6px; margin-bottom: 5px; font-size: 12px; display: flex; gap: 10px; align-items: center; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #4a5568; }
        .active .dot { background: var(--acc); box-shadow: 0 0 8px var(--acc); }
        .modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: none; align-items: center; justify-content: center; }
    </style>
</head>
<body>
    <aside>
        <h2>Scholar Pro</h2>
        <div id="auth-status" style="font-size: 12px;">Checking...</div>
        <label>Task</label><input id="task-input" placeholder="Search for...">
        <button id="run-btn">EXECUTE</button>
        <div id="queue-box" style="margin-top: 20px;"></div>
    </aside>
    <main>
        <div class="viewport"><img id="live-stream" style="max-width: 100%; max-height: 100%;"></div>
        <div class="log-box" id="log-box"></div>
    </main>
    <aside id="right"><h3>Plan</h3><div id="plan-box"></div></aside>
    <div class="modal" id="modal"><div style="background: #1e212b; padding: 20px; border-radius: 8px; border: 1px solid var(--acc);"><h3>Approve Plan?</h3><button onclick="feedback(true)">Approve</button><button onclick="feedback(false)" style="background: red;">Cancel</button></div></div>
    <script>
        let ws, taskId, planState = [];
        async function checkAuth() {
            const r = await fetch('/auth/status'); const d = await r.json();
            document.getElementById('auth-status').innerText = d.is_logged_in ? "Logged In" : "Guest";
        }
        async function init() {
            checkAuth(); ws = new WebSocket(`ws://${location.host}/api/v1/ws/agent`);
            ws.onmessage = e => handle(JSON.parse(e.data));
            document.getElementById('run-btn').onclick = async () => {
                fetch('/api/v1/agent/run', { method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task: document.getElementById('task-input').value})
                });
            };
        }
        async function feedback(approved) {
            document.getElementById('modal').style.display = 'none';
            fetch('/api/v1/agent/feedback', { method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({task_id: taskId, approved})
            });
        }
        function handle(m) {
            if(m.type === 'status' && m.content.status === 'waiting_for_user') { taskId = m.content.task_id; document.getElementById('modal').style.display = 'flex'; }
            if(m.type === 'browser_stream') document.getElementById('live-stream').src = 'data:image/jpeg;base64,' + m.content.image;
            if(m.type === 'log') document.getElementById('log-box').innerHTML += `<div>> ${m.content}</div>`;
            if(m.type === 'plan') { taskId = m.content.task_id; document.getElementById('plan-box').innerHTML = m.content.plan.map(s => `<div class="item" id="s-${s.id}"><div class="dot"></div>${s.description}</div>`).join(''); }
            if(m.type === 'plan_update') document.getElementById(`s-${m.content.id}`).className = `item ${m.content.status}`;
        }
        window.onload = init;
    </script>
</body></html>""",

    "requirements.txt": """fastapi
uvicorn[standard]
pydantic>=2.0
pydantic-settings
langchain
langchain-core
langchain-google-genai
langchain-openai
langchain-community
langchain-ollama
huggingface_hub
browser-use
tqdm
requests
python-multipart
python-dotenv
google-auth
google-auth-oauthlib
playwright
jinja2
itsdangerous
aiohttp
""",
    "run.py": """import uvicorn
import sys, os
import asyncio
if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app.backend:create_app", host="0.0.0.0", port=8000, reload=True, factory=True)
""",
    "app/__init__.py": ""
}

def create_files():
    print("--- 📂 Creating Project Files ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for filepath, content in FILES.items():
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f: f.write(content)
        print(f"✅ Created: {filepath}")

def install_dependencies():
    print("\\n--- ⬇️  Installing Dependencies ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_application():
    print("\\n--- 🚀 Launching Application ---")
    try: subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt: print("\\nStopped.")

if __name__ == "__main__":
    create_files()
    install_dependencies()
    run_application()