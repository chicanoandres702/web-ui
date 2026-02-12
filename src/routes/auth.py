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
""",

    # ---------------------------------------------------------
    # 2. BROWSER LOGIC
    # ---------------------------------------------------------
    "app/browser.py": """import json
import logging
import asyncio
import base64
import os
from typing import List, Optional, Any

try:
    from browser_use import Agent as BUAgent
    from browser_use.browser.browser import Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig
    from langchain.schema import HumanMessage
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    class BrowserConfig: 
        def __init__(self, **kwargs): pass
    class BrowserContextConfig: 
        def __init__(self, **kwargs): pass

from app.models import SubTask 

logger = logging.getLogger(__name__)

COOKIE_PATH = "./tmp/cookies.json"

class BrowserAgentWrapper:
    def __init__(self, llm):
        self.llm = llm

    async def decompose_task(self, task: str, class_name: str):
        if not HAS_LIBS:
            return [SubTask(description=f"Analyze {task}")]

        prompt = f"Role: Research Assistant. Break academic task '{task}' for context '{class_name}' into 3 distinct, executable web-browsing steps. Return ONLY a raw JSON list of strings."
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0: 
                raise ValueError("No JSON list found in response")
            steps = json.loads(content[start:end])
            return [SubTask(description=s) for s in steps]
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [SubTask(description=f"Research: {task}")]

    async def run_step(self, description: str, class_name: str, headless: bool = False, quality: int = 50, callback=None) -> bool:
        if not HAS_LIBS:
            if callback: await callback("log", "Browser libs missing. Skipping step.")
            return True

        browser = None
        context = None
        
        try:
            if callback: await callback("log", f"Initializing Browser (Headless: {headless})...")
            
            browser_config = BrowserConfig(
                headless=headless,
                disable_security=True,
                extra_browser_args=["--disable-gpu", "--no-sandbox", "--disable-blink-features=AutomationControlled"] 
            )
            browser = Browser(config=browser_config)

            context_config = BrowserContextConfig(
                browser_window_size={'width': 1280, 'height': 1100}
            )
            
            try:
                context = await browser.new_context(config=context_config)
            except TypeError:
                context = await browser.new_context()

            if os.path.exists(COOKIE_PATH):
                try:
                    with open(COOKIE_PATH, 'r') as f:
                        cookies = json.load(f)
                    await context.context.add_cookies(cookies)
                    if callback: await callback("log", "Cookies loaded into browser context.")
                except Exception as e:
                    logger.warning(f"Failed to load cookies: {e}")

            agent = BUAgent(
                task=f"Context: {class_name}. Instruction: {description}", 
                llm=self.llm, 
                browser_context=context
            )
            
            run_task = asyncio.create_task(agent.run(max_steps=10))
            
            while not run_task.done():
                try:
                    page = await context.get_current_page()
                    if page:
                        screenshot = await page.screenshot(type='jpeg', quality=quality)
                        encoded = base64.b64encode(screenshot).decode('utf-8')
                        if callback:
                            await callback("browser_stream", {
                                "image": encoded,
                                "url": page.url,
                                "title": await page.title()
                            })
                except Exception:
                    pass 
                await asyncio.sleep(0.5) 
            
            await run_task
            return True

        except Exception as e:
            logger.error(f"Browser Execution Error: {e}")
            if callback: await callback("log", f"Browser Error: {e}")
            return False
        finally:
            if context: await context.close()
            if browser: await browser.close()
""",

    # ---------------------------------------------------------
    # 3. AUTH LOGIC (VERBATIM IMPLEMENTATION)
    # ---------------------------------------------------------
    "app/auth.py": """from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse

# from app import authentication_service, user_service # Commented out: Modules do not exist in this environment
from fastapi.templating import Jinja2Templates
from typing import Optional
import os
router = APIRouter()

templates = Jinja2Templates(directory="templates")

@router.get("/login_page")
async def login_page(request: Request):
    \"\"\"Serves a simple login page with a button.\"\"\"
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/logout_page")
async def logout_page(request: Request):
    \"\"\"Serves a simple logout page with a button.\"\"\"
    return templates.TemplateResponse("logout.html", {"request": request})

@router.get("/auth/login")
async def login(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email', 'openid', 'https://www.googleapis.com/auth/generative-language.peruserquota']
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=scopes,
                redirect_uri=redirect_uri
            )
        else:
            client_config = {
                "web": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            if not client_config["web"]["client_id"] or not client_config["web"]["client_secret"]:
                return HTMLResponse("Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars not set, and client_secret.json not found.")

            flow = Flow.from_client_config(
                client_config,
                scopes=scopes,
                redirect_uri=redirect_uri
            )
            
        authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
        request.session['state'] = state
        return RedirectResponse(authorization_url)
    except ImportError:
        return HTMLResponse("Error: google-auth-oauthlib not installed.")

@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        from google_auth_oauthlib.flow import Flow
        state = request.session.get('state')
        scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/userinfo.email', 'openid', 'https://www.googleapis.com/auth/generative-language.peruserquota']
        redirect_uri = str(request.url_for('auth_callback'))

        if os.path.exists("client_secret.json"):
            flow = Flow.from_client_secrets_file(
                "client_secret.json",
                scopes=scopes,
                state=state,
                redirect_uri=redirect_uri
            )
        else:
            client_config = {
                "web": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            flow = Flow.from_client_config(
                client_config,
                scopes=scopes,
                state=state,
                redirect_uri=redirect_uri
            )
            
        flow.fetch_token(authorization_response=str(request.url))
        creds = flow.credentials
        request.session['google_creds'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': getattr(creds, 'token_uri', "https://oauth2.googleapis.com/token"),
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        return RedirectResponse("/")
    except Exception as e:
        return HTMLResponse(f"Auth Error: {e}")

@router.get("/auth/status")
async def auth_status(request: Request):
    return {"is_logged_in": "google_creds" in request.session}

@router.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

@router.get("/metrics")
async def metrics():
    \"\"\"Endpoint to expose Prometheus metrics.\"\"\"
    return Response(content="Metrics not implemented", media_type="text/plain")
""",

    # ---------------------------------------------------------
    # 4. BACKEND SERVER (Modified to include Auth)
    # ---------------------------------------------------------
    "app/backend.py": """import os
import uuid
import json
import logging
import asyncio
from functools import lru_cache
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict
import requests

# --- DEPENDENCIES (LangChain) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# --- LOCAL IMPORTS ---
from app.browser import BrowserAgentWrapper
from app.models import (
    TaskStatus, SubTask, AgentTaskRequest, AgentState, UserFeedback
)
# Import the new auth router
from app.auth import router as auth_router

# ==========================================
#               CONFIGURATION
# ==========================================

class Settings(BaseSettings):
    APP_NAME: str = "Scholar Agent Pro"
    VERSION: str = "17.0.2"
    
    # Auth & Keys
    GEMINI_API_KEY: str = "" 
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    SESSION_SECRET: str = "scholar-pro-super-secret-key"
    
    # LLM Defaults
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

# ==========================================
#               LLM FACTORY
# ==========================================

def create_llm(model_override: str = None):
    provider = settings.LLM_PROVIDER.lower()
    model = model_override or settings.MODEL_NAME

    # 1. Gemini
    if (model and model.startswith("gemini")) or provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model if model and model.startswith("gemini") else "gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY
        )

    # 2. Ollama (Default)
    return ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=model)

# ==========================================
#             ORCHESTRATOR
# ==========================================

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
                if cb: await cb("log", "Task cancelled by user.")
                return

        state.status = TaskStatus.RUNNING
        if cb:
            await cb("queue_update", [t.model_dump() for t in self.tasks.values()])
            await cb("status", "running")

        for sub in state.plan:
            sub.status = "active"
            if cb: await cb("plan_update", sub.model_dump())
            
            success = await agent_wrapper.run_step(
                sub.description,
                req.class_name,
                req.headless,
                req.stream_quality,
                callback=cb
            )

            sub.status = "completed" if success else "failed"
            if cb: await cb("plan_update", sub.model_dump())

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

# ==========================================
#               API ROUTER
# ==========================================

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
async def health():
    return {"status": "ok", "version": settings.VERSION}

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
    
    # 1. Add Session Middleware (Required by app/auth.py)
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)
    
    # 2. Include Routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(auth_router) # Routes from auth.py (auth/login, etc)

    static_path = os.path.join(settings.BASE_DIR, "static")
    os.makedirs(static_path, exist_ok=True)
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    return app
""",

    # ---------------------------------------------------------
    # 5. TEMPLATES (Required by auth.py)
    # ---------------------------------------------------------
    "templates/login.html": """<!DOCTYPE html>
<html>
<head><title>Login</title><link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
<style>body{font-family:'Inter',sans-serif;background:#0b0d11;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#151921;padding:2rem;border:1px solid #2d3748;border-radius:8px;text-align:center}
a{display:inline-block;background:#3b82f6;color:white;padding:10px 20px;text-decoration:none;border-radius:6px;margin-top:10px}
</style></head>
<body><div class="card"><h2>Scholar Agent Login</h2><p>Access your research dashboard</p><a href="/auth/login">Login with Google</a></div></body></html>""",

    "templates/logout.html": """<!DOCTYPE html>
<html>
<head><title>Logout</title><link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
<style>body{font-family:'Inter',sans-serif;background:#0b0d11;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#151921;padding:2rem;border:1px solid #2d3748;border-radius:8px;text-align:center}
a{display:inline-block;background:#3b82f6;color:white;padding:10px 20px;text-decoration:none;border-radius:6px;margin-top:10px}
</style></head>
<body><div class="card"><h2>Logged Out</h2><p>You have been successfully logged out.</p><a href="/login_page">Log In Again</a></div></body></html>""",

    # ---------------------------------------------------------
    # 6. DEPENDENCIES & RUNNER
    # ---------------------------------------------------------
    "app/static/index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scholar Pro | Command Center</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root { --bg:#0b0d11; --pnl:#151921; --acc:#3b82f6; --txt:#e0e6ed; --input:#1c232d; --border:#2d3748; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; color: var(--txt); }
        body { background: var(--bg); height: 100vh; overflow: hidden; }
        .shell { display: grid; grid-template-columns: 280px 1fr 300px; height: 100vh; }
        aside { background: var(--pnl); padding: 1.2rem; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; }
        aside#right { border-right: none; border-left: 1px solid var(--border); }
        main { padding: 1.2rem; display: flex; flex-direction: column; gap: 1rem; overflow: hidden; position: relative; }
        input, select { width: 100%; background: var(--input); border: 1px solid var(--border); color: #fff; padding: 10px; border-radius: 6px; font-size: 13px; outline: none; }
        button { background: var(--acc); color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }
        .viewport { flex: 1; background: #000; border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; position: relative; }
        .log-box { height: 160px; background: #050505; border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a0aec0; }
        .item { background: var(--input); padding: 10px; border-radius: 8px; margin-bottom: 8px; font-size: 11px; display: flex; align-items: center; gap: 10px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #4a5568; }
        .active .dot { background: var(--acc); box-shadow: 0 0 8px var(--acc); }
        .modal-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); display:none; align-items:center; justify-content:center; z-index: 1000; }
        .modal { background: #1e212b; padding: 30px; border-radius: 12px; width: 400px; border: 1px solid var(--acc); }
    </style>
</head>
<body>
    <div class="shell">
        <aside id="left">
            <h1 style="font-size:1.1rem; margin:0 0 0.5rem 0;">Scholar Pro</h1>
            <div style="font-size:11px; color:#a0aec0; margin-bottom:10px;">
                <span id="auth-status">Checking auth...</span> 
                <a href="/login_page" style="color:var(--acc); margin-left:5px;">Login</a>
                <a href="/auth/logout" style="color:#ef4444; margin-left:5px;">Logout</a>
            </div>
            <label>Context</label><input id="class-name" value="History 101">
            <label>Mode</label><select id="provider-mode"><option value="ollama">Ollama (Local)</option><option value="gemini">Gemini (Cloud)</option></select>
            <label>Model</label><select id="model-select"><option>Loading...</option></select>
            <label style="display:flex; align-items:center; gap:8px; margin-top:5px;"><input type="checkbox" id="headless" style="width:auto;"> Headless Mode</label>
            <div style="margin-top:auto; border-top:1px solid var(--border); padding-top:1rem;"><h3>QUEUE</h3><div id="queue-box"></div></div>
        </aside>
        <main>
            <div style="display:flex; gap:10px;"><input id="task-input" placeholder="Enter research objective..." style="margin:0;"><button id="run-btn" style="width:100px;">EXECUTE</button></div>
            <div class="viewport">
                <div style="background: #1a202c; padding: 8px 12px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid var(--border);">
                    <div style="flex: 1; background: #000; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #a0aec0;" id="url-bar">about:blank</div>
                </div>
                <div style="flex:1; display:flex; align-items:center; justify-content:center; background:#111;"><img id="live-stream" style="max-width:100%; max-height:100%; object-fit:contain;"></div>
            </div>
            <div class="log-box" id="log-box"></div>
        </main>
        <aside id="right">
            <h3>PLAN</h3><div id="plan-box" style="flex:1; overflow-y:auto; margin-top:10px;"></div>
        </aside>
    </div>
    <div class="modal-overlay" id="user-modal"><div class="modal"><h2>Approve Workflow</h2><p style="opacity:0.7; margin-bottom:20px;">The agent has generated a task sequence.</p><div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;"><button id="ok-btn" style="background:#10b981;">APPROVE</button><button id="no-btn" style="background:#ef4444;">CANCEL</button></div></div></div>
    <script>
        let ws, taskId, planState = [];
        async function checkAuth() {
            try {
                const r = await fetch('/auth/status');
                const d = await r.json();
                const el = document.getElementById('auth-status');
                if(d.is_logged_in) { el.innerText = "Authenticated"; el.style.color = "#10b981"; document.getElementById('provider-mode').value = 'gemini'; }
                else { el.innerText = "Not Logged In"; el.style.color = "#ef4444"; }
                refreshModelList();
            } catch(e) {}
        }
        async function refreshModelList() {
            const mode = document.getElementById('provider-mode').value;
            const select = document.getElementById('model-select');
            select.innerHTML = '<option>Loading...</option>';
            if (mode === 'ollama') {
                try {
                    const r = await fetch('/api/v1/ollama/models');
                    const d = await r.json();
                    select.innerHTML = (d.models || []).map(m => `<option value="${m.name}">${m.name}</option>`).join('');
                } catch(e) { select.innerHTML = '<option>Ollama Offline</option>'; }
            } else {
                select.innerHTML = '<option value="gemini-1.5-flash">gemini-1.5-flash</option>';
            }
        }
        async function init() {
            checkAuth();
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
            if(m.type === 'status' && m.content.status === 'waiting_for_user') { 
                taskId = m.content.task_id; document.getElementById('user-modal').style.display = 'flex'; 
            }
            if(m.type === 'browser_stream') {
                document.getElementById('live-stream').src = 'data:image/jpeg;base64,' + m.content.image;
                document.getElementById('url-bar').innerText = m.content.url;
            }
            if(m.type === 'log') { box.innerHTML += `<div>> ${m.content}</div>`; box.scrollTop = box.scrollHeight; }
            if(m.type === 'plan') {
                taskId = m.content.task_id; planState = m.content.plan;
                const b = document.getElementById('plan-box'); b.innerHTML = '';
                planState.forEach(s => b.innerHTML += `<div class="item" id="s-${s.id}"><div class="dot"></div><div style="font-size:11px;">${s.description}</div></div>`);
            }
            if(m.type === 'plan_update') { const el = document.getElementById(`s-${m.content.id}`); if(el) el.className = `item ${m.content.status}`; }
            if(m.type === 'queue_update') {
                const qb = document.getElementById('queue-box'); qb.innerHTML = '';
                m.content.forEach(t => qb.innerHTML += `<div class="item ${t.status}"><div class="dot"></div><div>${t.task_input.substring(0,18)}...</div></div>`);
            }
        }
        window.onload = init;
    </script>
</body>
</html>
""",

    "requirements.txt": """fastapi
uvicorn[standard]
pydantic>=2.0
pydantic-settings
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
""",
    "run.py": """import uvicorn
import sys, os
import asyncio

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    print("🚀 Starting Scholar Agent Pro...")
    print("👉 Open http://localhost:8000/login_page in your browser")
    uvicorn.run("app.backend:create_app", host="0.0.0.0", port=8000, reload=True, factory=True)
""",
    "app/__init__.py": ""
}

def create_files():
    """Writes all the files defined in FILES to the disk."""
    print("--- 📂 Creating Project Files ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filepath, content in FILES.items():
        full_path = os.path.join(base_dir, filepath)
        directory = os.path.dirname(full_path)
        
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Created: {filepath}")

def install_dependencies():
    """Installs pip packages and Playwright browsers."""
    print("\\n--- ⬇️  Installing Dependencies ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Installing Playwright browsers...")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

def run_application():
    """Runs the uvicorn server via the run.py script."""
    print("\\n--- 🚀 Launching Application ---")
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("\\nStopped.")

if __name__ == "__main__":
    create_files()
    install_dependencies()
    run_application()