import asyncio
import argparse
import base64
import importlib.util
import inspect
import json
import logging
import os
import sys
import shutil
import threading
import uuid
import urllib.request
from pydantic import BaseModel
from pathlib import Path
import warnings

# Filter warnings immediately
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import BaseCallbackHandler
import uvicorn

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.agent.deep_research.deep_research_agent import DeepResearchAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.utils import llm_provider
from src.agent.deep_research.search_tool import _AGENT_STOP_FLAGS
from src.agent.deep_research.state_manager import DeepResearchStateManager
from src.utils.utils import ensure_default_extraction_models, suppress_asyncio_cleanup_errors

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow OAuth over HTTP for localhost
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"))
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")

@app.on_event("startup")
async def startup_check_ollama():
    logger.info("🚀 Server starting up...")
    suppress_asyncio_cleanup_errors()
    os.makedirs("./tmp", exist_ok=True)
    
    logger.info("📦 Ensuring default extraction models are present...")
    await asyncio.to_thread(ensure_default_extraction_models)
    logger.info("✅ Extraction models ready.")

    logger.info("Checking for Ollama instance...")
    try:
        url = "http://localhost:11434/api/tags"
        def check():
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False
        
        is_running = await asyncio.to_thread(check)
        if is_running:
            logger.info("✅ Ollama is detected and running at http://localhost:11434")
        else:
            logger.warning("⚠️ Ollama is NOT detected at http://localhost:11434. (Ensure it's running for local LLMs)")
    except Exception as e:
        logger.warning(f"⚠️ Error checking Ollama status: {e}")

    # Check Google OAuth configuration
    if os.path.exists("client_secret.json"):
        logger.info("✅ Google OAuth credentials detected (client_secret.json).")
    elif not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        logger.warning("⚠️ Google OAuth credentials (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET) not found. 'Sign in with Google' will fail.")
    else:
        logger.info("✅ Google OAuth credentials detected.")

def load_model_from_file(model_name: str):
    try:
        path = Path(f"./storage/{model_name}.py")
        if not path.exists():
            return None
        
        spec = importlib.util.spec_from_file_location(model_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                if obj.__module__ == model_name:
                    return obj
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None
        


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


def get_gemini_models():
    """
    Returns a list of available Gemini models.
    """
    return ["gemini-2.0-flash-exp", "gemini-flash-latest\", "gemini-pro-vision", "gemini-1.5-pro-latest"]  # Add other Gemini models as needed


@app.get("/api/gemini_models")
async def gemini_models():
    """
    Endpoint to return available Gemini models.
    """
    try:
        models = get_gemini_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching Gemini models: {e}")
        return {"error": "Could not retrieve Gemini models"}, 500





@app.get("/extraction_models")
async def get_extraction_models():
    models_dir = Path("./storage")
    if not models_dir.exists():
        return []
    files = [f.stem for f in models_dir.glob("*.py") if f.stem != "__init__"]
    return files

@app.get("/ollama_models")
async def get_ollama_models():
    try:
        url = "http://localhost:11434/api/tags"
        def fetch():
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode())
        
        data = await asyncio.to_thread(fetch)
        return [model['name'] for model in data.get('models', [])]
    except Exception:
        return []

@app.get("/files")

async def list_files():
    files_list = []
    for root, dirs, files in os.walk("./tmp"):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
            files_list.append(rel_path)
    return files_list

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = Path("./tmp/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(file_path)}

@app.get("/auth/login")
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

@app.get("/auth/callback")
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
        request.session['google_creds'] = {'token': creds.token, 'refresh_token': creds.refresh_token, 'token_uri': creds.token_uri, 'client_id': creds.client_id, 'client_secret': creds.client_secret, 'scopes': creds.scopes}
        return RedirectResponse("/")
    except Exception as e:
        return HTMLResponse(f"Auth Error: {e}")

@app.get("/auth/status")
async def auth_status(request: Request):
    return {"is_logged_in": "google_creds" in request.session}

@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    browser = None
    browser_context = None
    runner_task = None
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action = payload.get("action", "run")
                
                if action == "stop":
                    if runner_task and not runner_task.done():
                        runner_task.cancel()
                    continue
                    
                if action == "run":
                    task = payload.get("task")
                    agent_type = payload.get("agent_type", "browser")

                    if not task:
                        continue
                        
                    if runner_task and not runner_task.done():
                        await websocket.send_json({"type": "error", "content": "Agent already running"})
                        continue
                    
                    # Extract settings
                    llm_settings = payload.get("llm", {})
                    agent_settings = payload.get("agent", {})
                    browser_settings = payload.get("browser", {})
                    
                    # Deep Research specific
                    resume_task_id = payload.get("resume_task_id")
                    mcp_config = payload.get("mcp_config")
                    extraction_model_name = payload.get("extraction_model")
                    
                    async def run_agent_job():
                        # Local variables to ensure thread safety and cleanup per task
                        nonlocal browser, browser_context
                        local_task_id = None

                        # Streamlining: Close previous context to ensure clean slate for new task
                        if browser_context:
                            logger.info("♻️  Resetting browser context for new task...")
                            await browser_context.close()
                            browser_context = None
                        
                        try:
                            await websocket.send_json({"type": "log", "content": f"Initializing {agent_type} agent for task: {task}"})
                            
                            provider = llm_settings.get("provider", "openai")
                            if provider == "gemini":
                                try:
                                    from langchain_google_genai import ChatGoogleGenerativeAI
                                    
                                    # Check for OAuth credentials in session
                                    creds_data = websocket.session.get("google_creds")
                                    
                                    # Check for Service Account file
                                    service_account_path = "service_account.json"
                                    
                                    if os.path.exists(service_account_path):
                                        from google.oauth2 import service_account
                                        creds = service_account.Credentials.from_service_account_file(service_account_path)
                                        await websocket.send_json({"type": "log", "content": "Using Google Gemini with Service Account"})
                                        llm = ChatGoogleGenerativeAI(
                                            model=llm_settings.get("model_name", "gemini-2.0-flash-exp"),
                                            temperature=float(llm_settings.get("temperature", 0.8)),
                                            credentials=creds
                                        )
                                    elif creds_data:
                                        from google.oauth2.credentials import Credentials
                                        creds = Credentials(**creds_data)
                                        await websocket.send_json({"type": "log", "content": "Using Google Gemini with OAuth Credentials"})
                                        llm = ChatGoogleGenerativeAI(
                                            model=llm_settings.get("model_name", "gemini-2.0-flash-exp"),
                                            temperature=float(llm_settings.get("temperature", 0.8)),
                                            credentials=creds
                                        )
                                    else:
                                        # Fallback to API Key
                                        await websocket.send_json({"type": "log", "content": "Using Google Gemini with API Key"})
                                        llm = ChatGoogleGenerativeAI(
                                            model=llm_settings.get("model_name", "gemini-2.0-flash-exp"),
                                            temperature=float(llm_settings.get("temperature", 0.8)),
                                            google_api_key=llm_settings.get("api_key", "")
                                        )
                                except ImportError:
                                    await websocket.send_json({"type": "error", "content": "Please install langchain-google-genai to use Gemini."})
                                    return
                            else:
                                llm = llm_provider.get_llm_model(
                                    provider=provider,
                                    model_name=llm_settings.get("model_name", "gpt-4o"),
                                    temperature=float(llm_settings.get("temperature", 0.8)),
                                    base_url=llm_settings.get("base_url", ""),
                                    api_key=llm_settings.get("api_key", ""),
                                    project_id=llm_settings.get("google_project_id", "")
                                )

                            if agent_type == "deep_research":
                                await websocket.send_json({"type": "log", "content": "🔬 Initializing full Deep Research Agent..."})
                                
                                deep_research_agent = DeepResearchAgent(
                                    llm=llm,
                                    browser_config=browser_settings,
                                    mcp_server_config=mcp_config
                                )
                                
                                local_task_id = resume_task_id if resume_task_id else str(uuid.uuid4())
                                output_dir = os.path.join("./tmp/deep_research", local_task_id)
                                os.makedirs(output_dir, exist_ok=True)
                                
                                stop_event = threading.Event()
                                _AGENT_STOP_FLAGS[local_task_id] = stop_event
                                
                                agent_tools = await deep_research_agent._setup_tools(
                                    task_id=local_task_id,
                                    stop_event=stop_event,
                                    max_parallel_browsers=agent_settings.get("max_parallel_browsers", 1)
                                )

                                initial_state = {
                                    "task_id": local_task_id,
                                    "topic": task,
                                    "research_plan": [],
                                    "search_results": [],
                                    "messages": [],
                                    "llm": llm,
                                    "tools": agent_tools,
                                    "output_dir": Path(output_dir),
                                    "browser_config": browser_settings,
                                    "final_report": None,
                                    "current_category_index": 0,
                                    "current_task_index_in_category": 0,
                                    "stop_requested": False,
                                    "error_message": None,
                                    "memory_file": None, # Not supported in simple server yet
                                    "google_docs_template_url": payload.get("google_docs_template_url", "a new Google Doc")
                                }

                                if resume_task_id:
                                    state_manager = DeepResearchStateManager(output_dir)
                                    loaded_state = await asyncio.to_thread(state_manager.load_state)
                                    if loaded_state:
                                        initial_state.update(loaded_state)
                                        initial_state["topic"] = task

                                final_report = None
                                final_state = None
                                async for state_update in deep_research_agent.graph.astream(initial_state):
                                    node_name = list(state_update.keys())[0]
                                    node_output = state_update[node_name]
                                    final_state = node_output

                                    await websocket.send_json({"type": "log", "content": f"Executing node: {node_name}"})

                                    if node_name == "synthesize_report" and node_output.get("final_report"):
                                        final_report = node_output.get("final_report")

                                    if node_output.get("error_message"):
                                        await websocket.send_json({"type": "error", "content": node_output.get("error_message")})
                                        break

                                if final_report:
                                    await websocket.send_json({"type": "result", "content": final_report})
                                elif not (final_state and final_state.get("error_message")):
                                    await websocket.send_json({"type": "error", "content": "Deep research finished without generating a report."})

                            else:
                                # Standard Browser Agent
                                
                                # Check if we need to re-initialize browser due to settings change
                                if browser_settings and browser:
                                    await browser.close()
                                    browser = None
                                    browser_context = None

                                if browser is None:
                                    browser = CustomBrowser(
                                        config=BrowserConfig(
                                            headless=browser_settings.get("headless", False),
                                            disable_security=browser_settings.get("disable_security", True),
                                            wss_url=browser_settings.get("wss_url"),
                                            cdp_url=browser_settings.get("cdp_url"),
                                            extra_browser_args=browser_settings.get("extra_browser_args", [])
                                        )
                                    )
                                
                                if browser_context is None:
                                    browser_context = await browser.new_context(
                                        config=BrowserContextConfig(
                                            window_width=browser_settings.get("window_w", 1280),
                                            window_height=browser_settings.get("window_h", 1100),
                                            save_recording_path=browser_settings.get("save_recording_path"),
                                            trace_path=browser_settings.get("save_trace_path")
                                        )
                                    )

                                # Initialize Planner/Confirmer if configured
                                planner_llm = None
                                if agent_settings.get("planner", {}).get("enabled", False):
                                    p_conf = agent_settings.get("planner", {})
                                    planner_llm = llm_provider.get_llm_model(
                                        provider=p_conf.get("provider", llm_settings.get("provider", "openai")),
                                        model_name=p_conf.get("model_name", llm_settings.get("model_name", "gpt-4o")),
                                        temperature=p_conf.get("temperature", 0.8),
                                        base_url=p_conf.get("base_url", ""),
                                        api_key=p_conf.get("api_key", "")
                                    )
                                
                                confirmer_llm = None
                                if agent_settings.get("confirmer", {}).get("enabled", False):
                                    c_conf = agent_settings.get("confirmer", {})
                                    confirmer_llm = llm_provider.get_llm_model(
                                        provider=c_conf.get("provider", llm_settings.get("provider", "openai")),
                                        model_name=c_conf.get("model_name", llm_settings.get("model_name", "gpt-4o")),
                                        temperature=c_conf.get("temperature", 0.8),
                                        base_url=c_conf.get("base_url", ""),
                                        api_key=c_conf.get("api_key", "")
                                    )
                                
                                # --- Heuristic Model Switching Setup ---
                                enable_smart_retry = agent_settings.get("enable_smart_retry", False)
                                enable_cost_saver = agent_settings.get("enable_cost_saver", False)
                                model_priority_list_config = agent_settings.get("model_priority_list", [])
                                model_priority_list = []

                                if (enable_smart_retry or enable_cost_saver) and model_priority_list_config:
                                    await websocket.send_json({"type": "log", "content": "⚙️ Initializing model priority list for heuristic switching..."})
                                    for model_conf in model_priority_list_config:
                                        try:
                                            priority = model_conf.get("priority")
                                            if priority is None:
                                                await websocket.send_json({"type": "log", "content": f"⚠️ Skipping model in priority list due to missing 'priority': {model_conf.get('model_name')}"})
                                                continue
                                            
                                            heuristic_llm = llm_provider.get_llm_model(
                                                provider=model_conf.get("provider"),
                                                model_name=model_conf.get("model_name"),
                                                temperature=float(model_conf.get("temperature", 0.1)),
                                                base_url=model_conf.get("base_url", ""),
                                                api_key=model_conf.get("api_key", "")
                                            )
                                            model_priority_list.append({"priority": int(priority), "llm": heuristic_llm})
                                        except Exception as e:
                                            await websocket.send_json({"type": "log", "content": f"⚠️ Failed to initialize heuristic model {model_conf.get('model_name')}: {e}"})
                                    
                                    model_priority_list.sort(key=lambda x: x['priority'])
                                
                                output_model = None
                                if extraction_model_name:
                                    output_model = load_model_from_file(extraction_model_name)
                                controller = CustomController(output_model=output_model)
                                
                                async def validation_callback(think, reason, is_confirmed):
                                    await websocket.send_json({
                                        "type": "validation",
                                        "think": think,
                                        "reason": reason,
                                        "is_confirmed": is_confirmed
                                    })


                                agent = BrowserUseAgent(
                                    task=task,
                                    llm=llm,
                                    browser=browser,
                                    browser_context=browser_context,
                                    controller=controller,
                                    use_vision=agent_settings.get("use_vision", True),
                                    planner_llm=planner_llm,
                                    confirmer_llm=confirmer_llm,
                                    inhibit_close=True,
                                    enable_smart_retry=enable_smart_retry,
                                    enable_cost_saver=enable_cost_saver,
                                    model_priority_list=model_priority_list,
                                    validation_callback=validation_callback,
                                    tool_calling_method=agent_settings.get("tool_calling_method", "auto")
                                )
                                
                                async def step_callback(state, model_output, step_number):
                                    try:
                                        thought = getattr(model_output, "thought", "") if model_output else ""
                                        screenshot = state.screenshot if state.screenshot else None
                                        actions = []
                                        if model_output:
                                            output_actions = getattr(model_output, "action", None)
                                            if output_actions:
                                                if not isinstance(output_actions, list):
                                                    output_actions = [output_actions]
                                                for action in output_actions:
                                                    if hasattr(action, "model_dump"):
                                                        actions.append(action.model_dump())
                                        
                                        await websocket.send_json({
                                            "type": "step",
                                            "step": step_number,
                                            "thought": thought,
                                            "url": state.url,
                                            "screenshot": screenshot,
                                            "actions": actions
                                        })
                                    except Exception as e:
                                        logger.error(f"Error in callback: {e}")

                                agent.step_callback = step_callback
                                
                                await websocket.send_json({"type": "log", "content": "Agent started..."}) 
                                
                                stream_task = None
                                if browser_settings.get("enable_live_view", False):
                                    async def stream_browser():
                                        while True:
                                            try:
                                                if browser_context:
                                                    page = await browser_context.get_current_page()
                                                    if page:
                                                        screenshot = await page.screenshot(type='jpeg', quality=50)
                                                        encoded = base64.b64encode(screenshot).decode('utf-8')
                                                        await websocket.send_json({"type": "stream", "image": encoded})
                                            except Exception:
                                                pass
                                        await asyncio.sleep(0.5)
                                    
                                    stream_task = asyncio.create_task(stream_browser())

                                try:
                                    history = await agent.run(max_steps=agent_settings.get("max_steps", 100))
                                    result = history.final_result()
                                    await websocket.send_json({"type": "result", "content": result})
                                finally:
                                    if stream_task:
                                        stream_task.cancel()
                                        try:
                                            await stream_task
                                        except asyncio.CancelledError:                                        
                                            pass
                        
            
                        except Exception as e:
                            logger.error(f"Task execution error: {e}", exc_info=True)
                            await websocket.send_json({"type": "error", "content": str(e)})
                            if local_task_id:
                                await stop_browsers_for_task(local_task_id)
                    runner_task = asyncio.create_task(run_agent_job())
                
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                    
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")

        if runner_task and not runner_task.done():
            runner_task.cancel()
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if runner_task and not runner_task.done():
            
            runner_task.cancel()
        if browser_context:
            await browser_context.close()
        if browser:            
            await browser.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser Agent Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting FastAPI server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)