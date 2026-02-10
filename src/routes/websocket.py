import asyncio
import base64
import json
import logging
import os
import threading
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.callbacks import BaseCallbackHandler

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.agent.deep_research.deep_research_agent import DeepResearchAgent
from src.browser.custom_browser import CustomBrowser
from browser_use.browser.browser import BrowserConfig
from src.controller.custom_controller import CustomController
from src.utils.llm_manager import get_llm_model
from browser_use.browser.context import BrowserContextConfig
from src.utils import llm_provider
from src.agent.deep_research.search_tool import _AGENT_STOP_FLAGS
from src.agent.deep_research.state_manager import DeepResearchStateManager
from src.routes.models import load_model_from_file
from src.config import RATE_LIMIT_SECONDS, MAX_REQUESTS_PER_MINUTE
from src.agent.browser_use.components.cookie_manager import CookieManager
from src.utils.instruction_handler import InstructionHandler, create_instruction_handler
router = APIRouter()
logger = logging.getLogger(__name__)
COOKIE_PATH = "./tmp/cookies.json"

async def run_browser_agent(payload, websocket, browser, browser_context):
    task = payload.get("task")
    agent_settings = payload.get("agent", {})
    browser_settings = payload.get("browser", {})
    llm_settings = payload.get("llm", {})
    extraction_model_name = payload.get("extraction_model")

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
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            elif creds_data:
                from google.oauth2.credentials import Credentials
                creds = Credentials(**creds_data)
                await websocket.send_json({"type": "log", "content": "Using Google Gemini with OAuth Credentials"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            else:
                # Fallback to API Key
                await websocket.send_json({"type": "log", "content": "Using Google Gemini with API Key"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    google_api_key=llm_settings.get("api_key", "")
                )
        except ImportError as e:
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
        state=None,
        settings=agent_settings,
        task=task,
        llm=llm,
        browser_context=browser_context,
        message_manager=None,
        controller=controller,
        use_vision=agent_settings.get("use_vision", True),
        planner_llm=planner_llm,
        confirmer_llm=confirmer_llm,
        inhibit_close=True,
        enable_cost_saver=enable_cost_saver,
       cookie_path=COOKIE_PATH,
        model_priority_list=model_priority_list,
        validation_callback=validation_callback,        
        tool_calling_method=agent_settings.get("tool_calling_method", "auto")
    )
    instruction_handler = create_instruction_handler(agent)
    
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

async def run_deep_research_agent(payload, websocket):
    task = payload.get("task")
    llm_settings = payload.get("llm", {})
    agent_settings = payload.get("agent", {})
    browser_settings = payload.get("browser", {})
    resume_task_id = payload.get("resume_task_id")
    mcp_config = payload.get("mcp_config")

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
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            elif creds_data:
                from google.oauth2.credentials import Credentials
                creds = Credentials(**creds_data)
                await websocket.send_json({"type": "log", "content": "Using Google Gemini with OAuth Credentials"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    credentials=creds
                )
            else:
                # Fallback to API Key
                await websocket.send_json({"type": "log", "content": "Using Google Gemini with API Key"})
                llm = ChatGoogleGenerativeAI(
                    model=llm_settings.get("model_name", "gemini-1.5-flash-latest"),
                    temperature=float(llm_settings.get("temperature", 0.8)),
                    google_api_key=llm_settings.get("api_key", "")
                )
        except ImportError as e:
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

    await websocket.send_json({"type": "log", "content": "🔬 Initializing full Deep Research Agent..."})
    
    deep_research_agent = DeepResearchAgent(
        llm=llm,
        browser_config=browser_settings,
        mcp_server_config=mcp_config
    )
    instruction_handler = create_instruction_handler(deep_research_agent)
    
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



@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    browser = None
    browser_context = None
    runner_task = None

    # Rate limiting
    client_ip = websocket.client.host
    request_counts = defaultdict(int)
    
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

                    async def run_agent_job():
                        nonlocal browser, browser_context
                        try:
                            await websocket.send_json({"type": "log", "content": f"Initializing {agent_type} agent for task: {task}"})
                            
                            if agent_type == "deep_research":
                                await run_deep_research_agent(payload, websocket)
                            else:
                                await run_browser_agent(payload, websocket, browser, browser_context)

                        except Exception as e:
                            logger.error(f"Task execution error: {e}", exc_info=True)
                            await websocket.send_json({"type": "error", "content": str(e)})

                    runner_task = asyncio.create_task(run_agent_job())
                
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_json({"type": "error", "content": f"Error processing message: {e}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")

        if runner_task and not runner_task.done():
            runner_task.cancel()
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

        if client_ip in request_counts:
            del request_counts[client_ip]
    finally:
        if runner_task and not runner_task.cancelled():
            runner_task.cancel()
        if browser_context:
            await browser_context.close()
        if browser:
            # Clear rate limit on disconnect

            CookieManagerClass = get_cookie_manager()
            if CookieManagerClass and browser_context:
                cookie_manager = CookieManagerClass()
                await cookie_manager.save_cookies(browser_context)

            await browser.close()
