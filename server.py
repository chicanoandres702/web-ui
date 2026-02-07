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
import uvicorn

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.agent.deep_research.deep_research_agent import DeepResearchAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.utils import llm_provider
from src.utils.browser_factory import create_browser, create_context
from src.agent.deep_research.search_tool import stop_browsers_for_task, _AGENT_STOP_FLAGS
from src.agent.deep_research.state_manager import DeepResearchStateManager
from src.utils.utils import ensure_default_extraction_models, suppress_asyncio_cleanup_errors

load_dotenv(override=True) # Added override=True to ensure .env takes precedence

KB_DIR = "./tmp/knowledge_base" # Define a directory for the general knowledge base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow OAuth over HTTP for localhost
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"))
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/tmp", StaticFiles(directory="./tmp"), name="tmp")
os.makedirs(KB_DIR, exist_ok=True) # Ensure the knowledge base directory exists

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

# Simple HTML client for testing the WebSocket connection
""" # The HTML content is moved to static/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Use Agent</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: #111827;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent: #3b82f6;
            --border: #374151;
            --success: #10b981;
            --error: #ef4444;
        }
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; height: 100vh; display: flex; overflow: hidden; }
        .sidebar { width: 350px; background-color: var(--panel-bg); border-right: 1px solid var(--border); padding: 20px; display: flex; flex-direction: column; gap: 15px; overflow-y: auto; }
        .main { flex: 1; display: flex; flex-direction: column; padding: 20px; gap: 20px; overflow: hidden; }
        .header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        h1 { margin: 0; font-size: 1.2rem; color: var(--accent); }
        h3 { margin: 0 0 10px 0; font-size: 0.9rem; color: var(--text-secondary); text-transform: uppercase; }
        
        /* Form Elements */
        label { display: block; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 5px; }
        input, select, textarea { width: 100%; background: var(--bg-color); border: 1px solid var(--border); color: var(--text-primary); padding: 8px; border-radius: 6px; box-sizing: border-box; font-family: inherit; }
        input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
        
        button { width: 100%; padding: 10px; background: var(--accent); color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; transition: 0.2s; }
        button:hover { opacity: 0.9; }
        button:disabled { background: var(--border); cursor: not-allowed; }
        button.secondary { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); }
        button.secondary:hover { border-color: var(--text-primary); color: var(--text-primary); }
        #stopBtn { background: var(--error); }
        
        /* Chat */
        #messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; padding-right: 10px; }
        .message { background: var(--panel-bg); padding: 15px; border-radius: 8px; border: 1px solid var(--border); }
        .message.user { border-left: 3px solid var(--accent); }
        .message.agent { border-left: 3px solid var(--success); }
        .message.error { border-left: 3px solid var(--error); background: rgba(239, 68, 68, 0.1); }
        .step-header { font-weight: bold; margin-bottom: 8px; display: flex; justify-content: space-between; color: var(--accent); }
        .thought { color: var(--text-secondary); font-style: italic; margin-bottom: 8px; display: block; line-height: 1.4; }
        /* Agent Status Display */
        #agentStatusDisplay {
            display: none;
            font-size: 4rem;
            font-weight: bold;
            text-align: center;
            color: var(--accent);
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            pointer-events: none; /* Allow clicks through */
        }

        /* Live Monitor */
        #liveMonitor {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        #liveMonitor div { margin-bottom: 5px; }
        #liveMonitor .monitor-label { font-weight: bold; color: var(--accent); margin-right: 5px; }
        #liveMonitor .monitor-value { color: var(--text-primary); }
        #liveMonitor #monitorGoal { font-size: 1.1rem; color: var(--success); }
        #liveMonitor #monitorState { font-size: 1.1rem; color: var(--accent); }


        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        .url { background: rgba(59, 130, 246, 0.1); color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-family: monospace; display: inline-block; }
        .screenshot { max-width: 100%; border-radius: 6px; margin-top: 10px; border: 1px solid var(--border); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

        /* General grouping for sidebar and control panel */
        .control-group {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            marg
            in-bottom: 15px; /* Spacing between groups */
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .control-group h3 {
            margin-top: 0;
            margin-bottom: 5px;
            font-size: 1rem;
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            padding-bottom: 5px;
        }
        .control-group.primary-actions {
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .control-group.agent-interaction,
        .control-group.agent-tuning,
        .control-group.specialized-tools {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); /* Responsive grid for buttons */
            gap: 10px;
        }
        .control-panel {
            margin-top: 20px; /* Space above the control panel */
        }
    </style>
</head>
<body>
    <!-- Confirmation Dialog -->
    <div id="confirmationOverlay" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.7); z-index: 1000; display: flex; justify-content: center; align-items: center;">
        <div id="confirmationDialog" style="background: var(--panel-bg); padding: 30px; border-radius: 10px; border: 1px solid var(--border); box-shadow: 0 5px 15px rgba(0,0,0,0.3); width: 500px; max-width: 90%;">
            <h2 style="margin-top: 0; color: var(--text-primary); text-align: center;">Action Required</h2>
            <p id="confirmationMessage" style="color: var(--text-secondary); margin-bottom: 15px; font-size: 1.1rem; text-align: center;">Are you sure you want to proceed with this action?</p>
            <div style="margin-bottom: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                <div style="margin-bottom: 5px;"><span style="font-weight: bold; color: var(--accent);">[INTEL]:</span> <span id="dialogIntel"></span></div>
                <div><span style="font-weight: bold; color: var(--accent);">[NEXT TASK]:</span> <span id="dialogNextTask"></span></div>
            </div>
            <input type="text" id="customTaskInput" placeholder="Type a new custom task here (optional)" style="width: calc(100% - 20px); margin-bottom: 15px; padding: 10px; font-size: 0.9rem;" />
            <div style="display: flex; justify-content: center; gap: 10px;">
                <button id="confirmYesBtn" style="width: 120px; background: var(--success);">Yes</button>
                <button id="confirmNoBtn" style="width: 120px; background: var(--error);">No</button>
            </div>
        </div>
    </div>

    <div class="sidebar">
        <div class="header">
            <h1>Agent Configuration</h1>
        </div>
        
        <div class="control-group">
            <h3>Agent & Task Setup</h3>
            <div>
                <label>Agent Type</label>
                <select id="agentType">
                    <option value="browser">Browser Agent</option>
                    <option value="deep_research">Deep Research Agent</option>
                </select>
            </div>
            <div>
                <label>Extraction Model</label>
                <select id="extractionModel">
                    <option value="">No Extraction Model</option>
                </select>
            </div>
            <div>
                <label>Resume Task ID</label>
                <input type="text" id="resumeTaskId" placeholder="Optional UUID" />
            </div>
            <div>
                <label>Google Docs Template</label>
                <input type="text" id="googleDocsTemplate" placeholder="Optional URL" />
            </div>
        </div>
        
        <div class="control-group">
            <h3>LLM Configuration</h3>
            <div>
                <label>LLM Provider</label>
                <select id="llmProvider" onchange="updateLLMSettings()">
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini (Google)</option>
                    <option value="vertex">Vertex AI</option>
                    <option value="ollama">Ollama</option>
                    <option value="anthropic">Anthropic</option>
                </select>
            </div>
            <div id="googleLoginPanel" style="display: none; padding: 10px; background: rgba(66, 133, 244, 0.1); border-radius: 4px;">
                <div id="loginStatus" style="font-size: 0.8rem; margin-bottom: 5px; color: var(--text-secondary);">Not logged in</div>
                <a id="loginBtn" href="/auth/login" style="text-decoration: none;">
                    <button type="button" style="background: #4285F4; border: none; color: white;">Sign in with Google</button>
                </a>
                <a id="logoutBtn" href="/auth/logout" style="text-decoration: none; display: none;">
                    <button type="button" style="background: var(--border); border: none; color: var(--text-primary);">Sign out</button>
                </a>
            </div>
            <div id="apiKeyField">
                <label>API Key</label>
                <input type="password" id="apiKey" placeholder="Enter API Key..." onchange="updateLLMSettings()" />
            </div>
            <div id="googleProjectIdField" style="display: none;">
                <label>Google Project ID</label>
                <input type="text" id="googleProjectId" placeholder="Enter Google Project ID..." onchange="updateLLMSettings()" />
            </div>
            <div id="baseUrlField" style="display: none;">
                <label>Base URL</label>
                <input type="text" id="baseUrl" placeholder="http://localhost:11434" value="http://localhost:11434" onchange="updateLLMSettings()" />
            </div>
            <div>
                <label>Model Name</label>
                <input type="text" id="modelName" placeholder="gemini-flash-latest" onchange="updateLLMSettings()" />
            </div>
            <div>
                <label>Quick Load Ollama Model</label>
                <select id="ollamaModel" onchange="updateOllamaSettings()">
                    <option value="">Select a model...</option>
                </select>
            </div>
        </div>
        
        <div class="control-group">
            <h3>Agent Behavior</h3>
            <div style="display: flex; align-items: center; gap: 5px;">
                <input type="checkbox" id="useVision" style="width: auto;" onchange="updateAgentSettings()" checked />
                <label for="useVision" style="margin: 0; cursor: pointer;">Use Vision</label>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <input type="checkbox" id="showConfirmerReasoning" style="width: auto;" onchange="toggleConfirmerReasoning()" />
                <label for="showConfirmerReasoning" style="margin: 0; cursor: pointer;">Show Confirmer Reasoning</label>
            </div>
            <div>
                <label>Max Consecutive Failures</label>
                <input type="number" id="maxConsecutiveFailures" value="5" onchange="updateAgentSettings()" />
            </div>
        </div>
        
        <div class="control-group">
            <label>File Upload</label>
            <div style="display: flex; gap: 5px;">
                <input type="file" id="fileInput" style="font-size: 0.8rem;" />
                <button onclick="uploadFile(event)" style="width: auto; padding: 5px 10px;">⬆️</button>
            </div>
            <div id="uploadStatus" style="font-size: 0.8rem; margin-top: 5px; color: var(--text-secondary);"></div>
        </div>
        
        <div class="control-group" style="flex: 1; display: flex; flex-direction: column;">
            <label>Advanced Settings (JSON)</label>
            <textarea id="settingsJson" style="flex: 1; font-family: monospace; font-size: 0.8rem;">{
  "llm": {
    "provider": "gemini",
    "model_name": "gemini-flash-latest",
    "temperature": 0.0
  },
  "agent": {
    "max_steps": 500,
    "use_vision": true,
    "max_parallel_browsers": 1,
    "enable_smart_retry": false,
    "enable_cost_saver": false,
    "model_priority_list": [],
    "tool_calling_method": "auto"
  },
  "browser": {
    "headless": false,
    "window_w": 1280,
    "window_h": 1100,
    "enable_live_view": true
  }
}</textarea>
        </div>
    </div>
    
    <div class="main">
        <div id="liveView" style="display:none; margin-bottom: 10px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: #000;">
            <div id="liveMonitor">
                <div><span class="monitor-label"># [STATE]:</span> <span id="monitorState" class="monitor-value"></span></div>
                <div><span class="monitor-label"># [GOAL]:</span> <span id="monitorGoal" class="monitor-value"></span></div>
                <div><span class="monitor-label"># [DATA]:</span> <span id="monitorData" class="monitor-value"></span></div>
            </div>
            <div id="agentStatusDisplay"></div>
            <div style="padding: 5px 10px; background: var(--panel-bg); border-bottom: 1px solid var(--border); color: var(--text-secondary); font-size: 0.8rem; display: flex; justify-content: space-between;">
                <span>LIVE VIEW</span>
                <span style="color: var(--error);">● REC</span>
            </div>
            <img id="liveImg" style="width: 100%; display: block;" />
        </div>
        <div id="messages">
            <div class="message agent">
                <div class="step-header">System</div>
                <div>Ready. Configure settings on the left and enter a task below.</div>
            </div>
        </div>
        
        <!-- Center Control Bar -->
        <div class="center-console" style="margin-bottom: 20px;">
            <div class="console-section">
                <h3>Manual Override</h3>
                <button id="forceReviewBtn" onclick="sendControl('force_review')">👁️ Force Review</button>
                <button id="forceParticipateBtn" onclick="sendControl('force_participate')">✍️ Force Participate</button>
                <button id="emergencyStopBtn" onclick="stopAgent(event)" style="background: var(--error);">🛑 Emergency All-Stop</button>
            </div>
            <div class="console-section">
                <h3>Context Shifters</h3>
                <button id="readVitalSourceBtn" onclick="sendControl('read_vitalsource')">📖 Read VitalSource</button>
                <button id="syncHotlinksBtn" onclick="sendControl('sync_hotlinks')">🌐 Sync Hotlinks</button>
                <button id="rubricCheckBtn" onclick="sendControl('rubric_check')">📋 Rubric Check</button>
            </div>
            <div class="console-section">
                <h3>Navigation & Speed</h3>
                <label for="speedSlider">Speed: <span id="speedValue">Human Pace</span></label>
                <input type="range" id="speedSlider" min="0" max="10" value="5" oninput="updateSpeed(this.value)">
                <button id="skipStepBtn" onclick="sendControl('skip_step')">⏭️ Skip Step</button>
                <button id="reloadRetryBtn" onclick="sendControl('reload_retry')">🔄 Reload/Retry</button>
            </div>
            <div class="console-section">
                <h3>Handshake Response</h3>
                <button id="letsGoBtn" onclick="sendConfirmation('yes')">✅ LETS GO</button>
                <button id="editDraftBtn" onclick="showEditDraftDialog()">📝 EDIT DRAFT</button>
                <button id="explainBtn" onclick="sendControl('explain_action')">❓ EXPLAIN</button>
                <div id="editDraftDialog" style="display: none; margin-top: 10px;">
                    <textarea id="editDraftText" placeholder="Edit AI's draft here..." style="width: 100%; height: 80px; margin-bottom: 5px;"></textarea>
                    <button onclick="sendEditedDraft()">Submit Edit</button>
                    <button onclick="hideEditDraftDialog()" class="secondary">Cancel</button>
                </div>
            </div>
            <div class="console-section">
                <h3>Utility</h3>
                <button id="showFocusBtn" onclick="sendControl('toggle_ghost_mouse')">🖱️ Show Focus</button>
            </div>
        </div>
        
        <form onsubmit="sendMessage(event)" style="display: flex; gap: 10px;">
            <input type="text" id="messageText" autocomplete="off" placeholder="Enter your task here..." style="padding: 15px; font-size: 1rem;" />
            <button id="sendBtn" type="submit" style="width: 100px;">Run</button>
            <button id="stopBtn" onclick="stopAgent(event)" type="button" disabled style="width: 80px;">Stop</button>
        </form>
    </div>
    
    <div class="right-sidebar">
        <div class="header">
            <h1>Task Management</h1>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h3>Current Task</h3>
            <div id="currentTaskDisplay" style="background: var(--bg-color); padding: 10px; border-radius: 6px; border: 1px solid var(--border); color: var(--text-primary);">No task running</div>
        </div>
        
        <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
            <h3>Task Queue</h3>
            <ul id="taskList" style="list-style: none; padding: 0; margin: 0; flex: 1; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; background: var(--bg-color);">
                <!-- Task items will be added here by JS -->
                <li style="padding: 8px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
                    <span>Example Task 1</span> <button style="width: auto; padding: 4px 8px; font-size: 0.7rem; background: var(--error);">✖️</button>
                </li>
            </ul>
            <div style="display: flex; gap: 5px;">
                <input type="text" id="newTaskInput" placeholder="Add new task..." style="flex: 1;" /> 
                <button id="addTaskBtn" style="width: auto; padding: 8px 12px;">➕</button>
            </div>
        </div>
    </div>

    <script>
        var ws = new WebSocket("ws://" + window.location.host + "/ws");
        var sendBtn = document.getElementById("sendBtn");
        var stopBtn = document.getElementById("stopBtn");
        var currentConfirmationPayload = null; // To store the payload from request_confirmation
        
        // Task Queue Manager
        document.getElementById('addTaskBtn').onclick = function() {
            var input = document.getElementById('newTaskInput');
            var task = input.value;
            if (!task) return;
            
            var list = document.getElementById('taskList');
            // Clear example if present
            if (list.children.length > 0 && list.children[0].innerText.includes("Example Task")) {
                list.innerHTML = "";
            }

            var li = document.createElement('li');
            li.style.cssText = "padding: 8px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;";
            li.innerHTML = "<span>" + task + "</span> <button onclick='this.parentElement.remove()' style='width: auto; padding: 4px 8px; font-size: 0.7rem; background: var(--error);'>✖️</button>";
            list.appendChild(li);
            
            sendControl('add_task', task);
            input.value = '';
        };
        
        ws.onopen = function() {
            console.log("Connected to WebSocket");
            fetch('/extraction_models')
                .then(response => response.json())
                .then(data => {
                    var select = document.getElementById('extractionModel');
                    data.forEach(model => {
                        var option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        select.appendChild(option);
                    });
                });
            fetch('/ollama_models')
                .then(response => response.json())
                .then(data => {
                    var select = document.getElementById('ollamaModel');
                    if (data.length > 0) {
                        data.forEach(model => {
                            var option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            select.appendChild(option);
                        });
                    } else {
                        var option = document.createElement('option');
                        option.textContent = "No models found (is Ollama running?)";
                        select.appendChild(option);
                        select.disabled = true;
                    }
                });
        };

        ws.onmessage = function(event) {
            var messages = document.getElementById('messages');
            var data = JSON.parse(event.data);
            
            var div = document.createElement('div');
            div.className = 'message';
            
            if (data.type === 'log') {
                div.classList.add('agent');
                div.innerHTML = "<div class='step-header'>Log</div><div class='thought'>ℹ️ " + data.content + "</div>";
            } else if (data.type === 'agent_status') {
                var statusDisplay = document.getElementById('agentStatusDisplay');
                if (data.status) {
                    statusDisplay.textContent = data.status;
                    statusDisplay.style.display = 'block';
                } else {
                    statusDisplay.style.display = 'none';
                }
            } else if (data.type === 'agent_goal') {
                document.getElementById('monitorGoal').textContent = data.goal;
                document.getElementById('liveMonitor').style.display = 'block';
            } else if (data.type === 'agent_data') {
                document.getElementById('monitorData').textContent = data.data;
                document.getElementById('liveMonitor').style.display = 'block';
            } else if (data.type === 'stream') {
                var view = document.getElementById('liveView');
                var img = document.getElementById('liveImg');
                view.style.display = 'block';
                img.src = "data:image/jpeg;base64," + data.image;
                return;
            } else if (data.type === 'step') {
                var actionsHtml = "";
                if (data.actions && data.actions.length > 0) {
                    actionsHtml = "<div style='margin-top: 5px; padding: 5px; background: rgba(255,255,255,0.05); border-radius: 4px;'>";
                    data.actions.forEach(action => {
                        actionsHtml += "<div style='font-family: monospace; font-size: 0.8rem;'>🛠️ " + JSON.stringify(action) + "</div>";
                    });
                    actionsHtml += "</div>";
                }

                div.classList.add('agent');
                div.innerHTML = "<div class='step-header'>Step " + data.step + "</div>" +
                                "<span class='thought'>" + (data.thought || "No thought") + "</span>" +
                                "<div class='url'>" + (data.url || "N/A") + "</div>" +
                                actionsHtml;
                if (data.screenshot) {
                    div.innerHTML += "<img src='data:image/png;base64," + data.screenshot + "' class='screenshot' />";
                }
            } else if (data.type === 'validation') {
                div.classList.add('agent');
                var icon = data.is_confirmed ? "✅" : "⚠️";
                div.innerHTML = "<div class='step-header'>" + icon + " Confirmer</div>" +
                                "<div>" + data.reason + "</div>";
                if (data.think) {
                    div.innerHTML += "<div class='confirmer-reasoning' style='margin-top:5px; padding:8px; background:rgba(0,0,0,0.3); border-left:2px solid #9ca3af; font-family:monospace; font-size:0.85rem; white-space: pre-wrap; display:none;'>" + data.think + "</div>";
                }
            } else if (data.type === 'request_confirmation') {
                showConfirmationDialog(
                    data.message || "Are you sure you want to proceed with this action?", // message
                    data.intel || "No specific new information.", // intel
                    data.next_task || "Agent is considering its next move." // next_task
                );
                currentConfirmationPayload = data; // Store the full payload
                return; // Do not append to messages, it's a modal
            } else if (data.type === 'result') {
                div.classList.add('agent');
                div.style.borderColor = 'var(--success)';
                div.innerHTML = "<div class='step-header' style='color: var(--success)'>✅ Final Result</div><div>" + data.content + "</div>";
                sendBtn.disabled = false;
                sendBtn.textContent = "Run";
                stopBtn.disabled = true;
                document.getElementById('liveMonitor').style.display = 'none'; // Hide monitor on task end
                document.getElementById('agentStatusDisplay').style.display = 'none'; // Hide agent status on task end
                document.getElementById('currentTaskDisplay').textContent = "No task running"; // Clear current task
            } else if (data.type === 'error') {
                div.classList.add('error');
                div.innerHTML = "<div class='step-header' style='color: var(--error)'>❌ Error</div><div>" + data.content + "</div>";
                sendBtn.disabled = false;
                sendBtn.textContent = "Run";
                stopBtn.disabled = true;
                document.getElementById('liveMonitor').style.display = 'none'; // Hide monitor on task end
                document.getElementById('agentStatusDisplay').style.display = 'none'; // Hide agent status on task end
                document.getElementById('currentTaskDisplay').textContent = "No task running"; // Clear current task
            }
            
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        };
        
        ws.onclose = function() {
            var messages = document.getElementById('messages');
            var div = document.createElement('div');
            div.className = 'message error';
            div.textContent = "⚠️ Disconnected from server. Please refresh.";
            messages.appendChild(div);
        };

        function sendMessage(event) { // Changed to be called by onclick, not form onsubmit
            event.preventDefault();
            var input = document.getElementById("messageText");
            if (!input.value) return;
            
            var messages = document.getElementById('messages');
            document.getElementById('liveMonitor').style.display = 'none'; // Hide monitor on new task
            document.getElementById('agentStatusDisplay').style.display = 'none'; // Hide status on new task
            messages.innerHTML = ''; // Clear previous
            
            var agentType = document.getElementById("agentType").value;
            var settings = {};
            try {
                settings = JSON.parse(document.getElementById("settingsJson").value);
            } catch(e) {
                console.error("Invalid JSON settings", e);
            }
            
            var resumeTaskId = document.getElementById("resumeTaskId").value;
            var googleDocsTemplate = document.getElementById("googleDocsTemplate").value;
            var extractionModel = document.getElementById("extractionModel").value;
            
            // Add user message
            var userDiv = document.createElement('div');
            userDiv.className = 'message user';
            userDiv.innerHTML = "<div class='step-header'>User Task</div><div>" + input.value + "</div>";
            
            // Update current task display in the right sidebar
            document.getElementById('currentTaskDisplay').textContent = input.value;
            messages.appendChild(userDiv);
            
            ws.send(JSON.stringify({action: "run", task: input.value, agent_type: agentType, resume_task_id: resumeTaskId, google_docs_template_url: googleDocsTemplate, extraction_model: extractionModel, ...settings}));
            
            sendBtn.disabled = true;
            stopBtn.disabled = false;
            sendBtn.textContent = "Running...";
            input.value = '';
        }
        
        function stopAgent(event) {
            event.preventDefault();
            ws.send(JSON.stringify({action: "stop"}));
            stopBtn.disabled = true;
        }

        // New functions for confirmation dialog
        document.getElementById('confirmYesBtn').onclick = function() { sendConfirmation('yes'); };
        document.getElementById('confirmNoBtn').onclick = function() { sendConfirmation('no'); };
        function showConfirmationDialog(message, intel, nextTask) {
            document.getElementById('confirmationMessage').textContent = message;
            document.getElementById('dialogIntel').textContent = intel;
            document.getElementById('dialogNextTask').textContent = nextTask;
            document.getElementById('confirmationOverlay').style.display = 'flex';
        }

        async function uploadFile(event) {
            event.preventDefault();
            var fileInput = document.getElementById('fileInput');
            var status = document.getElementById('uploadStatus');
            
            if (!fileInput.files[0]) {
                status.textContent = "Select file first.";
                return;
            }
            
            var formData = new FormData();
            formData.append("file", fileInput.files[0]);
            
            status.textContent = "Uploading...";
            
            try {
                var response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                var result = await response.json();
                if (response.ok) {
                    status.textContent = "✅ Uploaded: " + result.filename;
                } else {
                    status.textContent = "❌ Error: " + (result.detail || "Failed");
                }
            } catch (e) {
                status.textContent = "❌ Error: " + e.message;
            }
        }

        function updateAgentSettings() {
            var useVision = document.getElementById('useVision').checked;
            var maxConsecutiveFailures = document.getElementById('maxConsecutiveFailures').value;
            try {
                var settings = JSON.parse(document.getElementById("settingsJson").value);
                if (!settings.agent) settings.agent = {};
                settings.agent.use_vision = useVision;
                settings.agent.max_consecutive_failures = parseInt(maxConsecutiveFailures);
                document.getElementById("settingsJson").value = JSON.stringify(settings, null, 2);
            } catch(e) {
                console.error("Invalid JSON in settings", e);
            }
        }

        function toggleConfirmerReasoning() {
            var checked = document.getElementById('showConfirmerReasoning').checked;
            var style = document.getElementById('confirmer-style');
            if (!style) {
                style = document.createElement('style');
                style.id = 'confirmer-style';
                document.head.appendChild(style);
            }
            style.textContent = checked ? ".confirmer-reasoning { display: block !important; }" : ".confirmer-reasoning { display: none !important; }";
        }

        // New functions for Center Console controls
        function sendControl(command, value = null) {
            // Emergency stop is handled by the existing stopAgent function
            ws.send(JSON.stringify({action: "control", command: command, value: value}));
        }

        function updateSpeed(value) {
            const speedLabel = document.getElementById('speedValue');
            let speedText = '';
            if (value == 0) speedText = 'Paused';
            else if (value <= 3) speedText = 'Human Pace (Slow/Safe)';
            else if (value <= 7) speedText = 'Normal';
            else speedText = 'Turbo (Fast/High-Risk)';
            speedLabel.textContent = speedText;
            sendControl('set_speed', value);
        }

        function sendConfirmation(responseType, customTask = null, editedText = null) {
            // Use currentConfirmationPayload if available for context, otherwise default
            const payloadToSend = {
                action: "confirmation_response",
                response: responseType,
                custom_task: customTask || document.getElementById('customTaskInput').value,
                edited_text: editedText, // New field for edited draft
                // Optionally, include original intel/next_task from currentConfirmationPayload if needed by backend
                original_intel: currentConfirmationPayload ? currentConfirmationPayload.intel : null,
                original_next_task: currentConfirmationPayload ? currentConfirmationPayload.next_task : null,
            };
            ws.send(JSON.stringify(payloadToSend));
            document.getElementById('confirmationOverlay').style.display = 'none';
            document.getElementById('customTaskInput').value = ''; // Clear input after sending
            currentConfirmationPayload = null; // Clear stored payload
        }

        function showEditDraftDialog() {
            document.getElementById('editDraftDialog').style.display = 'block';
            document.getElementById('editDraftText').value = ''; // Clear previous
        }

        function hideEditDraftDialog() {
            document.getElementById('editDraftDialog').style.display = 'none';
        }

        function sendEditedDraft() {
            const editedText = document.getElementById('editDraftText').value;
            sendConfirmation('edit', null, editedText);
            hideEditDraftDialog();
        }

        function updateOllamaSettings() {
            var select = document.getElementById('ollamaModel');
            var model = select.value;
            if (!model) return;
            
            // Update UI fields
            document.getElementById('llmProvider').value = 'ollama';
            document.getElementById('modelName').value = model;
            
            updateLLMSettings();
        }

        function updateLLMSettings() {
            var provider = document.getElementById('llmProvider').value;
            var model = document.getElementById('modelName').value;
            var key = document.getElementById('apiKey').value;
            var baseUrl = document.getElementById('baseUrl').value;
            var googleProjectId = document.getElementById('googleProjectId').value;
            
            // Toggle Login vs API Key
            var loginPanel = document.getElementById('googleLoginPanel');
            var apiKeyField = document.getElementById('apiKeyField');
            var baseUrlField = document.getElementById('baseUrlField');
            var googleProjectIdField = document.getElementById('googleProjectIdField');
            
            if (provider === 'gemini') {
                loginPanel.style.display = 'block';
                apiKeyField.style.display = 'none';
                baseUrlField.style.display = 'none';
                googleProjectIdField.style.display = 'none';
            } else if (provider === 'ollama') {
                loginPanel.style.display = 'none';
                apiKeyField.style.display = 'none';
                baseUrlField.style.display = 'block';
                googleProjectIdField.style.display = 'none';
            } else if (provider === 'vertex') {
                loginPanel.style.display = 'none';
                apiKeyField.style.display = 'none';
                baseUrlField.style.display = 'none';
                googleProjectIdField.style.display = 'block';
            }
            else {
                loginPanel.style.display = 'none';
                apiKeyField.style.display = 'block';
                baseUrlField.style.display = 'none';
                googleProjectIdField.style.display = 'none';
            }

            try {
                var settings = JSON.parse(document.getElementById("settingsJson").value);
                if (!settings.llm) settings.llm = {};
                
                settings.llm.provider = provider;
                settings.llm.model_name = model;
                
                if (key) settings.llm.api_key = key;
                else delete settings.llm.api_key;

                if (googleProjectId) settings.llm.google_project_id = googleProjectId;
                else delete settings.llm.google_project_id;
                
                if (provider === 'ollama') settings.llm.base_url = baseUrl;
                else delete settings.llm.base_url;
                
                document.getElementById("settingsJson").value = JSON.stringify(settings, null, 2);
            } catch(e) {
                console.error("Invalid JSON in settings", e);
            }
        }
        
        // Init fields from JSON
        try {
            var s = JSON.parse(document.getElementById("settingsJson").value);
            if (s.llm) {
                if (s.llm.provider) document.getElementById('llmProvider').value = s.llm.provider;
                if (s.llm.model_name) document.getElementById('modelName').value = s.llm.model_name;
                if (s.llm.api_key) document.getElementById('apiKey').value = s.llm.api_key;
                if (s.llm.google_project_id) document.getElementById('googleProjectId').value = s.llm.google_project_id;
                if (s.llm.base_url) document.getElementById('baseUrl').value = s.llm.base_url;
            }
            if (s.agent) {
                document.getElementById('useVision').checked = s.agent.use_vision !== false;
                if (s.agent.max_consecutive_failures) document.getElementById('maxConsecutiveFailures').value = s.agent.max_consecutive_failures;
            }
            updateLLMSettings(); // Set initial state
        } catch(e) {}
        
        // Check login status
        fetch('/auth/status').then(r => r.json()).then(data => {
            if (data.is_logged_in) {
                document.getElementById('loginStatus').textContent = "✅ Logged in with Google";
                document.getElementById('loginStatus').style.color = "var(--success)";
                document.getElementById('loginBtn').style.display = 'none';
                document.getElementById('logoutBtn').style.display = 'block';
            }
        });
    </script>
</body>
</html>""" # End of HTML content

@app.get("/")
async def get():
    return FileResponse("static/index.html")

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
    agent_control_queue = asyncio.Queue() # Queue for sending control commands to the running agent

    confirmation_event = asyncio.Event()
    confirmation_response_queue = asyncio.Queue()
    
    async def run_agent_job_wrapper(task_payload):
        # Local variables to ensure thread safety and cleanup per task
        nonlocal browser, browser_context
        local_task_id = None
        
        task = task_payload.get("task")
        agent_type = task_payload.get("agent_type", "browser")
        llm_settings = task_payload.get("llm", {})
        agent_settings = task_payload.get("agent", {})
        browser_settings = task_payload.get("browser", {})

        # Get browser settings from payload, defaulting to None if not present or empty string
        browser_binary_path = browser_settings.get("browser_binary_path") or None
        browser_user_data_dir = browser_settings.get("browser_user_data_dir") or None
        keep_browser_open_setting = browser_settings.get("keep_browser_open", False)

        # Deep Research specific
        resume_task_id = task_payload.get("resume_task_id")
        mcp_config = task_payload.get("mcp_config")
        extraction_model_name = task_payload.get("extraction_model")

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
                    "google_docs_template_url": task_payload.get("google_docs_template_url", "a new Google Doc")
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
                # Define the callback for agent status updates and confirmation requests
                async def send_agent_message_callback(message_payload):
                    await websocket.send_json(message_payload)

                # Standard Browser Agent - Use browser_factory for consistent setup
                
                # Initialize Planner/Confirmer if configured
                planner_llm = llm # Assign the main LLM to the planner
                
                confirmer_llm = llm # Assign the main LLM to the confirmer
                
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
                controller = CustomController(output_model=output_model, kb_dir=KB_DIR)
                
                async def validation_callback(think, reason, is_confirmed):
                    await websocket.send_json({
                        "type": "validation",
                        "think": think,
                        "reason": reason,
                        "is_confirmed": is_confirmed
                    })

                # Browser and context creation
                # Update browser_settings with user-provided paths, or None for auto-detection
                browser_settings["browser_binary_path"] = browser_binary_path
                browser_settings["browser_user_data_dir"] = browser_user_data_dir
                browser_settings["use_own_browser"] = True # Always enable for custom paths or auto-detection

                browser_config_instance = BrowserConfig(**browser_settings)
                
                # If we are not keeping the browser open, or if it's the first run, or if settings changed significantly,
                # close existing browser/context and create new ones.
                # For simplicity, if keep_browser_open_setting is false, always create new.
                # If keep_browser_open_setting is true, only create if browser is None.
                if not keep_browser_open_setting or browser is None:
                    if browser_context:
                        await browser_context.close()
                        browser_context = None
                    if browser:
                        await browser.close()
                        browser = None
                    
                    browser = create_browser(browser_config_instance.model_dump()) # Pass dict for now
                    browser_context = await create_context(browser, browser_config_instance.new_context_config.model_dump()) # Pass dict for now
                
                agent = BrowserUseAgent(
                    task=task,
                    llm=llm,
                    browser=browser,
                    browser_context=browser_context,
                    controller=controller,
                    use_vision=agent_settings.get("use_vision", True), # Use the setting from agent_settings
                    planner_llm=planner_llm,
                    confirmer_llm=confirmer_llm,
                    inhibit_close=keep_browser_open_setting, # Inhibit closing if keep_browser_open is true
                    enable_smart_retry=enable_smart_retry,
                    enable_cost_saver=enable_cost_saver,
                    model_priority_list=model_priority_list,
                    validation_callback=validation_callback,
                    use_memory=True, # Enable custom memory manager for site knowledge
                    enable_user_interaction_dialog=agent_settings.get("enable_user_interaction_dialog", False), # New setting
                    tool_calling_method=agent_settings.get("tool_calling_method", "auto"),
                    send_agent_message_callback=send_agent_message_callback,
                    confirmation_event=confirmation_event,
                    confirmation_response_queue=confirmation_response_queue,
                    agent_control_queue=agent_control_queue # Pass the control queue to the agent
                )
                
                await send_agent_message_callback({"type": "agent_status", "status": "Participating"})
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
            logger.error(f"Error in agent wrapper: {e}", exc_info=True)
            await websocket.send_json({"type": "error", "content": str(e)})
        finally:
            # Cleanup logic after agent run
            if not keep_browser_open_setting:
                if browser_context:
                    await browser_context.close()
                    browser_context = None
                if browser:
                    await browser.close()
                    browser = None
            
            if local_task_id and local_task_id in _AGENT_STOP_FLAGS:
                del _AGENT_STOP_FLAGS[local_task_id]

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action = payload.get("action", "run")
                
                if action == "stop":
                    logger.info("Received 'stop' action. Cancelling runner task.")
                    if runner_task and not runner_task.done():
                        runner_task.cancel()
                    continue

                if action == "control":
                    command = payload.get("command")
                    value = payload.get("value")
                    logger.info(f"Received control command: {command} with value: {value}")
                    
                    if runner_task and not runner_task.done():
                        if command == "pause":
                            await websocket.send_json({"type": "agent_status", "status": "Paused ⏸️"})
                        elif command == "resume":
                            await websocket.send_json({"type": "agent_status", "status": "Participating"})

                    await agent_control_queue.put({"command": command, "value": value})
                    continue

                if action == "run":
                    task = payload.get("task") # This is the main task for the agent
                    agent_type = payload.get("agent_type", "browser")

                    if not task:
                        continue
                        
                    if runner_task and not runner_task.done():
                        await websocket.send_json({"type": "error", "content": "Agent already running"})
                        continue

                    # Reset confirmation state for a new run
                    confirmation_event.clear()
                    while not confirmation_response_queue.empty(): # Clear any stale responses
                        try: await confirmation_response_queue.get_nowait()
                        except asyncio.QueueEmpty: pass
                
                    runner_task = asyncio.create_task(run_agent_job_wrapper(payload))
                
                elif action == "confirmation_response" and runner_task and not runner_task.done():
                    response_data = {
                        "response": payload.get("response"),
                        "custom_task": payload.get("custom_task"),
                        "edited_text": payload.get("edited_text") # Handle edited draft
                    }
                    # The confirmation_response_queue is used by the agent to get user input for confirmation dialogs
                    logger.info(f"Received confirmation response: {response_data}")
                    await confirmation_response_queue.put(response_data)
                    continue
                else:
                    # If we reach here, it's an unhandled action
                    continue
                                 
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_json({"type": "error", "content": "Invalid JSON received"})
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "content": str(e)})
                                 
    except asyncio.CancelledError: # This catches cancellation of the websocket itself
        logger.info("Client disconnected or WebSocket task cancelled.")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally: # Final cleanup to ensure no browsers are left hanging
        if runner_task and not runner_task.done():
            runner_task.cancel() # Ensure the agent job is cancelled too
        if browser_context: await browser_context.close()
        if browser: await browser.close()
        logger.info("WebSocket connection closed and resources cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser Agent Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"Starting FastAPI server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)




    