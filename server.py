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
html = """
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
        .url { background: rgba(59, 130, 246, 0.1); color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-family: monospace; display: inline-block; }
        .screenshot { max-width: 100%; border-radius: 6px; margin-top: 10px; border: 1px solid var(--border); }
        
        /* Artifacts */
        .artifacts { background: var(--panel-bg); padding: 15px; border-radius: 8px; border: 1px solid var(--border); max-height: 150px; overflow-y: auto; }
        .artifacts ul { list-style: none; padding: 0; margin: 0; }
        .artifacts li { margin-bottom: 5px; font-size: 0.9rem; }
        .artifacts a { color: var(--accent); text-decoration: none; }
        .artifacts a:hover { text-decoration: underline; }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="header">
            <h1>Browser Use UI</h1>
        </div>
        
        <div>
            <label>Agent Type</label>
            <select id="agentType">
                <option value="browser">Browser Agent</option>
                <option value="deep_research">Deep Research Agent</option>
            </select>
        </div>
        
        <div style="background: var(--panel-bg); padding: 10px; border: 1px solid var(--border); border-radius: 6px;">
            <label>LLM Provider</label>
            <select id="llmProvider" onchange="updateLLMSettings()">
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini (Google)</option>
                <option value="vertex">Vertex AI</option>
                <option value="ollama">Ollama</option>
                <option value="anthropic">Anthropic</option>
            </select>
            
            <div id="googleLoginPanel" style="display: none; margin-top: 10px; padding: 10px; background: rgba(66, 133, 244, 0.1); border-radius: 4px;">
                <div id="loginStatus" style="font-size: 0.8rem; margin-bottom: 5px; color: var(--text-secondary);">Not logged in</div>
                <a id="loginBtn" href="/auth/login" style="text-decoration: none;">
                    <button type="button" style="background: #4285F4; border: none; color: white;">Sign in with Google</button>
                </a>
                <a id="logoutBtn" href="/auth/logout" style="text-decoration: none; display: none;">
                    <button type="button" style="background: var(--border); border: none; color: var(--text-primary);">Sign out</button>
                </a>
            </div>

            <div id="apiKeyField" style="margin-top: 10px;">
                <label>API Key</label>
                <input type="password" id="apiKey" placeholder="Enter API Key..." onchange="updateLLMSettings()" />
            </div>
            <div id="googleProjectIdField" style="margin-top: 10px; display: none;">
                <label>Google Project ID</label>
                <input type="text" id="googleProjectId" placeholder="Enter Google Project ID..." onchange="updateLLMSettings()" />
            </div>
            <div id="baseUrlField" style="margin-top: 10px; display: none;">
                <label>Base URL</label>
                <input type="text" id="baseUrl" placeholder="http://localhost:11434" value="http://localhost:11434" onchange="updateLLMSettings()" />
            </div>
            <div style="margin-top: 10px;">
                <label>Model Name</label>
                <input type="text" id="modelName" placeholder="gpt-4o / gemini-2.0-flash-exp" onchange="updateLLMSettings()" />
            </div>
            <div style="margin-top: 10px; display: flex; align-items: center; gap: 5px;">
                <input type="checkbox" id="useVision" style="width: auto;" onchange="updateAgentSettings()" checked />
                <label for="useVision" style="margin: 0; cursor: pointer;">Use Vision</label>
            </div>
            <div style="margin-top: 5px; display: flex; align-items: center; gap: 5px;">
                <input type="checkbox" id="showConfirmerReasoning" style="width: auto;" onchange="toggleConfirmerReasoning()" />
                <label for="showConfirmerReasoning" style="margin: 0; cursor: pointer;">Show Confirmer Reasoning</label>
            </div>
        </div>
        
        <div>
            <label>Quick Load Ollama Model</label>
            <select id="ollamaModel" onchange="updateOllamaSettings()">
                <option value="">Select a model...</option>
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
        
        <div style="border-top: 1px solid var(--border); padding-top: 15px;">
            <label>File Upload</label>
            <div style="display: flex; gap: 5px;">
                <input type="file" id="fileInput" style="font-size: 0.8rem;" />
                <button onclick="uploadFile(event)" style="width: auto; padding: 5px 10px;">⬆️</button>
            </div>
            <div id="uploadStatus" style="font-size: 0.8rem; margin-top: 5px; color: var(--text-secondary);"></div>
        </div>
        
        <div style="flex: 1; overflow: hidden; display: flex; flex-direction: column;">
            <label>Advanced Settings (JSON)</label>
            <textarea id="settingsJson" style="flex: 1; font-family: monospace; font-size: 0.8rem;">{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4o",
    "temperature": 0.0
  },
  "agent": {
    "max_steps": 50,
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
    "enable_live_view": false
  }
}</textarea>
        </div>
    </div>
    
    <div class="main">
        <div id="liveView" style="display:none; margin-bottom: 10px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: #000;">
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
        
        <div class="artifacts">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <h3>Generated Artifacts</h3>
                <button class="secondary" onclick="refreshFiles()" style="width: auto; padding: 2px 8px; font-size: 0.8rem;">Refresh</button>
            </div>
            <ul id="fileList"></ul>
        </div>
        
        <form onsubmit="sendMessage(event)" style="display: flex; gap: 10px;">
            <input type="text" id="messageText" autocomplete="off" placeholder="Enter your task here..." style="padding: 15px; font-size: 1rem;" />
            <button id="sendBtn" type="submit" style="width: 100px;">Run</button>
            <button id="stopBtn" onclick="stopAgent(event)" type="button" disabled style="width: 80px;">Stop</button>
        </form>
    </div>

    <script>
        var ws = new WebSocket("ws://" + window.location.host + "/ws");
        var sendBtn = document.getElementById("sendBtn");
        var stopBtn = document.getElementById("stopBtn");
        
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
            refreshFiles();
        };

        ws.onmessage = function(event) {
            var messages = document.getElementById('messages');
            var data = JSON.parse(event.data);
            
            var div = document.createElement('div');
            div.className = 'message';
            
            if (data.type === 'log') {
                div.classList.add('agent');
                div.innerHTML = "<div class='step-header'>Log</div><div class='thought'>ℹ️ " + data.content + "</div>";
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
            } else if (data.type === 'result') {
                div.classList.add('agent');
                div.style.borderColor = 'var(--success)';
                div.innerHTML = "<div class='step-header' style='color: var(--success)'>✅ Final Result</div><div>" + data.content + "</div>";
                sendBtn.disabled = false;
                sendBtn.textContent = "Run";
                stopBtn.disabled = true;
            } else if (data.type === 'error') {
                div.classList.add('error');
                div.innerHTML = "<div class='step-header' style='color: var(--error)'>❌ Error</div><div>" + data.content + "</div>";
                sendBtn.disabled = false;
                sendBtn.textContent = "Run";
                stopBtn.disabled = true;
            }
            
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            refreshFiles();
        };

        ws.onclose = function() {
            var messages = document.getElementById('messages');
            var div = document.createElement('div');
            div.className = 'message error';
            div.textContent = "⚠️ Disconnected from server. Please refresh.";
            messages.appendChild(div);
        };

        function sendMessage(event) {
            event.preventDefault();
            var input = document.getElementById("messageText");
            if (!input.value) return;
            
            var messages = document.getElementById('messages');
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

        async function refreshFiles() {
            try {
                var response = await fetch('/files');
                var files = await response.json();
                var list = document.getElementById('fileList');
                list.innerHTML = '';
                files.forEach(file => {
                    var li = document.createElement('li');
                    li.innerHTML = '<a href="/' + file + '" target="_blank">' + file + '</a>';
                    list.appendChild(li);
                });
            } catch(e) {
                console.error("Failed to load files", e);
            }
        }

        function updateAgentSettings() {
            var useVision = document.getElementById('useVision').checked;
            try {
                var settings = JSON.parse(document.getElementById("settingsJson").value);
                if (!settings.agent) settings.agent = {};
                settings.agent.use_vision = useVision;
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
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)
async def read_root():
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
                        try:
                            keep_browser_open_setting = browser_settings.get("keep_browser_open", False)

                            # Determine if browser/context needs to be recreated or reused.
                            # Recreate if keep_browser_open is false, or if browser is not yet initialized.
                            # More sophisticated logic could compare current browser_settings with previous ones.
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
                                # Standard Browser Agent - Use browser_factory for consistent setup
                                
                                # If browser is not initialized, or if keep_browser_open is false, recreate the browser.
                                if not browser or not keep_browser_open_setting:
                                    if browser: # If browser exists but we need to recreate it
                                        logger.info("♻️  Closing existing browser for recreation due to settings or keep_browser_open=False...")
                                        await browser.close()
                                    logger.info("🚀 Creating new browser instance using browser_factory...")
                                    browser = create_browser(browser_settings)
                                else:
                                    logger.info("✅ Reusing existing browser instance.")
                                
                                # Always close previous context if it exists, to ensure fresh state for new task
                                # If keep_browser_open is true, we want to reuse the browser instance but get a fresh context.
                                if browser_context: 
                                    await browser_context.close() 
                                    browser_context = None

                                logger.info("✨ Creating new browser context using browser_factory...")
                                browser_context = await create_context(browser, browser_settings)

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
                                controller = CustomController(output_model=output_model, kb_dir=KB_DIR)
                                
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
                                    use_vision=agent_settings.get("use_vision", True), # Use the setting from agent_settings
                                    planner_llm=planner_llm,
                                    confirmer_llm=confirmer_llm,
                                    inhibit_close=keep_browser_open_setting, # Inhibit closing if keep_browser_open is true
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
                        
                        finally:
                            if local_task_id:
                                await stop_browsers_for_task(local_task_id)

                    runner_task = asyncio.create_task(run_agent_job())
                
            except Exception as e:
                # This outer try-except catches errors in parsing payload or starting run_agent_job
                logger.error(f"WebSocket message error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"WebSocket message processing error: {e}"})
                    
    except asyncio.CancelledError: # This catches cancellation of the websocket 8itself
        logger.info("Client disconnected or WebSocket task cancelled.")
        if runner_task and not runner_task.done():
            runner_task.cancel() # Ensure the agent job is cancelled too
        # Ensure browser and context are closed on client disconnect if they were kept open
        if browser_context: await browser_context.close()
        if browser: 
            await browser.close()
            logger.error(f"WebSocket message error: {e}")
                    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        if runner_task and not runner_task.done():
            runner_task.cancel()
        # Ensure browser and context are closed on client disconnect if they were kept open
        if browser_context: await browser_context.close() # Ensure context is closed
        if browser: await browser.close() # Ensure browser is closed
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Final cleanup to ensure no browsers are left hanging
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