
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Agent Pro - Ultimate Edition</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #0f1419;
            --bg-tertiary: #141922;
            --surface: #1a1f2e;
// Control Bar Module
function initControlBar() {
    const runButton = document.querySelector('.btn[onclick="runAgent()"]');
    const stopButton = document.querySelector('.btn.btn-secondary[onclick="stopAgent()"]');
    const mainInput = document.getElementById('mainInput');

    runButton.addEventListener('click', runAgent);
    stopButton.addEventListener('click', stopAgent);

    mainInput.addEventListener('keydown', function (event) {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            addTaskBefore(this.value);
        }
    });
}

function setMode(mode) {
    showTicker('setMode: ' + mode);
}

            --surface-hover: #222838;
            --surface-active: #2a324a;
            --text-primary: #ffffff;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #3b82f6;
            --accent-hover: #2563eb;
            --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --priority-high: #ef4444;
            --priority-med: #f59e0b;
            --priority-low: #10b981;
            --border: #2d3748;
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-lg: 0 10px 25px rgba(0,0,0,0.5);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            font-size: 13px;
        }

        /* Layout - Enhanced Grid */
        .app-container {
            display: grid;
            grid-template-columns: 300px 1fr 400px; /* Wider Task Queue */
            height: 100vh;
        }

        /* Global UI Components */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px 16px;
            background: var(--accent-primary);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }
        .btn:hover { background: var(--accent-hover); transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .btn-ghost { background: transparent; color: var(--text-secondary); }
        .btn-ghost:hover { background: var(--surface-hover); color: var(--text-primary); }
        .btn-icon { padding: 6px; aspect-ratio: 1; border-radius: var(--radius-sm); }
        .btn-sm { padding: 4px 10px; font-size: 11px; }
        .btn-secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text-secondary); }
        .btn-secondary:hover { border-color: var(--text-muted); color: var(--text-primary); }

        input, select, textarea {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: var(--radius-sm);
            font-family: inherit;
            font-size: 12px;
            width: 100%;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent-primary);
        }

        /* Sidebar Base */
        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .sidebar.right { border-left: 1px solid var(--border); border-right: none; }
        
        .sidebar-header {
            padding: 16px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sidebar-title { font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; }
        .sidebar-content { flex: 1; overflow-y: auto; padding: 16px; }

        /* Left Sidebar Specifics */
        .config-section { margin-bottom: 20px; }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            cursor: pointer;
            user-select: none;
        }
        .section-title { 
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--text-muted); font-weight: 600; 
        }
        .section-content { display: block; }
        .section-content.collapsed { display: none; }
        
        .form-row { margin-bottom: 12px; }
        .form-label { display: block; margin-bottom: 6px; color: var(--text-secondary); font-size: 11px; }

        /* Main Content */
        .main-content { display: flex; flex-direction: column; background: var(--bg-primary); min-width: 0; }
        
        .monitor-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-bottom: 1px solid var(--border);
            position: relative;
            background: #000;
        }
        
        .monitor-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }
        .stat-box {
            padding: 12px;
            border-right: 1px solid var(--border);
            text-align: center;
        }
        .stat-label { font-size: 10px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
        .stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--accent-primary); }

        .viewport-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        .viewport-img { max-width: 100%; max-height: 100%; object-fit: contain; }
        
        /* Control Bar */
        .control-bar {
            padding: 16px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
        }
        .ticker {
            background: var(--bg-tertiary);
            padding: 8px 12px;
            border-radius: var(--radius-sm);
            margin-bottom: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--border);
        }

        /* Right Sidebar - Task Queue Ultimate */
        .task-toolbar {
            display: flex;
            gap: 8px;
            padding: 12px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
        }
        
        .search-box {
            position: relative;
            flex: 1;
        }
        .search-input { padding-left: 28px; }
        .search-icon { position: absolute; left: 8px; top: 50%; transform: translateY(-50%); color: var(--text-muted); }

        .task-list { padding: 12px; display: flex; flex-direction: column; gap: 8px; }

        /* Draggable Task Item */
        .task-item {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0;
            transition: all 0.2s;
            position: relative;
            display: flex;
            flex-direction: column;
        }
        .task-item:hover { border-color: var(--text-muted); transform: translateY(-1px); box-shadow: var(--shadow-sm);}
        .task-item.dragging { opacity: 0.5; border: 1px dashed var(--accent-primary); }
        .task-item.active { border-color: var(--accent-primary); background: var(--surface-active); }
        .task-item.done { opacity: 0.6; }
        .task-item.done .task-title { text-decoration: line-through; }

        .task-header {
            display: flex;
            align-items: center;
            padding: 10px;
            gap: 8px;
        }

        .drag-handle {
            color: var(--text-muted);
            cursor: grab;
            padding: 4px;
            font-size: 14px;
            opacity: 0.5;
        }
        .drag-handle:hover { opacity: 1; color: var(--text-primary); }

        .priority-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border); }
        .p-high { background: var(--priority-high); box-shadow: 0 0 5px rgba(239, 68, 68, 0.4); }
        .p-med { background: var(--priority-med); }
        .p-low { background: var(--priority-low); }

        .task-content { flex: 1; min-width: 0; cursor: pointer; }
        .task-title { font-weight: 500; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .task-meta { display: flex; gap: 8px; font-size: 10px; color: var(--text-muted); align-items: center; }

        .task-actions { opacity: 0; transition: opacity 0.2s; display: flex; gap: 2px; }
        .task-item:hover .task-actions { opacity: 1; }

        /* Task Details (Expanded) */
        .task-details {
            border-top: 1px solid var(--border);
            padding: 10px;
            background: var(--bg-tertiary);
            font-size: 11px;
            display: none;
            border-bottom-left-radius: var(--radius-md);
            border-bottom-right-radius: var(--radius-md);
        }
        .task-item.expanded .task-details { display: block; }
        .task-note { color: var(--text-secondary); margin-bottom: 8px; font-style: italic; }

        /* Progress Bar */
        .progress-container {
            height: 4px;
            background: var(--bg-tertiary);
            width: 100%;
            position: absolute;
            bottom: 0;
            left: 0;
            display: none;
        }
        .task-item.active .progress-container { display: block; }
        .progress-bar { height: 100%; background: var(--accent-primary); width: 0%; transition: width 0.3s; }

        /* Modals */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
        }
        .modal-overlay.active { opacity: 1; pointer-events: all; }
        .modal {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            width: 500px;
            max-width: 90%;
            box-shadow: var(--shadow-lg);
            transform: scale(0.95);
            transition: transform 0.2s;
        }
        .modal-overlay.active .modal { transform: scale(1); }
        
        .modal-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-body { padding: 20px; }
        .modal-footer {
            padding: 16px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            background: var(--bg-secondary);
            border-bottom-left-radius: var(--radius-lg);
            border-bottom-right-radius: var(--radius-lg);
        }

        /* Tooltip */
        .tooltip {
            position: absolute;
            background: var(--surface-active);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            pointer-events: none;
            border: 1px solid var(--border);
            z-index: 2000;
            opacity: 0;
            transition: opacity 0.2s;
        }

        /* Keyboard Shortcuts Help */
        .kbd {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 2px 6px;
            font-family: monospace;
            font-size: 10px;
            color: var(--text-secondary);
        }

        .shortcuts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .shortcut-item { display: flex; justify-content: space-between; align-items: center; padding: 8px; border-radius: 4px; background: var(--bg-tertiary); }
    </style>
</head>
<body>

    <div class="app-container">
        
        <!-- LEFT SIDEBAR: CONFIG -->
        <aside class="sidebar" id="configSidebar">
            <div class="sidebar-header">
                <div class="sidebar-title" id="configSidebarTitle">⚡ Browser Agent Pro</div>
                <button class="btn-icon btn-ghost" onclick="toggleShortcuts()" title="Keyboard Shortcuts (Ctrl+/)">⌨️</button>
            </div>
            
            <div class="sidebar-content">
                <!-- Agent Settings -->
                <div class="config-section">
                    <div class="section-header" onclick="toggleSection('agent-config')">
                        <span class="section-title" id="agentConfigTitle">Agent Configuration</span>
                        <span>▼</span>
                    </div>
                    <div id="agent-config" class="section-content">
                        <div class="form-row">
                            <label class="form-label">Model Provider</label>
                            <select id="llmProvider">
                                <option value="openai">OpenAI (GPT-4o)</option>
                                <option value="anthropic">Anthropic (Claude 3.5)</option>
                                <option value="google">Google (Gemini 1.5)</option>
                                <option value="ollama">Local (Ollama)</option>
                            </select>
                        </div>
                        <div class="form-row">
                            <label class="form-label">Mode</label>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-sm btn-secondary" style="flex:1" onclick="setMode('fast')">Fast</button>
                                <button class="btn btn-sm btn-secondary active" style="flex:1" style="border-color: var(--accent-primary);">Deep</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Advanced Settings (Collapsed) -->
                <div class="config-section">
                    <div class="section-header" onclick="toggleSection('adv-config')">
                        <span class="section-title" id="advancedControlTitle">Advanced Control</span>
                        <span>▼</span>
                    </div>
                    <div id="adv-config" class="section-content collapsed">
                        <div class="form-row">
                            <label class="form-label">Max Steps: <span id="maxStepVal">50</span></label>
                            <input type="range" min="10" max="100" value="50" oninput="document.getElementById('maxStepVal').innerText=this.value">
                        </div>
                        <div class="form-row">
                            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                <input type="checkbox" checked>
                                <span class="form-label" style="margin:0">Auto-save Tasks</span>
                            </label>
                        </div>
                        <div class="form-row">
                            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                <input type="checkbox" checked>
                                <span class="form-label" style="margin:0">Use Vision</span>
                            </label>
                        </div>
                    </div>
                </div>

                <!-- JSON Config -->
                <div class="config-section">
                     <div class="section-header">
                        <span class="section-title" id="rawConfigTitle">Raw Config</span>
                     </div>
                     <textarea style="height: 150px; font-family: monospace; font-size: 11px;">{
  "temperature": 0.0,
  "top_p": 0.9,
  "browser_dims": [1280, 800]
}</textarea>
                </div>
            </div>
        </aside>

        <!-- CENTER: MONITOR & CHAT -->
        <main class="main-content" id="mainContent">
         <!-- Enhanced Monitor Stats -->
            <div class="monitor-stats">
                <div class="stat-box">
                    <div class="stat-label">Status</div>
                    <div class="stat-value" id="statusText" style="color: var(--success);">IDLE</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Step</div>
                    <div class="stat-value"><span id="currentStep">0</span>/<span id="totalSteps">--</span></div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Progress</div>
                    <div class="stat-value" id="progressPct">0%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Time</div>
                    <div class="stat-value" id="elapsedTime">00:00</div>
                </div>
            </div>

          <!-- Live Viewport -->
            <div class="viewport-container">
                <div style="text-align: center; color: var(--text-muted);">
                    <div style="font-size: 40px; margin-bottom: 10px;">🌐</div>
                    <div>Agent Browser View</div>
                </div>
                <!-- Actual Image would go here: <img src="..." class="viewport-img"> -->
            </div>

         <!-- Control Bar -->
            <div class="control-bar" id="controlBar">
                <div class="ticker">
                    <span>🤖</span>
                    <span id="tickerText">Ready for instructions...</span>
                </div>
                <div style="display: flex; gap: 10px;">
                    <div class="search-box">
                        <span class="search-icon">⚡</span>
                        <input type="text" id="mainInput" class="search-input" placeholder="What should the agent do next? (Ctrl+Enter to run)" autocomplete="off">
                    </div>
                    <button class="btn" onclick="runAgent()">▶ Run</button>
                    <button class="btn btn-secondary" onclick="stopAgent()" style="color: var(--error);">⏹ Stop</button>
                </div>
            </div>
        </main>

        <!-- RIGHT SIDEBAR: ULTIMATE TASK QUEUE -->
        <aside class="sidebar right" id="taskQueueSidebar">
         <div class="sidebar-header">
                <div class="sidebar-title">📋 Task Queue</div>
                <div style="display: flex; gap: 4px;">
                    <button class="btn-icon btn-ghost" title="Import/Export" onclick="exportTasks()">💾</button>
                    <button class="btn-icon btn-ghost" title="Clear All" onclick="clearTasks()">🗑️</button>
                </div>
            </div>

            <!-- Task Toolbar -->
          <div class="task-toolbar">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="taskSearch" class="search-input" placeholder="Filter tasks..." onkeyup="filterTasks()">
                </div>
                <select style="width: 100px;" onchange="sortTasks(this.value)">
                    <option value="order">Order</option>
                    <option value="priority">Priority</option>
                    <option value="status">Status</option>
                </select>
            </div>

           <!-- Quick Templates -->
            <div style="padding: 12px; display: flex; gap: 8px; overflow-x: auto; border-bottom: 1px solid var(--border);">
                <button class="btn btn-sm btn-secondary" onclick="loadTemplate('research')">🔍 Research</button>
                <button class="btn btn-sm btn-secondary" onclick="loadTemplate('analysis')">📊 Analysis</button>
                <button class="btn btn-sm btn-secondary" onclick="loadTemplate('test')">🧪 Test</button>
            </div>

            <div class="sidebar-content" style="padding: 0;">
                <div id="taskList" class="task-list">
                    <!-- Tasks injected by JS -->
                </div>
                <div style="padding: 12px;">
                    <button class="btn btn-secondary" style="width: 100%; border-style: dashed;" onclick="openTaskEditor()">+ Add New Task</button>
                </div>
            </div>
        </aside>

    </div>

    <!-- MODALS -->
    <!-- Task Editor Modal -->
    <div id="taskEditorModal" class="modal-overlay" >
        <div class="modal">
            <div class="modal-header">
                <h3>✏️ Edit Task</h3>
                <button class="btn-icon btn-ghost" onclick="closeModal('taskEditorModal')">✕</button>
            </div>
            <div class="modal-body">
                <input type="hidden" id="editTaskId">
                <div class="form-row">
                    <label class="form-label">Task Title</label>
                    <input type="text" id="editTaskTitle" placeholder="Enter task description...">
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                    <div>
                        <label class="form-label">Priority</label>
                        <select id="editTaskPriority">
                            <option value="high">🔴 High</option>
                            <option value="med">🟡 Medium</option>
                            <option value="low">🟢 Low</option>
                        </select>
                    </div>
                    <div>
                        <label class="form-label">Est. Time (min)</label>
                        <input type="number" id="editTaskTime" value="5">
                    </div>
                </div>
                <div class="form-row">
                    <label class="form-label">Notes / Context</label>
                    <textarea id="editTaskNotes" rows="4" placeholder="Add extra context, URLs, or requirements..."></textarea>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal('taskEditorModal')">Cancel</button>
                <button class="btn" onclick="saveTaskFromModal()">Save Task</button>
            </div>
        </div>
    </div>

    <!-- Shortcuts Modal -->
    <div id="shortcutsModal" class="modal-overlay" >
        <div class="modal">
            <div class="modal-header">
                <h3>⌨️ Keyboard Shortcuts</h3>
                <button class="btn-icon btn-ghost" onclick="toggleShortcuts()">✕</button>
            </div>
            <div class="modal-body">
                <div class="shortcuts-grid">
                    <div class="shortcut-item"><span>Run Agent</span> <span class="kbd">Ctrl + Enter</span></div>
                    <div class="shortcut-item"><span>Stop Agent</span> <span class="kbd">Esc</span></div>
                    <div class="shortcut-item"><span>New Task</span> <span class="kbd">Ctrl + N</span></div>
                    <div class="shortcut-item"><span>Save Tasks</span> <span class="kbd">Ctrl + S</span></div>
                    <div class="shortcut-item"><span>Search Tasks</span> <span class="kbd">Ctrl + K</span></div>
                    <div class="shortcut-item"><span>Toggle Pause</span> <span class="kbd">Ctrl + P</span></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Tooltip Element -->
    <div id="tooltip" class="tooltip"></div>

    <script>
        // --- State Management ---
        let tasks = [];
        let config = {};
        let draggingId = null;
        let isRunning = false;
        let currentTaskIndex = -1; // Index of the currently running task
        let isPaused = false;
        
        // --- Initialization ---
        document.addEventListener('DOMContentLoaded', () => {
            initConfig();
            loadTasks();
            setupShortcuts();
            renderTasks();
            startTimer();
            initControlBar();
            initTaskQueue();
            initModals();
        });
        
        // --- DOM Elements ---
        const configSidebar = document.getElementById('configSidebar');
        const mainContent = document.getElementById('mainContent');
        const controlBar = document.getElementById('controlBar');
        const taskQueueSidebar = document.getElementById('taskQueueSidebar');


        // --- Task Logic ---
        function createTaskObj(title, priority = 'med', time = 5, notes = '') {
            return {
                id: Date.now().toString(),
                title,
                priority,
                time,
                notes,
                status: 'pending', // pending, active, done
                order: tasks.length,
                expanded: false
            };
        }

        function addTask(title, priority, time, notes) {
            tasks.push(createTaskObj(title, priority, time, notes));
            renderTasks();
            saveTasks();
        }


        function renderTasks() {
            const filter = document.getElementById('taskSearch').value.toLowerCase();
            const list = document.getElementById('taskList');

            if (!list) {
                console.warn('Task list element not found.');
                return;
            }        tasks.forEach((task, index) => {
                if (!task.title.toLowerCase().includes(filter)) return;

                const el = document.createElement('div');
                el.className = `task-item ${task.status} ${task.expanded ? 'expanded' : ''} ${task.status === 'active' ? 'active' : ''}`;
                el.draggable = true;
                
                // Priority Color Class
                const pClass = task.priority === 'high' ? 'p-high' : task.priority === 'low' ? 'p-low' : 'p-med';
                
                el.innerHTML = `
                    <div class="progress-container"><div class="progress-bar" style="width: 45%"></div></div>
                    <div class="task-header">
                        <div class="drag-handle" onmousedown="setDrag('${task.id}')">⋮⋮</div>
                        <div class="priority-dot ${pClass}" title="Priority: ${task.priority}"></div>
                        <div class="task-content" onclick="toggleExpand('${task.id}')">
                            <div class="task-title">${task.title}</div>
                            <div class="task-meta">
                                <span>⏱ ${task.time}m</span>
                                ${task.notes ? '<span>📝</span>' : ''}
                                <span>${task.status}</span>
                            </div>
                        </div>
                        <div class="task-actions">
                            <button class="btn-icon btn-ghost btn-sm" onclick="editTask('${task.id}')">✏️</button>
                            <button class="btn-icon btn-ghost btn-sm" style="color:var(--error)" onclick="deleteTask('${task.id}')">✕</button>
                        </div>
                    </div>
                    <div class="task-details">
                        ${task.notes ? `<div class="task-note">${task.notes}</div>` : ''}
                        <div style="display:flex; justify-content:flex-end; gap:8px;">
                            <button class="btn btn-sm btn-secondary" onclick="markStatus('${task.id}', 'pending')">Queue</button>
                            <button class="btn btn-sm btn-secondary" onclick="markStatus('${task.id}', 'active')">Run</button>
                            <button class="btn btn-sm btn-secondary" onclick="markStatus('${task.id}', 'done')">Done</button>
                        </div>
                    </div>
                `;

                // Drag Events
                el.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', task.id);
                    el.classList.add('dragging');
                    draggingId = task.id;
                });
                el.addEventListener('dragend', () => {
                    el.classList.remove('dragging');
                    draggingId = null;
                });
                el.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    const afterElement = getDragAfterElement(list, e.clientY);
                    if (afterElement == null) {
                        list.appendChild(el);
                    } else {
                        list.insertBefore(el, afterElement);
                    }
                });
                el.addEventListener('drop', (e) => {
                    e.preventDefault();
                    // Reorder array logic would go here based on DOM order
                    saveOrderFromDom();
                });

                list.appendChild(el);
            });
        }

        // --- Drag & Drop Helpers ---
        function setDrag(id) { draggingId = id; }
        
        function getDragAfterElement(container, y) {
            const draggableElements = [...container.querySelectorAll('.task-item:not(.dragging)')];
            return draggableElements.reduce((closest, child) => {
                const box = child.getBoundingClientRect();
                const offset = y - box.top - box.height / 2;
                if (offset < 0 && offset > closest.offset) {
                    return { offset: offset, element: child };
                } else {
                    return closest;
                }
            }, { offset: Number.NEGATIVE_INFINITY }).element;
        }

        function saveOrderFromDom() {
            // Re-map tasks array based on DOM
            // This is a placeholder for the actual reorder logic
            // In a real app, map DOM IDs back to data array
        }

        // --- Task Editor Modal ---
        function openTaskEditor(id = null) {
            const modal = document.getElementById('taskEditorModal');
            document.getElementById('editTaskId').value = id || '';
            
            if (id) {
                const task = tasks.find(t => t.id === id);
                document.getElementById('editTaskTitle').value = task.title;
                document.getElementById('editTaskPriority').value = task.priority;
                document.getElementById('editTaskTime').value = task.time;
                document.getElementById('editTaskNotes').value = task.notes;
            } else {
                document.getElementById('editTaskTitle').value = '';
                document.getElementById('editTaskNotes').value = '';
            }
            
            modal.classList.add('active');
            document.getElementById('editTaskTitle').focus();
        }

        function saveTaskFromModal() {
            const id = document.getElementById('editTaskId').value;
            const title = document.getElementById('editTaskTitle').value;
            const prio = document.getElementById('editTaskPriority').value;
            const time = document.getElementById('editTaskTime').value;
            const notes = document.getElementById('editTaskNotes').value;

            if (!title) return;

            if (id) {
                const task = tasks.find(t => t.id === id);
                task.title = title;
                task.priority = prio;
                task.time = time;
                task.notes = notes;
            } else {
                addTask(title, prio, time, notes);
            }
            
            closeModal('taskEditorModal');
            renderTasks();
            saveTasks();
        }

        function editTask(id) { openTaskEditor(id); }
        
        function deleteTask(id) {
            if(confirm('Delete this task?')) {
                tasks = tasks.filter(t => t.id !== id);
                renderTasks();
                saveTasks();
            }
        }

        function toggleExpand(id) {
            const task = tasks.find(t => t.id === id);
            task.expanded = !task.expanded;
            renderTasks();
        }

        function markStatus(id, status) {
            const task = tasks.find(t => t.id === id);
            task.status = status;
            renderTasks();
            saveTasks();
        }

        // --- Templates ---
        function loadTemplate(type) {
            const templates = {
                'research': [
                    createTaskObj('Search for latest trends', 'high', 10),
                    createTaskObj('Compile sources in Doc', 'med', 15),
                    createTaskObj('Summarize key findings', 'high', 20)
                ],
                'analysis': [
                    createTaskObj('Scrape competitor data', 'high', 30),
                    createTaskObj('Generate comparison matrix', 'med', 20)
                ],
                'test': [
                    createTaskObj('Run unit tests', 'high', 5),
                    createTaskObj('Check responsive layout', 'low', 10),
                ]
            };

            tasks = [...tasks, ...templates[type]];
            renderTasks();
            saveTasks();
            showTicker(`Template '${type}' loaded!`);
        }

         // --- Task Control ---
        function startTask(index) {
            if (index < 0 || index >= tasks.length) return;

            // Pause current task
            if (currentTaskIndex !== -1 && currentTaskIndex < tasks.length) {
                tasks[currentTaskIndex].status = 'pending';
            }

            // Start new task
            currentTaskIndex = index;
            tasks[currentTaskIndex].status = 'active';
            renderTasks();
            saveTasks();
        }

        function pauseTask() {
            isPaused = true;
            document.getElementById('statusText').textContent = "PAUSED";
            document.getElementById('statusText').style.color = "var(--priority-med)";
        }

        // --- Utilities ---
        function filterTasks() { renderTasks(); }
        
        function sortTasks(criteria) {
            if (criteria === 'priority') {
                const map = { high: 0, med: 1, low: 2 };
                tasks.sort((a, b) => map[a.priority] - map[b.priority]);
            } else if (criteria === 'status') {
                const map = { active: 0, pending: 1, done: 2 };
                tasks.sort((a, b) => map[a.status] - map[b.status]);
            }
            renderTasks();
        }

        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }

        function toggleShortcuts() {
            const m = document.getElementById('shortcutsModal');
            m.classList.toggle('active');
        }

        function toggleSection(id) {
            document.getElementById(id).classList.toggle('collapsed');
        }

        function showTicker(msg) {
            if (typeof msg !== 'string') {
                msg = JSON.stringify(msg);
            }
            document.getElementById('tickerText').textContent = msg;
            // Add flash animation logic here
        }

        // --- LocalStorage ---
        function saveTasks() { localStorage.setItem('agentTasks', JSON.stringify(tasks)); }
        function loadTasks() {
            const saved = localStorage.getItem('agentTasks');
            if (saved) tasks = JSON.parse(saved);
        }
        function exportTasks() {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(tasks));
            const dlAnchorElem = document.createElement('a');
            dlAnchorElem.setAttribute("href", dataStr);
            dlAnchorElem.setAttribute("download", "agent_tasks.json");
            dlAnchorElem.click();
        }
        function clearTasks() {
            tasks = [];
            renderTasks();
        }

        // --- Keyboard Shortcuts ---
        function setupShortcuts() {
            document.addEventListener('keydown', (e) => {
                // Ctrl/Cmd + Enter
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    runAgent();
                }
                // Ctrl/Cmd + S
                if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                    e.preventDefault();
                    saveTasks();
                    showTicker("Tasks saved manually.");
                }
                // Ctrl/Cmd + /
                if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                    e.preventDefault();
                    toggleShortcuts();
                }
                // Ctrl/Cmd + K
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    document.getElementById('taskSearch').focus();
                }
                // Ctrl/Cmd + N
                if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                    e.preventDefault();
                    openTaskEditor();
                }
            });
        }

        // --- Mock Agent Control ---
        function runAgent() {
            isRunning = true;
            document.getElementById('statusText').textContent = "RUNNING";
            document.getElementById('statusText').style.color = "var(--warning)";
            isPaused = false;
            showTicker("Agent started...");
        }

        function stopAgent() {
            isRunning = false;
            document.getElementById('statusText').textContent = "IDLE";
            document.getElementById('statusText').style.color = "var(--success)";
            showTicker("Agent stopped by user.");
        }

        function startTimer() {
            let seconds = 0;
            setInterval(() => {
                if (isRunning) {
                    seconds++;
                    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
                    const s = (seconds % 60).toString().padStart(2, '0');
                    document.getElementById('elapsedTime').textContent = `${m}:${s}`;
                }
            }, 1000);
        }

        // --- Control Bar Functions ---
        function addTaskBefore(title) {
            if (!title) return;

            let newTask = createTaskObj(title);
            tasks.unshift(newTask); // Add to beginning
            
            // Update order values
            tasks.forEach((task, index) => task.order = index);
            
            startTask(0); // Start the new task
            
            document.getElementById('mainInput').value = ''; // Clear input

            renderTasks();
            saveTasks();
        }

        // --- Add event listener for Ctrl+Enter to add task before ---
        document.getElementById('mainInput').addEventListener('keydown', function(event) {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                 addTaskBefore(this.value);
             }
        });

        // --- Component Modules ---

        // Config Sidebar Module
        function initConfig() {
            const fastModeButton = document.querySelector('.btn-sm.btn-secondary[onclick="setMode(\'fast\')"]');
            const deepModeButton = document.querySelector('.btn-sm.btn-secondary.active');

             function updateConfig() {
                config = {
                    llmProvider: llmProviderSelect.value,
                     mode: deepModeButton.classList.contains('active') ? 'deep' : 'fast'
                };
                showTicker(config);
            }

            fastModeButton.addEventListener('click', () => {
                fastModeButton.classList.add('active');
                deepModeButton.classList.remove('active');
                updateConfig();
            });
            deepModeButton.addEventListener('click', () => {
                deepModeButton.classList.add('active');
                fastModeButton.classList.remove('active');
                updateConfig();
            });
        }

        // Control Bar Module
        function initControlBar() {

            runButton.addEventListener('click', runAgent);
            stopButton.addEventListener('click', stopAgent);
            mainInput.addEventListener('keydown', function(event) {
                if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                    addTaskBefore(this.value);
                }
            });
        }

        // Task Queue Module
        function initTaskQueue() {
             taskSearchInput.addEventListener('keyup', filterTasks);

             function filterTasks() {
                const filter = taskSearchInput.value.toLowerCase();
                 renderTasks();
             }
        }

        // Modals Module
        function initModals() {

        }


        function setMode(mode) {
            showTicker('setMode: ' + mode);
        }


    </script>
</body>
</html>