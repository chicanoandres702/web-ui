document.addEventListener("DOMContentLoaded", () => {
    loadComponents();
});

function loadComponents() {
    const components = {
        'modals-container': 'components/modals.html',
        'left-sidebar-container': 'components/left-sidebar.html',
        'main-content-container': 'components/main-content.html',
        'right-sidebar-container': 'components/right-sidebar.html',
        'floating-controls-container': 'components/floating-controls.html',
        'quick-notes-container': 'components/quick-notes.html',
        'batch-operations-container': 'components/batch-operations.html'
    };

    for (const [containerId, componentUrl] of Object.entries(components)) {
        const container = document.getElementById(containerId);
        if (container) {
            fetch(componentUrl)
                .then(response => response.text())
                .then(html => {
                    container.innerHTML = html;
                })
                .catch(error => console.error(`Error loading component ${componentUrl}:`, error));
        }
    }
}

// ===== State Management =====
const ws = new WebSocket("ws://" + window.location.host + "/ws");
let stepCount = 0;
let tasks = [];
let taskGroups = [];
let currentFilter = 'all';
let currentEditTaskId = null;
let autoSaveEnabled = true;
let selectedTasks = new Set();
let sessionStartTime = Date.now();
let timerInterval = null;

// ===== Academic Templates =====
const taskTemplates = {
    quiz: {
        name: "Complete Quiz",
        tasks: [
            { text: "Open quiz on learning platform", priority: "high", time: 2, tags: ["quiz", "homework"] },
            { text: "Read all questions carefully", priority: "high", time: 5, tags: ["quiz"] },
            { text: "Answer all questions", priority: "high", time: 30, tags: ["quiz"] },
            { text: "Review answers before submitting", priority: "medium", time: 10, tags: ["quiz"] },
            { text: "Submit quiz", priority: "high", time: 2, tags: ["quiz"] }
        ]
    },
    reading: {
        name: "Reading Assignment",
        tasks: [
            { text: "Locate reading material", priority: "high", time: 5, tags: ["reading", "homework"] },
            { text: "Read assigned chapters/pages", priority: "high", time: 45, tags: ["reading"] },
            { text: "Take notes on key points", priority: "medium", time: 20, tags: ["reading", "notes"] },
            { text: "Review and highlight important sections", priority: "low", time: 15, tags: ["reading"] },
            { text: "Complete reading comprehension questions", priority: "medium", time: 15, tags: ["reading", "homework"] }
        ]
    },
    paper: {
        name: "Research Paper",
        tasks: [
            { text: "Research topic and gather sources", priority: "high", time: 60, tags: ["paper", "research"] },
            { text: "Create outline", priority: "high", time: 30, tags: ["paper", "writing"] },
            { text: "Write introduction", priority: "medium", time: 30, tags: ["paper", "writing"] },
            { text: "Write body paragraphs", priority: "high", time: 90, tags: ["paper", "writing"] },
            { text: "Write conclusion", priority: "medium", time: 20, tags: ["paper", "writing"] },
            { text: "Add citations and bibliography", priority: "high", time: 30, tags: ["paper", "citations"] },
            { text: "Proofread and edit", priority: "medium", time: 30, tags: ["paper", "editing"] },
            { text: "Final review and submission", priority: "high", time: 15, tags: ["paper"] }
        ]
    },
    yellowdig: {
        name: "Yellowdig Post",
        tasks: [
            { text: "Review discussion prompt/topic", priority: "high", time: 10, tags: ["yellowdig", "discussion"] },
            { text: "Research and gather supporting information", priority: "medium", time: 20, tags: ["yellowdig", "research"] },
            { text: "Draft post (250-300 words)", priority: "high", time: 25, tags: ["yellowdig", "writing"] },
            { text: "Add relevant hashtags and citations", priority: "medium", time: 5, tags: ["yellowdig"] },
            { text: "Post and engage with 2 peer posts", priority: "medium", time: 20, tags: ["yellowdig", "engagement"] }
        ]
    },
    homework: {
        name: "General Homework",
        tasks: [
            { text: "Review assignment instructions", priority: "high", time: 5, tags: ["homework"] },
            { text: "Gather necessary materials", priority: "medium", time: 10, tags: ["homework"] },
            { text: "Complete main assignment tasks", priority: "high", time: 45, tags: ["homework"] },
            { text: "Double-check work for errors", priority: "medium", time: 15, tags: ["homework"] },
            { text: "Submit assignment", priority: "high", time: 5, tags: ["homework"] }
        ]
    },
    project: {
        name: "Project Workflow",
        tasks: [
            { text: "Define project scope and objectives", priority: "high", time: 30, tags: ["project", "planning"] },
            { text: "Create project timeline", priority: "high", time: 20, tags: ["project", "planning"] },
            { text: "Research and data collection", priority: "high", time: 90, tags: ["project", "research"] },
            { text: "Develop project deliverables", priority: "high", time: 120, tags: ["project", "development"] },
            { text: "Create presentation materials", priority: "medium", time: 60, tags: ["project", "presentation"] },
            { text: "Practice presentation", priority: "medium", time: 30, tags: ["project", "presentation"] },
            { text: "Final review and submit", priority: "high", time: 20, tags: ["project"] }
        ]
    }
};

// ===== Session Timer =====
function startSessionTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - sessionStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('sessionTimer').textContent = 
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
}

// ===== WebSocket Handlers =====
ws.onopen = function() {
    console.log("🔌 Connected");
    loadModels();
    loadSavedData();
    startSessionTimer();
    showNotification("Connected to agent server", "success");
};

ws.onerror = function(error) {
    console.error("WebSocket error:", error);
    showNotification("Connection error", "error");
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleMessage(data);
};

function handleMessage(data) {
    switch(data.type) {
        case 'plan_update':
            updateTaskQueue(data.plan);
            break;
        case 'log':
            updateTicker('log', data.content);
            break;
        case 'step':
            stepCount++;
            document.getElementById('stepCounter').textContent = `${stepCount} Steps`;
            updateTicker('step', `Step ${data.step}: ${data.thought || "Processing..."}`);
            if (data.screenshot) {
                document.getElementById('liveImg').src = "data:image/png;base64," + data.screenshot;
            }
            updateProgress();
            break;
        case 'stream':
            document.getElementById('liveImg').src = "data:image/jpeg;base64," + data.image;
            if (data.state) document.getElementById('monitorState').textContent = data.state;
            if (data.goal) document.getElementById('monitorGoal').textContent = data.goal;
            if (data.data) document.getElementById('monitorData').textContent = data.data;
            break;
        case 'result':
            updateTicker('success', data.content);
            resetControls();
            showNotification("Task completed!", "success");
            break;
        case 'error':
            updateTicker('error', data.content);
            resetControls();
            showNotification("Task failed", "error");
            break;
        case 'confirmation_required':
            showConfirmationDialog(data);
            break;
        case 'agent_status':
            updateAgentStatus(data.status);
            break;
    }
}

// ===== UI Updates =====
function updateTicker(type, text) {
    const ticker = document.getElementById('messageTicker');
    const cleanText = text.replace(/<[^>]*>?/gm, '');
    const shortText = cleanText.length > 120 ? cleanText.substring(0, 120) + '...' : cleanText;
    
    const icons = {
        'step': '👣', 'log': '💬', 'success': '✅', 'error': '❌', 'info': '🤖'
    };
    
    ticker.innerHTML = `
        <span class="ticker-icon">${icons[type] || '🤖'}</span>
        <span class="ticker-text" title="${cleanText}">${shortText}</span>
    `;
}

function updateAgentStatus(status) {
    const statusSpan = document.querySelector('.status-indicator span');
    if (statusSpan) statusSpan.textContent = status;
    
    const overlay = document.getElementById('agentStatusDisplay');
    if (overlay) {
        const isActive = status.includes('Participating') || status.includes('Reviewing');
        overlay.textContent = isActive ? '' : status;
    }
}

function updateProgress() {
    const total = tasks.length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    document.getElementById('monitorProgress').textContent = `${progress}%`;
    document.getElementById('monitorTasks').textContent = `${completed}/${total}`;
    document.getElementById('taskProgressFill').style.width = `${progress}%`;
}

// ===== Task Management =====
function sendMessage(event) {
    event.preventDefault();
    const input = document.getElementById("messageText");
    const task = input.value.trim();
    
    if (!task) {
        showNotification("Please enter a task", "warning");
        return;
    }

    stepCount = 0;
    document.getElementById('stepCounter').textContent = '0 Steps';

    const agentType = document.getElementById("agentType").value;
    const settings = JSON.parse(document.getElementById("settingsJson").value);

    updateTicker('info', `Starting: ${task}`);
    document.getElementById('currentTaskDisplay').textContent = task;

    ws.send(JSON.stringify({
        action: "run",
        task: task,
        agent_type: agentType,
        ...settings
    }));

    document.getElementById("sendBtn").disabled = true;
    document.getElementById("sendBtn").innerHTML = '⏳ Running...';
    document.getElementById("stopBtn").disabled = false;
    input.value = '';
}

function stopAgent(event) {
    event.preventDefault();
    ws.send(JSON.stringify({action: "stop"}));
    resetControls();
    showNotification("Agent stopped", "warning");
}

function sendControl(command, value = null) {
    ws.send(JSON.stringify({
        action: "control",
        command: command,
        value: value
    }));
}

function resetControls() {
    document.getElementById("sendBtn").disabled = false;
    document.getElementById("sendBtn").innerHTML = '▶️ Run';
    document.getElementById("stopBtn").disabled = true;
}

function updateTaskQueue(plan) {
    if (!plan || plan.length === 0) return;

    tasks = plan.map((item, index) => ({
        id: Date.now() + index,
        text: `${index + 1}. ${item.step}`,
        status: item.status || 'pending',
        result: item.result || '',
        priority: 'medium',
        time: 15,
        dueDate: null,
        tags: [],
        notes: '',
        subtasks: [],
        expanded: false
    }));

    renderTasks();
    updateProgress();
}

function renderTasks() {
    const list = document.getElementById('taskList');
    list.innerHTML = '';

    let filteredTasks = filterTasksByStatus(tasks);
    
    const searchTerm = document.getElementById('taskSearch').value.toLowerCase();
    if (searchTerm) {
        filteredTasks = filteredTasks.filter(t => 
            t.text.toLowerCase().includes(searchTerm) ||
            t.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }

    document.getElementById('taskCount').textContent = filteredTasks.length;

    if (filteredTasks.length === 0) {
        list.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-muted); font-size: 11px;">No tasks found</div>';
        return;
    }

    filteredTasks.forEach(task => {
        const div = createTaskElement(task);
        list.appendChild(div);
    });

    setupDragAndDrop();
}

function createTaskElement(task) {
    const div = document.createElement('div');
    div.className = `task-item ${task.status} ${task.subtasks.length > 0 ? 'has-subtasks' : ''} ${selectedTasks.has(task.id) ? 'selected' : ''}`;
    div.draggable = true;
    div.dataset.taskId = task.id;
    if (task.status === 'in_progress') div.classList.add('active');

    const icons = {
        'pending': '⏳', 'in_progress': '▶️', 'completed': '✅', 'failed': '❌'
    };

    const isDue = task.dueDate && new Date(task.dueDate) < new Date();

    const mainRow = document.createElement('div');
    mainRow.className = 'task-main-row';
    mainRow.innerHTML = `
        <div class="task-checkbox ${selectedTasks.has(task.id) ? 'checked' : ''}" onclick="toggleTaskSelection(event, ${task.id})"></div>
        <span class="task-drag-handle">⋮⋮</span>
        <span class="task-icon">${icons[task.status] || '⏳'}</span>
        <div class="task-content">
            <div class="task-text">${task.text}</div>
            <div class="task-meta">
                <span>⏱️ ${task.time}m</span>
                ${task.dueDate ? `<span class="task-due-date ${isDue ? 'overdue' : ''}">📅 ${new Date(task.dueDate).toLocaleDateString()}</span>` : ''}
                ${task.tags.map(tag => `<span class="task-tag">${tag}</span>`).join('')}
                ${task.subtasks.length > 0 ? `<span>📋 ${task.subtasks.length}</span>` : ''}
            </div>
        </div>
        <div class="task-priority ${task.priority}"></div>
        <div class="task-actions">
            <button class="btn btn-xs tooltip" data-tooltip="Edit" onclick="editTask(event, ${task.id})">✏️</button>
            <button class="btn btn-xs tooltip" data-tooltip="Subtask" onclick="addSubtask(event, ${task.id})">➕</button>
            <button class="btn btn-xs btn-error tooltip" data-tooltip="Delete" onclick="deleteTask(event, ${task.id})">✕</button>
        </div>
        ${task.subtasks.length > 0 || task.notes ? `<span class="task-expand ${task.expanded ? 'expanded' : ''}" onclick="toggleTaskExpand(event, ${task.id})">▼</span>` : ''}
    `;
    div.appendChild(mainRow);

    if (task.expanded && (task.subtasks.length > 0 || task.notes)) {
        if (task.subtasks.length > 0) {
            const subtasksContainer = document.createElement('div');
            subtasksContainer.className = 'subtasks-container';
            
            task.subtasks.forEach((subtask, subIndex) => {
                const subtaskDiv = document.createElement('div');
                subtaskDiv.className = `subtask-item ${subtask.status}`;
                const subtaskIcons = {
                    'pending': '○', 'in_progress': '◐', 'completed': '●'
                };
                subtaskDiv.innerHTML = `
                    <span class="subtask-icon">${subtaskIcons[subtask.status] || '○'}</span>
                    <span class="subtask-text">${subtask.text}</span>
                    <div class="subtask-actions">
                        <button class="btn btn-xs" onclick="toggleSubtaskStatus(event, ${task.id}, ${subIndex})">✓</button>
                        <button class="btn btn-xs btn-error" onclick="deleteSubtask(event, ${task.id}, ${subIndex})">✕</button>
                    </div>
                `;
                subtasksContainer.appendChild(subtaskDiv);
            });

            div.appendChild(subtasksContainer);
        }

        if (task.notes) {
            const notesDiv = document.createElement('div');
            notesDiv.className = 'task-notes';
            notesDiv.textContent = `📝 ${task.notes}`;
            div.appendChild(notesDiv);
        }
    }

    return div;
}

function addNewTask() {
    const input = document.getElementById('newTaskInput');
    const taskText = input.value.trim();
    
    if (!taskText) return;
    
    const newTask = {
        id: Date.now(),
        text: taskText,
        status: 'pending',
        result: '',
        priority: 'medium',
        time: 15,
        dueDate: null,
        tags: [],
        notes: '',
        subtasks: [],
        expanded: false
    };

    tasks.push(newTask);
    renderTasks();
    updateProgress();
    autoSave();
    
    sendControl('add_task', taskText);
    input.value = '';
    showNotification("Task added", "success");
}

function deleteTask(event, taskId) {
    event.stopPropagation();
    tasks = tasks.filter(t => t.id !== taskId);
    selectedTasks.delete(taskId);
    renderTasks();
    updateProgress();
    updateBatchBar();
    autoSave();
    showNotification("Task removed", "info");
}

function editTask(event, taskId) {
    event.stopPropagation();
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;

    currentEditTaskId = taskId;
    document.getElementById('editTaskText').value = task.text;
    document.getElementById('editTaskPriority').value = task.priority;
    document.getElementById('editTaskTime').value = task.time;
    document.getElementById('editTaskDueDate').value = task.dueDate || '';
    document.getElementById('editTaskTags').value = task.tags.join(', ');
    document.getElementById('editTaskNotes').value = task.notes;
    document.getElementById('editTaskOverlay').classList.add('active');
}

document.getElementById('saveTaskBtn').onclick = function() {
    if (currentEditTaskId === null) return;

    const task = tasks.find(t => t.id === currentEditTaskId);
    if (task) {
        task.text = document.getElementById('editTaskText').value;
        task.priority = document.getElementById('editTaskPriority').value;
        task.time = parseInt(document.getElementById('editTaskTime').value) || 15;
        task.dueDate = document.getElementById('editTaskDueDate').value || null;
        task.tags = document.getElementById('editTaskTags').value.split(',').map(t => t.trim()).filter(t => t);
        task.notes = document.getElementById('editTaskNotes').value;
        
        renderTasks();
        autoSave();
        showNotification("Task updated", "success");
    }

    document.getElementById('editTaskOverlay').classList.remove('active');
    currentEditTaskId = null;
};

document.getElementById('cancelTaskBtn').onclick = function() {
    document.getElementById('editTaskOverlay').classList.remove('active');
    currentEditTaskId = null;
};

function addSubtask(event, taskId) {
    event.stopPropagation();
    const subtaskText = prompt("Enter subtask:");
    if (!subtaskText || !subtaskText.trim()) return;

    const task = tasks.find(t => t.id === taskId);
    if (task) {
        task.subtasks.push({
            text: subtaskText.trim(),
            status: 'pending'
        });
        task.expanded = true;
        renderTasks();
        autoSave();
        showNotification("Subtask added", "success");
    }
}

function toggleSubtaskStatus(event, taskId, subtaskIndex) {
    event.stopPropagation();
    const task = tasks.find(t => t.id === taskId);
    if (task && task.subtasks[subtaskIndex]) {
        const statuses = ['pending', 'in_progress', 'completed'];
        const currentStatus = task.subtasks[subtaskIndex].status;
        const nextIndex = (statuses.indexOf(currentStatus) + 1) % statuses.length;
        task.subtasks[subtaskIndex].status = statuses[nextIndex];
        renderTasks();
        autoSave();
    }
}

function deleteSubtask(event, taskId, subtaskIndex) {
    event.stopPropagation();
    const task = tasks.find(t => t.id === taskId);
    if (task) {
        task.subtasks.splice(subtaskIndex, 1);
        renderTasks();
        autoSave();
    }
}

function toggleTaskExpand(event, taskId) {
    event.stopPropagation();
    const task = tasks.find(t => t.id === taskId);
    if (task) {
        task.expanded = !task.expanded;
        renderTasks();
    }
}

function clearAllTasks() {
    if (confirm("Clear all tasks?")) {
        tasks = [];
        selectedTasks.clear();
        renderTasks();
        updateProgress();
        updateBatchBar();
        autoSave();
        showNotification("Queue cleared", "info");
    }
}

function sortTasks() {
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    tasks.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);
    renderTasks();
    showNotification("Tasks sorted by priority", "info");
}

// ===== Selection & Batch Operations =====
function toggleTaskSelection(event, taskId) {
    event.stopPropagation();
    if (selectedTasks.has(taskId)) {
        selectedTasks.delete(taskId);
    } else {
        selectedTasks.add(taskId);
    }
    renderTasks();
    updateBatchBar();
}

function selectAllTasks() {
    selectedTasks.clear();
    tasks.forEach(t => selectedTasks.add(t.id));
    renderTasks();
    updateBatchBar();
    showNotification("All tasks selected", "info");
}

function deselectAll() {
    selectedTasks.clear();
    renderTasks();
    updateBatchBar();
}

function updateBatchBar() {
    const bar = document.getElementById('batchOperations');
    const count = selectedTasks.size;
    document.getElementById('selectedCount').textContent = count;
    
    if (count > 0) {
        bar.classList.add('active');
    } else {
        bar.classList.remove('active');
    }
}

function batchSetPriority(priority) {
    selectedTasks.forEach(taskId => {
        const task = tasks.find(t => t.id === taskId);
        if (task) task.priority = priority;
    });
    renderTasks();
    autoSave();
    showNotification(`Set priority to ${priority}`, "success");
}

function batchDelete() {
    if (confirm(`Delete ${selectedTasks.size} tasks?`)) {
        tasks = tasks.filter(t => !selectedTasks.has(t.id));
        selectedTasks.clear();
        renderTasks();
        updateProgress();
        updateBatchBar();
        autoSave();
        showNotification("Tasks deleted", "success");
    }
}

// ===== Filters =====
function filterTasks() {
    renderTasks();
}

function setFilter(status) {
    currentFilter = status;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.filter-btn[onclick="setFilter('${status}')"]`).classList.add('active');
    renderTasks();
    showNotification(`Filter set to: ${status}`, "info");
}

function filterTasksByStatus(tasksToFilter) {
    if (currentFilter === 'all') {
        return tasksToFilter;
    }
    return tasksToFilter.filter(t => t.status === currentFilter || (currentFilter === 'done' && (t.status === 'completed' || t.status === 'failed')));
}

// ===== Drag and Drop =====
function setupDragAndDrop() {
    const list = document.getElementById('taskList');
    let draggedItem = null;

    list.addEventListener('dragstart', (e) => {
        draggedItem = e.target;
        setTimeout(() => {
            e.target.classList.add('dragging');
        }, 0);
    });

    list.addEventListener('dragend', (e) => {
        e.target.classList.remove('dragging');
        draggedItem = null;
        autoSave();
        showNotification("Task order saved", "info");
    });

    list.addEventListener('dragover', (e) => {
        e.preventDefault();
        const afterElement = getDragAfterElement(list, e.clientY);
        const currentElement = document.querySelector('.dragging');
        if (afterElement == null) {
            list.appendChild(currentElement);
        } else {
            list.insertBefore(currentElement, afterElement);
        }
        
        // Update tasks array order
        const taskId = parseInt(currentElement.dataset.taskId);
        const taskIndex = tasks.findIndex(t => t.id === taskId);
        const [movedTask] = tasks.splice(taskIndex, 1);
        
        if(afterElement) {
            const afterId = parseInt(afterElement.dataset.taskId);
            const afterIndex = tasks.findIndex(t => t.id === afterId);
            tasks.splice(afterIndex, 0, movedTask);
        } else {
            tasks.push(movedTask);
        }
    });
}

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

// ===== Modals and Dialogs =====
function showConfirmationDialog(data) {
    document.getElementById('dialogIntel').value = data.intel || 'N/A';
    document.getElementById('dialogNextTask').value = data.next_task || 'N/A';
    document.getElementById('confirmationOverlay').classList.add('active');

    document.getElementById('confirmYesBtn').onclick = () => {
        const customTask = document.getElementById('customTaskInput').value;
        sendControl('confirm_yes', customTask || null);
        document.getElementById('confirmationOverlay').classList.remove('active');
        document.getElementById('customTaskInput').value = '';
    };
    document.getElementById('confirmNoBtn').onclick = () => {
        sendControl('confirm_no');
        document.getElementById('confirmationOverlay').classList.remove('active');
    };
}

function showSaveGroupDialog() {
    document.getElementById('saveGroupOverlay').classList.add('active');
}

document.getElementById('saveGroupBtn').onclick = function() {
    const name = document.getElementById('groupNameInput').value.trim();
    if (!name) {
        showNotification("Group name is required", "warning");
        return;
    }
    
    const desc = document.getElementById('groupDescInput').value.trim();
    const group = {
        name: name,
        description: desc,
        tasks: tasks.map(t => ({...t, id: undefined})) // Remove runtime ID
    };

    taskGroups.push(group);
    renderTaskGroups();
    autoSave();
    showNotification("Task group saved", "success");

    document.getElementById('saveGroupOverlay').classList.remove('active');
    document.getElementById('groupNameInput').value = '';
    document.getElementById('groupDescInput').value = '';
};

document.getElementById('cancelGroupBtn').onclick = function() {
    document.getElementById('saveGroupOverlay').classList.remove('active');
};

function renderTaskGroups() {
    const list = document.getElementById('taskGroupsList');
    list.innerHTML = '';
    if (taskGroups.length === 0) {
        list.innerHTML = '<div style="text-align: center; font-size: 10px; color: var(--text-muted); padding: 5px 0;">No saved groups</div>';
    } else {
        taskGroups.forEach((group, index) => {
            const div = document.createElement('div');
            div.className = 'task-group-item';
            div.innerHTML = `
                <div>
                    <span class="task-group-name">${group.name}</span>
                    <span class="task-group-count">${group.tasks.length} tasks</span>
                </div>
                <div class="task-group-actions">
                    <button class="btn btn-xs" onclick="loadTaskGroup(${index})">Load</button>
                    <button class="btn btn-xs btn-error" onclick="deleteTaskGroup(${index})">✕</button>
                </div>
            `;
            list.appendChild(div);
        });
    }
}

function loadTaskGroup(index) {
    if (confirm(`Load group "${taskGroups[index].name}"? This will replace current tasks.`)) {
        tasks = taskGroups[index].tasks.map((t, i) => ({...t, id: Date.now() + i}));
        renderTasks();
        updateProgress();
        showNotification("Task group loaded", "success");
    }
}

function deleteTaskGroup(index) {
    if (confirm(`Delete group "${taskGroups[index].name}"?`)) {
        taskGroups.splice(index, 1);
        renderTaskGroups();
        autoSave();
        showNotification("Task group deleted", "info");
    }
}

function loadTemplate(templateName) {
    if (confirm(`Load template "${taskTemplates[templateName].name}"? This will replace current tasks.`)) {
        tasks = taskTemplates[templateName].tasks.map((t, i) => ({
            ...t,
            id: Date.now() + i,
            status: 'pending',
            result: '',
            notes: '',
            subtasks: [],
            expanded: false
        }));
        renderTasks();
        updateProgress();
        showNotification("Template loaded", "success");
    }
}

// ===== Sidebar, Panel & Notes Toggles =====
function toggleSidebar(side) {
    const sidebar = document.getElementById(side === 'left' ? 'leftSidebar' : 'rightSidebar');
    sidebar.classList.toggle('collapsed');
}

function toggleSection(headerElement) {
    const section = headerElement.closest('.collapsible-section');
    const content = section.querySelector('.section-content');
    const toggle = section.querySelector('.section-toggle');
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

function toggleFloatingPanel() {
    const panel = document.getElementById('floatingControls');
    panel.classList.toggle('minimized');
    document.getElementById('floatingToggleIcon').textContent = panel.classList.contains('minimized') ? '＋' : '−';
}

function toggleQuickNotes() {
    document.getElementById('quickNotes').classList.toggle('hidden');
}

// ===== Settings & Config =====
function updateLLMSettings() {
    const provider = document.getElementById("llmProvider").value;
    const apiKeyField = document.getElementById("apiKeyField");
    const settings = JSON.parse(document.getElementById("settingsJson").value);

    settings.llm.provider = provider;
    
    if (provider === "ollama") {
        apiKeyField.style.display = "none";
        settings.llm.model_name = "llama2";
    } else {
        apiKeyField.style.display = "block";
        if (provider === 'gemini') {
             settings.llm.model_name = "gemini-flash-latest";
        } else if (provider === 'openai') {
            settings.llm.model_name = "gpt-4-turbo-preview";
        } else if (provider === 'anthropic') {
            settings.llm.model_name = "claude-3-opus-20240229";
        }
    }
    
    document.getElementById("modelName").value = settings.llm.model_name;
    document.getElementById("settingsJson").value = JSON.stringify(settings, null, 2);
}

function toggleVision() {
    const toggle = document.getElementById('visionToggle');
    toggle.classList.toggle('active');
    const useVision = toggle.classList.contains('active');
    const settings = JSON.parse(document.getElementById("settingsJson").value);
    settings.agent.use_vision = useVision;
    document.getElementById("settingsJson").value = JSON.stringify(settings, null, 2);
    showNotification(`Vision ${useVision ? 'enabled' : 'disabled'}`, 'info');
}

function toggleConfirmer() {
    const toggle = document.getElementById('confirmerToggle');
    toggle.classList.toggle('active');
}

function toggleAutoSave() {
    const toggle = document.getElementById('autoSaveToggle');
    toggle.classList.toggle('active');
    autoSaveEnabled = toggle.classList.contains('active');
    showNotification(`Auto-save ${autoSaveEnabled ? 'enabled' : 'disabled'}`, 'info');
}

function updateSpeed(value) {
    const speedMap = ["Instant", "Very Fast", "Fast", "Normal", "Human Pace", "Slow", "Very Slow", "Educational", "Glacial", "Debug"];
    document.getElementById('speedValue').textContent = speedMap[Math.floor(value / 1.1)];
    const delayMap = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10];
    sendControl('set_speed', delayMap[Math.floor(value / 1.1)]);
}

// ===== Local Storage & Data Persistence =====
function loadModels() {
    fetch('/models')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('extractionModel');
            select.innerHTML = '<option value="">None</option>';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        });
}

function autoSave() {
    if (autoSaveEnabled) {
        saveData();
    }
}

function saveData() {
    const data = {
        tasks,
        taskGroups,
        settings: JSON.parse(document.getElementById("settingsJson").value),
        llmProvider: document.getElementById("llmProvider").value,
        apiKey: document.getElementById("apiKey").value,
        modelName: document.getElementById("modelName").value,
        agentType: document.getElementById("agentType").value,
        extractionModel: document.getElementById("extractionModel").value,
        quickNotes: document.getElementById('quickNotesText').value
    };
    localStorage.setItem('browserAgentProData', JSON.stringify(data));
}

function loadSavedData() {
    const saved = localStorage.getItem('browserAgentProData');
    if (saved) {
        const data = JSON.parse(saved);
        
        tasks = data.tasks || [];
        taskGroups = data.taskGroups || [];
        
        if(data.settings) document.getElementById("settingsJson").value = JSON.stringify(data.settings, null, 2);
        if(data.llmProvider) document.getElementById("llmProvider").value = data.llmProvider;
        if(data.apiKey) document.getElementById("apiKey").value = data.apiKey;
        if(data.modelName) document.getElementById("modelName").value = data.modelName;
        if(data.agentType) document.getElementById("agentType").value = data.agentType;
        if(data.extractionModel) document.getElementById("extractionModel").value = data.extractionModel;
        if(data.quickNotes) document.getElementById('quickNotesText').value = data.quickNotes;

        renderTasks();
        renderTaskGroups();
        updateProgress();
        updateLLMSettings();
        showNotification("Data loaded from last session", "info");
    }
}

function exportData() {
    saveData(); // Ensure current state is saved
    const data = localStorage.getItem('browserAgentProData');
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `browser_agent_pro_backup_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showNotification("Data exported", "success");
}

function importData() {
    const json = document.getElementById('importDataText').value;
    if (json) {
        try {
            const data = JSON.parse(json);
            localStorage.setItem('browserAgentProData', JSON.stringify(data));
            loadSavedData();
            showNotification("Data imported successfully", "success");
            document.getElementById('exportImportOverlay').classList.remove('active');
            document.getElementById('importDataText').value = '';
        } catch (e) {
            showNotification("Invalid JSON format", "error");
        }
    }
}

// ===== Shortcuts =====
function showShortcuts() {
    document.getElementById('shortcutsOverlay').classList.add('active');
}
document.getElementById('closeShortcutsBtn').onclick = () => document.getElementById('shortcutsOverlay').classList.remove('active');

function showExportImport() {
    document.getElementById('exportImportOverlay').classList.add('active');
}
document.getElementById('closeExportImportBtn').onclick = () => document.getElementById('exportImportOverlay').classList.remove('active');

document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to run task
    if (e.ctrlKey && e.key === 'Enter') {
        sendMessage(e);
    }
    // Ctrl+N for new task
    if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        document.getElementById('newTaskInput').focus();
    }
    // Ctrl+K to search
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        document.getElementById('taskSearch').focus();
    }
    // Escape to stop or close modal
    if (e.key === 'Escape') {
        if (document.querySelector('.modal-overlay.active')) {
            document.querySelector('.modal-overlay.active').classList.remove('active');
        } else {
            stopAgent(e);
        }
    }
    // Ctrl+/ to show shortcuts
    if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        showShortcuts();
    }
});

// ===== Notifications =====
function showNotification(message, type = 'info') {
    const div = document.createElement('div');
    div.style.position = 'fixed';
    div.style.bottom = '20px';
    div.style.left = '20px';
    div.style.background = `var(--${type})`;
    div.style.color = 'white';
    div.style.padding = '10px 15px';
    div.style.borderRadius = '6px';
    div.style.fontSize = '12px';
    div.style.zIndex = '2000';
    div.style.boxShadow = 'var(--shadow-lg)';
    div.style.opacity = '0';
    div.style.transition = 'all 0.3s ease';
    div.textContent = message;
    document.body.appendChild(div);

    setTimeout(() => {
        div.style.opacity = '1';
        div.style.transform = 'translateY(-10px)';
    }, 10);

    setTimeout(() => {
        div.style.opacity = '0';
        div.style.transform = 'translateY(10px)';
        setTimeout(() => div.remove(), 300);
    }, 3000);
}
