
document.addEventListener('DOMContentLoaded', () => {
    populateConfigSections();
    loadConfig();
    loadTasks();
    setupShortcuts();
    setupErrorHandling();
    renderTasks();
    startTimer();
});

function populateConfigSections() {
    document.getElementById('agent-config-section').innerHTML = createAgentConfigSection();
    document.getElementById('advanced-config-section').innerHTML = createAdvancedConfigSection();
}

function initConfig() {
    // In a real app, you might fetch this from a server
    config = {
        llmProvider: 'openai',
        mode: 'deep',
        maxSteps: 50,
        autoSave: true,
        useVision: true,
    };
}

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

function startTask(index) {
    if (index < 0 || index >= tasks.length) return;

    if (currentTaskIndex !== -1 && currentTaskIndex < tasks.length) {
        tasks[currentTaskIndex].status = 'pending';
    }

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

function filterTasks() { renderTasks(); }

function sortTasks(criteria) {
    if (criteria === 'priority') {
        const map = { high: 0, med: 1, low: 2 };
        tasks.sort((a, b) => map[a.priority] - map[b.priority]);
    } else if (criteria === 'status') {
        const map = { active: 0, pending: 1, done: 2 };
        tasks.sort((a, b) => map[a.status] - map[b.status]);
    } else if (criteria === 'order') {
        tasks.sort((a, b) => a.order - b.order);
    }
    renderTasks();
}

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
    if(confirm('Are you sure you want to clear all tasks? This cannot be undone.')) {
        tasks = [];
        renderTasks();
        saveTasks();
    }
}

function startTimer() {
    let seconds = 0;
    setInterval(() => {
        if (isRunning && !isPaused) {
            seconds++;
            const m = Math.floor(seconds / 60).toString().padStart(2, '0');
            const s = (seconds % 60).toString().padStart(2, '0');
            document.getElementById('elapsedTime').textContent = `${m}:${s}`;
        }
    }, 1000);
}

function addTaskBefore(title) {
    if (!title) return;

    let newTask = createTaskObj(title);
    tasks.unshift(newTask);
    
    tasks.forEach((task, index) => task.order = index);
    
    startTask(0);
    
    document.getElementById('mainInput').value = '';

    renderTasks();
    saveTasks();
}

function saveOrderFromDom() {
    const taskElements = document.querySelectorAll('#taskList .task-item');
    const newOrder = Array.from(taskElements).map(el => el.dataset.id);
    
    tasks.sort((a, b) => newOrder.indexOf(a.id) - newOrder.indexOf(b.id));
    
    tasks.forEach((task, index) => task.order = index);
    
    saveTasks();
    renderTasks();
}

function loadConfig() {
    const saved = localStorage.getItem('agentConfig');
    if (saved) {
        config = JSON.parse(saved);
        // Apply config to UI
        document.getElementById('llmProvider').value = config.llmProvider;
        document.getElementById('maxStepVal').innerText = config.maxSteps;
        document.querySelector(`input[type="range"]`).value = config.maxSteps;
        document.querySelector(`input[type="checkbox"]`).checked = config.autoSave;
        document.querySelectorAll(`input[type="checkbox"]`)[1].checked = config.useVision;

    } else {
        initConfig();
    }
}
