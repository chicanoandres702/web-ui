function createAgentConfigSection() {
    return `
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
                        <button class="btn btn-sm btn-secondary" style="flex:1" onclick="setMode('fast')" title="Fast Mode">Fast</button>
                        <button class="btn btn-sm btn-secondary active" style="flex:1" style="border-color: var(--accent-primary);" title="Deep Mode">Deep</button>
                    </div>
                </div>
            </div>
        </div>
    `;
}
function createAdvancedConfigSection() {
    return `
       <div class="config-section">
            <div class="section-header" onclick="toggleSection('adv-config')">
                <span class="section-title" id="advancedControlTitle">Advanced Control</span>
                <span>▼</span>
            </div>
            <div id="adv-config" class="section-content collapsed">
    `;
}


function renderTaskList() {
   const filter = document.getElementById('taskSearch').value.toLowerCase();
   const list = document.getElementById('taskList');
   if (!list) {
        console.warn('Task list element not found.');
        return;
    }
    list.innerHTML = '';
     tasks.forEach((task, index) => {
        if (!task.title.toLowerCase().includes(filter)) return;
    
        const el = document.createElement('div');
        el.className = `task-item ${task.status} ${task.expanded ? 'expanded' : ''} ${task.status === 'active' ? 'active' : ''}`;
        el.draggable = true;
        el.dataset.id = task.id;
        
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
