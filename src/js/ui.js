import { state, updateState } from '../../static/app/state.js';
import { 
    addNewTask, 
    deleteTask, 
    editTask, 
    saveTask, 
    cancelEdit,
    addSubtask,
    toggleSubtaskStatus,
    deleteSubtask,
    toggleTaskExpand,
    clearAllTasks,
    sortTasks,
    toggleTaskSelection,
    selectAllTasks,
    deselectAll,
    batchSetPriority,
    batchDelete,
    setFilter,
    loadTemplate,
    loadTaskGroup,
    deleteTaskGroup,
    saveData
} from './events.js';

// ===== UI Updates =====
export function updateTicker(type, text) {
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

export function updateAgentStatus(status) {
    const statusSpan = document.querySelector('.status-indicator span');
    if (statusSpan) statusSpan.textContent = status;
    
    const overlay = document.getElementById('agentStatusDisplay');
    if (overlay) {
        const isActive = status.includes('Participating') || status.includes('Reviewing');
        overlay.textContent = isActive ? '' : status;
    }
}

export function updateProgress() {
    const total = state.tasks.length;
    const completed = state.tasks.filter(t => t.status === 'completed').length;
    const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    document.getElementById('monitorProgress').textContent = `${progress}%`;
    document.getElementById('monitorTasks').textContent = `${completed}/${total}`;
    document.getElementById('taskProgressFill').style.width = `${progress}%`;
}

export function resetControls() {
    document.getElementById("sendBtn").disabled = false;
    document.getElementById("sendBtn").innerHTML = '▶️ Run';
    document.getElementById("stopBtn").disabled = true;
}

export function updateTaskQueue(plan) {
    if (!plan || plan.length === 0) return;

    const newTasks = plan.map((item, index) => ({
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
    updateState({ tasks: newTasks });

    renderTasks();
    updateProgress();
}


export function renderTasks() {
    const list = document.getElementById('taskList');
    list.innerHTML = '';

    let filteredTasks = filterTasksByStatus(state.tasks);
    
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
    div.className = `task-item ${task.status} ${task.subtasks.length > 0 ? 'has-subtasks' : ''} ${state.selectedTasks.has(task.id) ? 'selected' : ''}`;
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
        <div class="task-checkbox ${state.selectedTasks.has(task.id) ? 'checked' : ''}"></div>
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
            <button class="btn btn-xs tooltip edit-btn" data-tooltip="Edit">✏️</button>
            <button class="btn btn-xs tooltip subtask-btn" data-tooltip="Subtask">➕</button>
            <button class="btn btn-xs btn-error tooltip delete-btn" data-tooltip="Delete">✕</button>
        </div>
        ${task.subtasks.length > 0 || task.notes ? `<span class="task-expand ${task.expanded ? 'expanded' : ''}">▼</span>` : ''}
    `;
    div.appendChild(mainRow);
    
    // Add event listeners
    mainRow.querySelector('.task-checkbox').addEventListener('click', (e) => toggleTaskSelection(e, task.id));
    mainRow.querySelector('.edit-btn').addEventListener('click', (e) => editTask(e, task.id));
    mainRow.querySelector('.subtask-btn').addEventListener('click', (e) => addSubtask(e, task.id));
    mainRow.querySelector('.delete-btn').addEventListener('click', (e) => deleteTask(e, task.id));
    const expander = mainRow.querySelector('.task-expand');
    if (expander) {
        expander.addEventListener('click', (e) => toggleTaskExpand(e, task.id));
    }


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
                        <button class="btn btn-xs toggle-subtask-btn">✓</button>
                        <button class="btn btn-xs btn-error delete-subtask-btn">✕</button>
                    </div>
                `;
                subtaskDiv.querySelector('.toggle-subtask-btn').addEventListener('click', (e) => toggleSubtaskStatus(e, task.id, subIndex));
                subtaskDiv.querySelector('.delete-subtask-btn').addEventListener('click', (e) => deleteSubtask(e, task.id, subIndex));

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

function filterTasksByStatus(tasksToFilter) {
    if (state.currentFilter === 'all') {
        return tasksToFilter;
    }
    return tasksToFilter.filter(t => t.status === state.currentFilter || (state.currentFilter === 'done' && (t.status === 'completed' || t.status === 'failed')));
}

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
        
        const taskId = parseInt(currentElement.dataset.taskId);
        const taskIndex = state.tasks.findIndex(t => t.id === taskId);
        const [movedTask] = state.tasks.splice(taskIndex, 1);
        
        if(afterElement) {
            const afterId = parseInt(afterElement.dataset.taskId);
            const afterIndex = state.tasks.findIndex(t => t.id === afterId);
            state.tasks.splice(afterIndex, 0, movedTask);
        } else {
            state.tasks.push(movedTask);
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
export function showConfirmationDialog(data) {
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

export function showSaveGroupDialog() {
    document.getElementById('saveGroupOverlay').classList.add('active');
}

export function renderTaskGroups() {
    const list = document.getElementById('taskGroupsList');
    list.innerHTML = '';
    if (state.taskGroups.length === 0) {
        list.innerHTML = '<div style="text-align: center; font-size: 10px; color: var(--text-muted); padding: 5px 0;">No saved groups</div>';
    } else {
        state.taskGroups.forEach((group, index) => {
            const div = document.createElement('div');
            div.className = 'task-group-item';
            div.innerHTML = `
                <div>
                    <span class="task-group-name">${group.name}</span>
                    <span class="task-group-count">${group.tasks.length} tasks</span>
                </div>
                <div class="task-group-actions">
                    <button class="btn btn-xs load-group-btn" data-index="${index}">Load</button>
                    <button class="btn btn-xs btn-error delete-group-btn" data-index="${index}">✕</button>
                </div>
            `;
            list.appendChild(div);
        });
        list.querySelectorAll('.load-group-btn').forEach(btn => {
            btn.addEventListener('click', (e) => loadTaskGroup(e.target.dataset.index));
        });
        list.querySelectorAll('.delete-group-btn').forEach(btn => {
            btn.addEventListener('click', (e) => deleteTaskGroup(e.target.dataset.index));
        });
    }
}

// ===== Sidebar, Panel & Notes Toggles =====
export function toggleSidebar(side) {
    const sidebar = document.getElementById(side === 'left' ? 'leftSidebar' : 'rightSidebar');
    sidebar.classList.toggle('collapsed');
}

export function toggleSection(headerElement) {
    const section = headerElement.closest('.collapsible-section');
    const content = section.querySelector('.section-content');
    const toggle = section.querySelector('.section-toggle');
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

export function toggleFloatingPanel() {
    const panel = document.getElementById('floatingControls');
    panel.classList.toggle('minimized');
    document.getElementById('floatingToggleIcon').textContent = panel.classList.contains('minimized') ? '＋' : '−';
}

export function toggleQuickNotes() {
    document.getElementById('quickNotes').classList.toggle('hidden');
}

export function updateBatchBar() {
    const bar = document.getElementById('batchOperations');
    const count = state.selectedTasks.size;
    document.getElementById('selectedCount').textContent = count;
    
    if (count > 0) {
        bar.classList.add('active');
    } else {
        bar.classList.remove('active');
    }
}

// ===== Notifications =====
export function showNotification(message, type = 'info') {
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

// ===== Session Timer =====
export function startSessionTimer() {
    updateState({ sessionStartTime: Date.now() });
    const timerInterval = setInterval(() => {
        const elapsed = Date.now() - state.sessionStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('sessionTimer').textContent = 
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
    updateState({ timerInterval });
}