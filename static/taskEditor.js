/**
 * taskEditor.js
 *
 * This module provides functions for displaying a modal to edit tasks and subtasks
 * before adding them to the task queue.
 */
const TASK_EDITOR_OVERLAY_ID = 'taskEditorOverlay';

/**
 * Creates the modal overlay element.
 * @returns {HTMLElement} - The modal overlay element.
 */
function createModalOverlay() {
    const modalOverlay = document.createElement('div');
    modalOverlay.classList.add('modal-overlay', 'active');
    modalOverlay.id = TASK_EDITOR_OVERLAY_ID;
    return modalOverlay;
}

/**
 * Creates the modal element.
 * @returns {HTMLElement} - The modal element.
 */
function createModal() {
    const modal = document.createElement('div');
    modal.classList.add('modal');
    return modal;
}

/**
 * Creates the modal header element.
 * @returns {HTMLElement} - The modal header element.
 */
function createModalHeader() {
    const modalHeader = document.createElement('div');
    modalHeader.classList.add('modal-header');
    modalHeader.innerHTML = `<h2>Edit Tasks</h2><p>Modify tasks and subtasks before adding them to the queue.</p>`;
    return modalHeader;
}

/**
 * Creates the modal content element.
 * @param {Array} tasks - The tasks to be edited.
 * @returns {HTMLElement} - The modal content element.
 */
function createModalContent(tasks) {
    const modalContent = document.createElement('div');
    modalContent.classList.add('modal-content');

    const taskList = document.createElement('div');
    taskList.id = 'taskEditorList';
    taskList.innerHTML = generateTaskEditorList(tasks);
    modalContent.appendChild(taskList);

    return modalContent;
}

/**
 * Creates the modal actions element.
 * @returns {HTMLElement} - The modal actions element.
 */
function createModalActions() {
    const modalActions = document.createElement('div');
    modalActions.classList.add('modal-actions');
    modalActions.innerHTML = `
        <button class="btn btn-success btn-block" id="applyEditedTasksBtn">Apply Changes</button>
        <button class="btn btn-secondary btn-block" id="cancelEditTasksBtn">Cancel</button>
/**
 * Displays the task editor modal with the given tasks.
 * @param {Array} tasks - The tasks to be edited.
 * @param {function} renderTasks - The function to render the tasks after editing.
 */
export function showTaskEditorModal(tasks, renderTasks, applyEditedTasks) {
    // Create the modal overlay
    const modalOverlay = createModalOverlay();

    // Create the modal
    const modal = createModal();

    // Create the modal header
    const modalHeader = createModalHeader();

    // Create the modal content
    const modalContent = createModalContent(tasks);

    // Add event listeners to the buttons
    document.getElementById('cancelEditTasksBtn').addEventListener('click', closeTaskEditorModal);
    document.getElementById('applyEditedTasksBtn').addEventListener('click', () => {
        applyEditedTasks(getEditedTasks(), renderTasks);
        closeTaskEditorModal();
    });

    const modalActions = createModalActions();

    modal.appendChild(modalHeader);
    modal.appendChild(modalContent);
    modal.appendChild(modalActions);
    modalOverlay.appendChild(modal);
    document.body.appendChild(modalOverlay);
}

/**
 * Closes the task editor modal.
 */
function closeTaskEditorModal() {
    const modalOverlay = document.getElementById(TASK_EDITOR_OVERLAY_ID);

    if (modalOverlay) {
        modalOverlay.remove();
    }
}

/**
 * Generates the HTML for the task editor list.
 * @param {Array} tasks - The tasks to be edited.
 * @returns {string} - The HTML for the task editor list.
 */
function generateTaskEditorList(tasks) {
    let listHTML = '<ul>';
    tasks.forEach((task, index) => {
        listHTML += `
            <li class="task-item">
                <input type="text" value="${task.text}" data-task-id="${index}" class="task-editor-input" required>
                ${task.subtasks.map((subtask, subIndex) => `
                    <input type="text" value="${subtask.text}" data-task-id="${index}" data-subtask-id="${subIndex}" class="subtask-editor-input" required>
                `).join('')}
            </li>
        `;

    });
    listHTML += '</ul>';
    return listHTML;
}

/**
 * Retrieves the edited tasks from the modal.
 * @returns {Array} - The edited tasks.
 */
function getEditedTasks() {
    const taskInputs = document.querySelectorAll('.task-editor-input');
    const subtaskInputs = document.querySelectorAll('.subtask-editor-input');

    let tasks = [];

    taskInputs.forEach(taskInput => {
        const taskId = taskInput.dataset.taskId;

        if (!taskInput.value.trim()) {
            alert('Task text cannot be empty!');

            return;
        }
        tasks[taskId] = { text: taskInput.value, subtasks: [] };
    });

    return tasks;
}