/**
 * taskManager.js
 *
 * This module provides functions for managing tasks, including reordering,
 * locking, and interacting with an LLM for task organization suggestions.
 */

/**
 * Updates the order of tasks based on the current state of the task list.
 */
export function updateTaskOrder(tasks) {
    const taskList = document.getElementById('taskList');
    const newTaskOrder = Array.from(taskList.children).map(item => {
        const taskId = parseInt(item.querySelector('.task-text').textContent.split('.')[0]) - 1; // Extract task ID
        return tasks.find(task => task.id === taskId);
    }).filter(task => task !== undefined);

    return newTaskOrder.map((task, index) => ({
        ...task,
        id: index,
        text: `${index + 1}. ${task.text.split('. ')[1]}`
    }));
}

/**
 * Locks or unlocks a task, preventing it from being reordered.
 * @param {Event} event - The event triggered by the lock/unlock action.
 * @param {number} taskId - The ID of the task to lock or unlock.
 */
export function lockTask(event, taskId, renderTasks) {
    event.stopPropagation(); // Prevent task expansion when clicking the checkbox
    isTaskLocked[taskId] = event.target.checked;
    renderTasks(); // Re-render to update draggable attribute
}

/**
 * Calls the llm to suggest an optimized task order
 */
export async function suggestTaskOrder(tasks, originalPrompt) {
     try {
        const response = await fetch('/suggest_task_order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tasks: tasks, prompt: originalPrompt })
        });
        return await response.json();
    } catch (error) {
        console.error("Error fetching suggested task order:", error);
        return null;
    }
}
