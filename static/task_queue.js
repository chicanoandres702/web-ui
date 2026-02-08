
// Task Queue Module
function initTaskQueue() {
    const taskSearchInput = document.getElementById('taskSearch');
    taskSearchInput.addEventListener('keyup', filterTasks);

    function filterTasks() {
        const filter = taskSearchInput.value.toLowerCase();
        renderTasks();
    }
}