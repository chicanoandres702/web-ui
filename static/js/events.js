
function setupShortcuts() {
    document.addEventListener('keydown', (e) => {

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

function setupErrorHandling() {
    window.onerror = function(message, source, lineno, colno, error) {
        console.error('Global error handler:', message, source, lineno, colno, error);
        showTicker(`Error: ${message}. See console for details.`);
        return true; // Prevent default browser error handling
    };
}

document.getElementById('mainInput').addEventListener('keydown', function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
         addTaskBefore(this.value);
     }
 });
