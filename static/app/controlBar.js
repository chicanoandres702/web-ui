function initControlBar(ws, state, utils,test = false) {

    const messageText = document.getElementById('messageText');
    const sendBtn = document.getElementById('sendBtn');
    const stopBtn = document.getElementById('stopBtn');
    const pauseNextToggle = document.getElementById('pauseNextToggle');
    const speedSlider = document.getElementById('speedSlider');
    const undoBtn = document.getElementById('undoBtn');
    const redoBtn = document.getElementById('redoBtn');

    function saveToHistory() {
        state.taskHistory = state.taskHistory.slice(0, state.historyIndex + 1);
        state.taskHistory.push(JSON.stringify(state.tasks));
        updateHistoryButtons();
    }

}
    function updateHistoryButtons() {
        undoBtn.disabled = state.historyIndex <= 0;
        redoBtn.disabled = state.historyIndex >= state.taskHistory.length - 1;
    }

    function undo() {
        if (state.historyIndex > 0) {
            state.historyIndex--;
            state.tasks = JSON.parse(state.taskHistory[state.historyIndex]);
            utils.renderTasks();
            utils.updateProgress();
            utils.logToConsole('info', 'Undo performed');
            utils.showNotification('Undo performed', 'info');
            updateHistoryButtons();
        }
    }

    function redo() {
        if (state.historyIndex < state.taskHistory.length - 1) {
            state.historyIndex++;
            state.tasks = JSON.parse(state.taskHistory[state.historyIndex]);
            utils.renderTasks();
            utils.updateProgress();
            utils.logToConsole('info', 'Redo performed');
            utils.showNotification('Redo performed', 'info');
            updateHistoryButtons();
        }
    }

    function sendMessage(event) {
        event.preventDefault();
        const task = messageText.value.trim();
        
        if (!task) {
            utils.logToConsole('warning', 'No task entered');
            utils.showNotification('Please enter a task', 'warning');
            return;
        }

        // state.stepCount = 0; // Managed by main script
        // document.getElementById('stepCounter').textContent = '0';

        const agentType = document.getElementById("agentType").value;
        const settings = JSON.parse(document.getElementById("settingsJson").value);

        utils.logToConsole('info', `Starting task: ${task}`);
        document.getElementById('currentTaskDisplay').textContent = task;

        ws.send(JSON.stringify({
            action: "run",
            task: task,
            agent_type: agentType,
            ...settings
        }));

        sendBtn.disabled = true;
        sendBtn.innerHTML = '⏳';
        stopBtn.disabled = false;
        messageText.value = '';
    }

    function stopAgent(event) {
        event.preventDefault();
        ws.send(JSON.stringify({action: "stop"}));
        resetControls();
        utils.logToConsole('warning', 'Agent stopped by user');
        utils.showNotification('Agent stopped', 'warning');
    }

    function sendControl(command, value = null) {
        ws.send(JSON.stringify({
            action: "control",
            command: command,
            value: value
        }));
        utils.logToConsole('info', `Control command: ${command}`);
    }

    function resetControls() {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '▶️ Run';
        stopBtn.disabled = true;
    }

    function togglePauseNext() {
        const isActive = pauseNextToggle.classList.toggle('active');
        sendControl('pause_on_next_step', isActive);
        utils.logToConsole('info', isActive ? 'Will pause after next step' : 'Pause cancelled');
        utils.showNotification(isActive ? 'Will pause after next step' : 'Pause cancelled', 'info');
    }

    function updateSpeed(value) {
        const speeds = ['Instant', 'V.Fast', 'Fast', 'Quick', 'Moderate', 'Human', 'Careful', 'Deliberate', 'Slow', 'V.Slow', 'Ultra'];
        document.getElementById('speedValue').textContent = speeds[value];
        sendControl('update_speed', value);
    }
    
    document.querySelector('form.main-input-group').addEventListener('submit', sendMessage);
    stopBtn.addEventListener('click', stopAgent);
    undoBtn.addEventListener('click', undo);
    redoBtn.addEventListener('click', redo);
    document.querySelector('button[onclick="undo()"]').onclick = undo;
    document.querySelector('button[onclick="redo()"]').onclick = redo;

    pauseNextToggle.addEventListener('click', togglePauseNext);
    speedSlider.addEventListener('input', (e) => updateSpeed(e.target.value));
    
    const controlButtons = document.querySelectorAll('.control-section button[onclick^="sendControl"]');
    controlButtons.forEach(btn => {
        const command = btn.getAttribute('onclick').match(/'(.*?)'/)[1];
        btn.addEventListener('click', () => sendControl(command));
        btn.removeAttribute('onclick');
    });

    document.querySelector('.history-controls button[onclick="undo()"]').onclick = undo;
    document.querySelector('.history-controls button[onclick="redo()"]').onclick = redo;

    // Expose functions that might be needed by other modules
    return {
        resetControls,
        saveToHistory,
        sendControl,
        stopAgent
    };
}