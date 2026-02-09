import { updateState } from './state.js';
import { 
    updateTaskQueue, 
    updateTicker, 
    updateProgress, 
    resetControls, 
    showConfirmationDialog, 
    updateAgentStatus,
    showNotification
} from './ui.js';
import { loadModels, loadSavedData } from './main.js';

export function initWebSocket() {
    const ws = new WebSocket("ws://" + window.location.host + "/ws");
    updateState({ ws });

    ws.onopen = function() {
        console.log("🔌 Connected");
        loadModels();
        loadSavedData();
        // The session timer start is a UI function, will move it later
        // startSessionTimer(); 
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
}

function handleMessage(data) {
    switch(data.type) {
        case 'plan_update':
            updateTaskQueue(data.plan);
            break;
        case 'log':
            updateTicker('log', data.content);
            break;
        case 'step':
            updateState({ stepCount: state.stepCount + 1 });
            document.getElementById('stepCounter').textContent = `${state.stepCount} Steps`;
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
