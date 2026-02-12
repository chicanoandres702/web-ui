/**
 * Nexus Agent Client
 * Handles WebSocket connections, API requests, and UI state management.
 */

class AgentClient {
    constructor() {
        // Configuration
        this.apiBase = '/api/v1';
        this.wsBase = (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/api/v1/ws/agent';
        
        // State
        this.currentTaskId = null;
        this.socket = null;
        this.isProcessing = false;

        // DOM Elements
        this.elements = {
            apiKey: document.getElementById('api-key'),
            agentType: document.getElementById('agent-type'),
            taskInput: document.getElementById('task-input'),
            runBtn: document.getElementById('run-btn'),
            stopBtn: document.getElementById('stop-btn'),
            logsContainer: document.getElementById('logs-container'),
            browserView: document.getElementById('live-image'),
            placeholder: document.querySelector('.placeholder-state'),
            wsStatus: document.getElementById('ws-status'),
            wsText: document.getElementById('ws-text'),
            agentStatus: document.getElementById('agent-status'),
            agentText: document.getElementById('agent-text'),
            clearLogsBtn: document.getElementById('clear-logs'),
            stepBadge: document.getElementById('step-badge')
        };

        this.init();
    }

    init() {
        this.connectWebSocket();
        this.attachListeners();
    }

    /**
     * Establish WebSocket connection
     */
    connectWebSocket() {
        this.updateStatus('ws', 'connecting');
        
        this.socket = new WebSocket(this.wsBase);

        this.socket.onopen = () => {
            this.updateStatus('ws', 'connected');
            this.log('System', 'Connected to real-time event stream.', 'info');
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleSocketMessage(data);
        };

        this.socket.onclose = () => {
            this.updateStatus('ws', 'disconnected');
            // Reconnect logic could go here (exponential backoff)
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.socket.onerror = (err) => {
            console.error('WebSocket error:', err);
            this.updateStatus('ws', 'error');
        };
    }

    /**
     * UI Event Listeners
     */
    attachListeners() {
        this.elements.runBtn.addEventListener('click', () => this.runTask());
        this.elements.stopBtn.addEventListener('click', () => this.stopTask());
        this.elements.clearLogsBtn.addEventListener('click', () => {
            this.elements.logsContainer.innerHTML = '';
        });
        
        this.elements.taskInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isProcessing) {
                this.runTask();
            }
        });
    }

    /**
     * Start a new agent task
     */
    async runTask() {
        const task = this.elements.taskInput.value.trim();
        if (!task) return;

        const apiKey = this.elements.apiKey.value.trim();
        const agentType = this.elements.agentType.value;

        this.setProcessingState(true);
        this.log('User', `Starting task: "${task}"`, 'info');

        try {
            const headers = { 'Content-Type': 'application/json' };
            if (apiKey) headers['X-API-Key'] = apiKey;

            const response = await fetch(`${this.apiBase}/agent/run`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    task: task,
                    agent_type: agentType,
                    max_steps: 20
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentTaskId = data.task_id;
            this.updateStatus('agent', 'active', `Running Task ${data.task_id.substring(0, 6)}...`);

        } catch (error) {
            this.log('Error', error.message, 'error');
            this.setProcessingState(false);
        }
    }

    /**
     * Stop current task (stub for now, needs API support)
     */
    stopTask() {
        this.log('System', 'Stop requested (Not fully implemented on backend yet)', 'warning');
        this.setProcessingState(false);
        // Implement DELETE /agent/{id} or similar
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleSocketMessage(message) {
        // Expected format: { type: 'log'|'step'|'image', content: ... }
        
        // Since the backend 'broadcast' is currently just sending strings in the refactor,
        // we'll handle that. In a robust version, we'd parse JSON objects.
        
        // NOTE: This assumes the backend sends simple strings for now based on the previous file.
        // We'll treat it as a general log if it's a string.
        if (typeof message === 'string') {
            // Check for specific prefixes if we didn't implement full JSON yet
            this.log('Agent', message, 'info');
            return;
        }
        
        // If we upgrade the backend to send JSON (recommended):
        switch (message.type) {
            case 'step':
                this.log('Agent', `Step ${message.content.step}: ${message.content.action}`, 'success');
                this.elements.stepBadge.textContent = `Step ${message.content.step}`;
                break;
            case 'log':
                this.log('Agent', message.content, 'info');
                break;
            case 'image':
                this.updateBrowserImage(message.content);
                break;
            case 'result':
                this.log('System', `Task Complete: ${message.content}`, 'success');
                this.setProcessingState(false);
                break;
            case 'error':
                this.log('Error', message.content, 'error');
                this.setProcessingState(false);
                break;
        }
    }

    /**
     * Helper: Update Status Indicators
     */
    updateStatus(type, status, text = null) {
        const dot = this.elements[`${type}Status`];
        const label = this.elements[`${type}Text`];
        
        dot.className = `status-dot ${status}`;
        
        if (text) {
            label.textContent = text;
        } else {
            // Defaults
            if (type === 'ws') {
                label.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
            } else if (type === 'agent') {
                label.textContent = status === 'active' ? 'Running' : 'Idle';
            }
        }
    }

    /**
     * Helper: Toggle UI state during processing
     */
    setProcessingState(isRunning) {
        this.isProcessing = isRunning;
        this.elements.runBtn.disabled = isRunning;
        this.elements.stopBtn.disabled = !isRunning;
        this.elements.taskInput.disabled = isRunning;
        this.elements.agentType.disabled = isRunning;
        
        if (isRunning) {
            this.elements.placeholder.style.display = 'none';
            this.elements.browserView.style.display = 'block';
        } else {
            this.updateStatus('agent', 'idle');
        }
    }

    /**
     * Helper: Add log entry to UI
     */
    log(source, message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        
        const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        entry.innerHTML = `
            <span class="log-time">[${time}]</span>
            <strong>${source}:</strong> ${message}
        `;
        
        this.elements.logsContainer.appendChild(entry);
        this.elements.logsContainer.scrollTop = this.elements.logsContainer.scrollHeight;
    }

    /**
     * Helper: Update Browser Screenshot
     */
    updateBrowserImage(base64Data) {
        this.elements.browserView.src = `data:image/png;base64,${base64Data}`;
        this.elements.browserView.style.display = 'block';
        this.elements.placeholder.style.display = 'none';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AgentClient();
});