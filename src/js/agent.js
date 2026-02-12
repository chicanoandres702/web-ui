// --- Mock Agent Control ---
function runAgent() {
    isRunning = true;
    document.getElementById('statusText').textContent = "RUNNING";
    document.getElementById('statusText').style.color = "var(--warning)";
    isPaused = false;
    showTicker("Agent started...");
}

function stopAgent() {
    isRunning = false;
    document.getElementById('statusText').textContent = "IDLE";
    document.getElementById('statusText').style.color = "var(--success)";
    showTicker("Agent stopped by user.");
}
