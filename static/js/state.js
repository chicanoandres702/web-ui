// --- State Management ---
let tasks = [];
let config = {};
let draggingId = null;
let isRunning = false;
let currentTaskIndex = -1; // Index of the currently running task
let isPaused = false;
