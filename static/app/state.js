 // ===== State Management =====

 const ws = new WebSocket("ws://" + window.location.host + "/ws");
 let stepCount = 0;
 let tasks = [];
 let taskGroups = [];
 let currentFilter = 'all';
 let currentEditTaskId = null;
 let autoSaveEnabled = true;
 let consoleEntries = [];
 let minimapActive = false;