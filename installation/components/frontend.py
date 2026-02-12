"""
This module contains the frontend/UI file definitions for the project setup.
"""
FRONTEND_FILES = {
    "app/static/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scholar Pro | Command Center</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root { --bg:#0b0d11; --pnl:#151921; --acc:#3b82f6; --txt:#e0e6ed; --input:#1c232d; --border:#2d3748; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; color: var(--txt); }
        body { background: var(--bg); height: 100vh; overflow: hidden; }
        .shell { display: grid; grid-template-columns: 280px 1fr 300px; height: 100vh; }
        aside { background: var(--pnl); padding: 1.2rem; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; }
        aside#right { border-right: none; border-left: 1px solid var(--border); }
        main { padding: 1.2rem; display: flex; flex-direction: column; gap: 1rem; overflow: hidden; position: relative; }
        input, select { width: 100%; background: var(--input); border: 1px solid var(--border); color: #fff; padding: 10px; border-radius: 6px; font-size: 13px; outline: none; }
        button { background: var(--acc); color: white; border: none; padding: 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }
        .viewport { flex: 1; background: #000; border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; position: relative; }
        .log-box { height: 160px; background: #050505; border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a0aec0; }
        .item { background: var(--input); padding: 10px; border-radius: 8px; margin-bottom: 8px; font-size: 11px; display: flex; align-items: center; gap: 10px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #4a5568; }
        .active .dot { background: var(--acc); box-shadow: 0 0 8px var(--acc); }
        .modal-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); display:none; align-items:center; justify-content:center; z-index: 1000; }
        .modal { background: #1e212b; padding: 30px; border-radius: 12px; width: 400px; border: 1px solid var(--acc); }
    </style>
</head>
<body>
    <div class="shell">
        <aside id="left">
            <h1 style="font-size:1.1rem; margin:0 0 0.5rem 0;">Scholar Pro</h1>
            <div style="font-size:11px; color:#a0aec0; margin-bottom:10px;">
                <span id="auth-status">Checking auth...</span> 
                <a href="/login_page" style="color:var(--acc); margin-left:5px;">Login</a>
                <a href="/auth/logout" style="color:#ef4444; margin-left:5px;">Logout</a>
            </div>
            <label>Context</label><input id="class-name" value="History 101">
            <label>Mode</label><select id="provider-mode"><option value="ollama">Ollama (Local)</option><option value="gemini">Gemini (Cloud)</option></select>
            <label>Model</label><select id="model-select"><option>Loading...</option></select>
            <label style="display:flex; align-items:center; gap:8px; margin-top:5px;"><input type="checkbox" id="headless" style="width:auto;"> Headless Mode</label>
            <div style="margin-top:auto; border-top:1px solid var(--border); padding-top:1rem;"><h3>QUEUE</h3><div id="queue-box"></div></div>
        </aside>
        <main>
            <div style="display:flex; gap:10px;"><input id="task-input" placeholder="Enter research objective..." style="margin:0;"><button id="run-btn" style="width:100px;">EXECUTE</button></div>
            <div class="viewport">
                <div style="background: #1a202c; padding: 8px 12px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid var(--border);">
                    <div style="flex: 1; background: #000; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #a0aec0;" id="url-bar">about:blank</div>
                </div>
                <div style="flex:1; display:flex; align-items:center; justify-content:center; background:#111;"><img id="live-stream" style="max-width:100%; max-height:100%; object-fit:contain;"></div>
            </div>
            <div class="log-box" id="log-box"></div>
        </main>
        <aside id="right">
            <h3>PLAN</h3><div id="plan-box" style="flex:1; overflow-y:auto; margin-top:10px;"></div>
        </aside>
    </div>
    <div class="modal-overlay" id="user-modal"><div class="modal"><h2>Approve Workflow</h2><p style="opacity:0.7; margin-bottom:20px;">Plan generated.</p><div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;"><button id="ok-btn" style="background:#10b981;">APPROVE</button><button id="no-btn" style="background:#ef4444;">CANCEL</button></div></div></div>
    <script>
        let ws, taskId, planState = [];
        async function checkAuth() {
            try {
                const r = await fetch('/api/v1/health');
                const d = await r.json();
                const statusR = await fetch('/auth/status');
                const sData = await statusR.json();
                const el = document.getElementById('auth-status');
                if(sData.is_logged_in) { el.innerText = "Authenticated"; el.style.color = "#10b981"; document.getElementById('provider-mode').value = 'gemini'; }
                else { el.innerText = "Not Logged In"; el.style.color = "#ef4444"; }
                refreshModelList();
            } catch(e) {}
        }
        async function refreshModelList() {
            const mode = document.getElementById('provider-mode').value;
            const select = document.getElementById('model-select');
            select.innerHTML = '<option>Loading...</option>';
            if (mode === 'ollama') {
                try {
                    const r = await fetch('/api/v1/ollama/models');
                    const d = await r.json();
                    select.innerHTML = (d.models || []).map(m => `<option value="${m.name}">${m.name}</option>`).join('');
                } catch(e) { select.innerHTML = '<option>Ollama Offline</option>'; }
            } else {
                select.innerHTML = '<option value="gemini-flash-latest">gemini-flash-latest</option>';
            }
        }
        async function init() {
            checkAuth();
            const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${protocol}://${location.host}/api/v1/ws/agent`);
            ws.onmessage = e => handleMsg(JSON.parse(e.data));
            document.getElementById('run-btn').onclick = async () => {
                await fetch('/api/v1/agent/run', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({
                        task: document.getElementById('task-input').value, 
                        class_name: document.getElementById('class-name').value,
                        model_override: document.getElementById('model-select').value,
                        headless: document.getElementById('headless').checked
                    })
                });
            };
            document.getElementById('ok-btn').onclick = () => feedback(true);
            document.getElementById('no-btn').onclick = () => feedback(false);
            document.getElementById('provider-mode').onchange = refreshModelList;
        }
        async function feedback(approved) {
            document.getElementById('user-modal').style.display = 'none';
            await fetch('/api/v1/agent/feedback', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({task_id: taskId, approved, modified_plan: planState})
            });
        }
        function handleMsg(m) {
            const box = document.getElementById('log-box');
            if(m.type === 'status' && m.content.status === 'waiting_for_user') { 
                taskId = m.content.task_id; document.getElementById('user-modal').style.display = 'flex'; 
            }
            if(m.type === 'browser_stream') {
                document.getElementById('live-stream').src = 'data:image/jpeg;base64,' + m.content.image;
                document.getElementById('url-bar').innerText = m.content.url;
            }
            if(m.type === 'log') { box.innerHTML += `<div>> ${m.content}</div>`; box.scrollTop = box.scrollHeight; }
            if(m.type === 'plan') {
                taskId = m.content.task_id; planState = m.content.plan;
                const b = document.getElementById('plan-box'); b.innerHTML = '';
                planState.forEach(s => b.innerHTML += `<div class="item" id="s-${s.id}"><div class="dot"></div><div style="font-size:11px;">${s.description}</div></div>`);
            }
            if(m.type === 'plan_update') { const el = document.getElementById(`s-${m.content.id}`); if(el) el.className = `item ${m.content.status}`; }
            if(m.type === 'queue_update') {
                const qb = document.getElementById('queue-box'); qb.innerHTML = '';
                m.content.forEach(t => qb.innerHTML += `<div class="item ${t.status}"><div class="dot"></div><div>${t.task_input.substring(0,18)}...</div></div>`);
            }
        }
        window.onload = init;
    </script>
</body>
</html>
'''
}
