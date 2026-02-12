
let ws, taskId, planState = [];

async function loadComponent(id, file) {
    const r = await fetch(`components/${file}`);
    const html = await r.text();
    document.getElementById(id).innerHTML = html;
}

export async function init() {
    await Promise.all([
        loadComponent('left-sidebar', 'left.html'),
        loadComponent('main-view', 'main.html'),
        loadComponent('right-sidebar', 'right.html'),
        loadComponent('modal-layer', 'modal.html')
    ]);

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${protocol}://${location.host}/api/v1/ws/agent`);
    ws.onmessage = e => handleMsg(JSON.parse(e.data));

    document.getElementById('run-btn').onclick = run;
    document.getElementById('ok-btn').onclick = () => feedback(true);
    document.getElementById('no-btn').onclick = () => feedback(false);
    
    fetch('/api/v1/ollama/models').then(r => r.json()).then(d => {
        const s = document.getElementById('model-select');
        if (s && d.models) s.innerHTML = d.models.map(m => `<option value="\${m.name}">\${m.name}</option>`).join('');
    });

    window.onGoogleAuth = (r) => {
        const p = JSON.parse(atob(r.credential.split('.')[1]));
        document.getElementById('user-info').style.display = 'flex';
        document.getElementById('user-name').innerText = p.name;
        document.getElementById('user-avatar').src = p.picture;
    };
}

async function run() {
    const res = await fetch('/api/v1/agent/run', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({task: document.getElementById('task-input').value, class_name: document.getElementById('class-name').value})
    });
    taskId = await res.json();
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
    if(m.type === 'status' && m.content === 'waiting_for_user') document.getElementById('user-modal').style.display = 'flex';
    if(m.type === 'log') { box.innerHTML += `<div>> \${m.content}</div>`; box.scrollTop = box.scrollHeight; }
    if(m.type === 'plan') {
        planState = m.content;
        const b = document.getElementById('plan-box'); b.innerHTML = '';
        m.content.forEach(s => b.innerHTML += `<div class="plan-item" id="s-\${s.id}"><div class="plan-dot"></div><input value="\${s.description}" style="margin:0; border:none; background:transparent; font-size:12px; color:white; flex:1;"></div>`);
    }
    if(m.type === 'plan_update') {
        const el = document.getElementById(`s-\${m.content.id}`);
        if(el) el.className = `plan-item \${m.content.status}`;
    }
}
