function load_confirmation_modal(modal_id, overlay_id)
{
    const confirmationModal = document.getElementById(modal_id);
    const confirmationOverlay = document.getElementById(overlay_id);

    confirmationModal.innerHTML = `            <div class="modal-header">
    <h2>🤖 Action Confirmation</h2>
    <p>Review the agent's next planned action</p>
</div>
<div class="modal-content"></div>`
}


function load_save_group_modal(modal_id, overlay_id)
{
    const saveGroupModal = document.getElementById(modal_id);
    const saveGroupOverlay = document.getElementById(overlay_id);

    saveGroupModal.innerHTML = `
        <div class="modal-header">
            <h2>📁 Save Task Group</h2>
            <p>Save the current sequence of tasks as a reusable template</p>
        </div>
        <div class="modal-body">
            <div class="form-row">
                <label class="form-label">Group Name</label>
                <input type="text" id="saveGroupName" placeholder="e.g., Daily Research Workflow">
            </div>
            <div class="form-row">
                <label class="form-label">Description</label>
                <textarea id="saveGroupDescription" rows="2" placeholder="What does this group of tasks accomplish?"></textarea>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="closeModal('${modal_id}')">Cancel</button>
            <button class="btn" id="confirmSaveGroup">Save Template</button>
        </div>`;
}

function load_edit_task_modal(modal_id, overlay_id)
{
    const editTaskModal = document.getElementById(modal_id);
    const editTaskOverlay = document.getElementById(overlay_id);

    editTaskModal.innerHTML = `
        <div class="modal-header">
            <h3>✏️ Edit Task</h3>
            <button class="btn-icon btn-ghost" onclick="closeModal('${modal_id}')">✕</button>
        </div>
        <div class="modal-body">
            <input type="hidden" id="editTaskId">
            <div class="form-row">
                <label class="form-label">Task Title</label>
                <input type="text" id="editTaskTitle" placeholder="Enter task description...">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                <div>
                    <label class="form-label">Priority</label>
                    <select id="editTaskPriority">
                        <option value="high">🔴 High</option>
                        <option value="med">🟡 Medium</option>
                        <option value="low">🟢 Low</option>
                    </select>
                </div>
                <div>
                    <label class="form-label">Est. Time (min)</label>
                    <input type="number" id="editTaskTime" value="5">
                </div>
            </div>
            <div class="form-row">
                <label class="form-label">Notes / Context</label>
                <textarea id="editTaskNotes" rows="4" placeholder="Add extra context, URLs, or requirements..."></textarea>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="closeModal('')">Cancel</button>
            <button class="btn" onclick="saveTaskFromModal()">Save Task</button>
        </div>`;
}