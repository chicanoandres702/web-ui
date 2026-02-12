function onLoad()
{
    load_animated_background("animated-background");
    load_confirmation_modal("confirmationModal", "confirmationOverlay");
    load_save_group_modal("saveGroupModal", "saveGroupOverlay");
    load_edit_task_modal("editTaskModal", "editTaskOverlay");
}

function toggleSection(header) {
    const section = header.parentElement;
    const content = header.nextElementSibling;
    const toggle = header.querySelector('.section-toggle');

    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}