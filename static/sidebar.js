class Sidebar extends HTMLElement {
    constructor() {
        super();
        this.isOpen = true; // Default state: open
    }

    connectedCallback() {
        this.render();
        this.toggleButton = this.querySelector('.sidebar-toggle');
        this.toggleButton.addEventListener('click', () => this.toggleSidebar());
    }

    toggleSidebar() {
        this.isOpen = !this.isOpen;
        this.render();

        const appContainer = document.querySelector('.app-container');
        this.classList.toggle('collapsed', !this.isOpen);
    }

    render() {
        // Ensure the toggle button is correctly oriented based on the sidebar's position
        const isRight = this.classList.contains('right');
        const borderRadius = isRight ? '8px 0 0 8px' : '0 8px 8px 0';

        this.innerHTML = `
            <div class="sidebar-toggle" style="border-radius: ${borderRadius};">
                ${this.isOpen ? (isRight ? '❮' : '❯') : (isRight ? '❯' : '❮')}
            </div>
            ${this.innerHTML}
        `;
    }
}

customElements.define('custom-sidebar', Sidebar);