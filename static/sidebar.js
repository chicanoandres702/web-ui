class Sidebar extends HTMLElement {
    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
        this.isOpen = true; // Default state: open
    }

    connectedCallback() {
        this.render();
        this.toggleButton = this.shadow.querySelector('.sidebar-toggle');
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

        this.shadow.innerHTML = `
            <style>
            .sidebar-toggle {
                position: absolute;
                top: 50%;
                right: -11px;
                transform: translateY(-50%);
                width: 24px;
                height: 60px;
                background: var(--accent-primary);
                border-radius: 0 8px 8px 0;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                z-index: 101;
                transition: all 0.3s ease;
                font-size: 12px;
                color: white;
                box-shadow: var(--shadow);
            }
            </style>
            <div class="sidebar-toggle" style="border-radius: ${borderRadius};">
                ${this.isOpen ? (isRight ? '❮' : '❯') : (isRight ? '❯' : '❮')}
            </div>
        `;
    }

    static get observedAttributes() {
        return ['side', 'title'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue) {
            this.render();
        }
    }

}

customElements.define('side-bar', Sidebar);