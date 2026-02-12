class MainContent extends HTMLElement {
    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    render() {
        this.shadow.innerHTML = `<h1>Main Content</h1>`;
    }
}
customElements.define('main-content', MainContent);