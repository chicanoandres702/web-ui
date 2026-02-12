class TaskQueue extends HTMLElement {
    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    render() {
        this.shadow.innerHTML = `<h1>Task Queue</h1>`;
    }
}
customElements.define('task-queue', TaskQueue);