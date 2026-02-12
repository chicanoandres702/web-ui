﻿class ControlBar extends HTMLElement {
    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    render() {
        this.shadow.innerHTML = `<h1>Control Bar</h1>`;
    }
}
customElements.define('control-bar', ControlBar);