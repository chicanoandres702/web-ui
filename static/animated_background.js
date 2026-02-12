class AnimatedBackground extends HTMLElement {
    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    render() {
        this.shadow.innerHTML = `<h1>Animated Background</h1>`;
    }
}
customElements.define('animated-background', AnimatedBackground);