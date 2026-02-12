// animated_background.js

document.addEventListener('DOMContentLoaded', function() {
    const animatedBg = document.getElementById('animated-background');

    function createParticle() {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        particle.style.width = `${Math.random() * 20}px`;
        particle.style.height = `${Math.random() * 20}px`;
        particle.style.background = `rgba(59, 130, 246, ${Math.random() * 0.5})`;
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;
        animatedBg.appendChild(particle);
        setTimeout(() => particle.remove(), 2000);
    }

    setInterval(createParticle, 100);
});