 // ===== Particle Effects =====

 function createParticles(x, y, color = 'var(--accent-primary)') {
    for (let i = 0; i < 8; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = x + 'px';
        particle.style.top = y + 'px';
        particle.style.width = (Math.random() * 6 + 4) + 'px';
        particle.style.height = particle.style.width;
        particle.style.background = color;
        particle.style.boxShadow = `0 0 ${Math.random() * 20 + 10}px ${color}`;

        const angle = (Math.PI * 2 * i) / 8;
        const velocity = Math.random() * 3 + 2;
        particle.style.setProperty('--tx', Math.cos(angle) * velocity * 20 + 'px');
        particle.style.setProperty('--ty', Math.sin(angle) * velocity * 20 + 'px');

        document.body.appendChild(particle);

        setTimeout(() => particle.remove(), 2000);
    }
}

// ===== Ripple Effect =====
function createRipple(event) {
    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    ripple.style.left = (event.clientX - 250) + 'px';
    ripple.style.top = (event.clientY - 250) + 'px';
    document.body.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
}

// Add ripple to button clicks
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('btn')) {
        createRipple(e);
        createParticles(e.clientX, e.clientY);
    }
});