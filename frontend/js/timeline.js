export function renderTimeline(trace) {
    // In a real application, trace data would come from the API payload (e.g. metadata.execution_path)
    // Here we simulate the agent progression for UI demonstration.
    const items = document.querySelectorAll('.timeline-item');
    
    // Reset
    items.forEach(item => item.classList.remove('active'));
    
    // Animate completion
    let delay = 0;
    items.forEach((item, index) => {
        setTimeout(() => {
            item.classList.add('active');
            item.innerHTML = `<div class="timeline-icon">✓</div> ${item.textContent}`;
        }, delay);
        delay += 600; // Fake 600ms latency per agent
    });
}

export function resetTimeline() {
    const items = document.querySelectorAll('.timeline-item');
    const labels = ["Intent Analysis", "Planner", "Metadata Execution", "Vector Search", "Evidence Collation"];
    
    items.forEach((item, index) => {
        item.classList.remove('active');
        item.innerHTML = `<div class="timeline-icon"></div> ${labels[index]}`;
    });
}
