export function renderBubble(text, sender = "agent") {
    const chatBox = document.getElementById("chat-box");
    
    // Remove thinking indicator if present
    const thinking = document.getElementById("thinking-bubble");
    if (thinking && sender === "agent") {
        thinking.remove();
    }
    
    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender} slide-in`;
    bubble.innerText = text;
    
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}

export function showThinking() {
    const chatBox = document.getElementById("chat-box");
    const bubble = document.createElement("div");
    bubble.id = "thinking-bubble";
    bubble.className = "bubble agent slide-in";
    bubble.innerHTML = `
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}
