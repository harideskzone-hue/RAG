import { checkHealth, sendChatQuery } from './api.js';
import { renderBubble, showThinking } from './chat.js';
import { setHealthStatus } from './ui.js';
import { renderEvidence } from './evidence.js';

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Initial Health Check
    const health = await checkHealth();
    setHealthStatus(health);

    // 2. Bind Chat Interface
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    async function handleSend() {
        const query = chatInput.value.trim();
        if (!query) return;

        // Reset UI State
        chatInput.value = "";
        renderBubble(query, "user");
        
        // Show loading state
        showThinking();

        try {
            // Trigger Agent Orchestration
            const response = await sendChatQuery(query);
            
            // Render Response
            renderBubble(response.answer, "agent");
            
            // Render Evidence if present
            if (response.evidence && response.evidence.length > 0) {
                renderEvidence(response.evidence);
            }
            
        } catch (e) {
            renderBubble("Error: Connection to Supervisor lost or pipeline failed.", "agent");
            console.error(e);
        }
    }

    sendBtn.addEventListener("click", handleSend);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            handleSend();
        }
    });
});
