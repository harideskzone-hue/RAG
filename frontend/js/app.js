import { checkHealth, sendChatQuery, getDevAuthToken } from './api.js';
import { renderUserMessage, showThinkingLoader, renderAgentResponse } from './chat.js';
import { renderEvidenceCards, renderVerificationSummary } from './evidence.js';
import { renderTimeline } from './timeline.js';

let currentResponse = null;
let currentMode = "simple";

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Initial Health & Auth Check
    initHealthAndAuthCheck();

    // 2. Bind Chat Input & Send Button
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    async function handleSend(queryText) {
        const query = (queryText || chatInput.value).trim();
        if (!query) return;

        chatInput.value = "";
        
        // Render User Message
        renderUserMessage(query);

        // Show Pixel-Grid Thinking Loader
        showThinkingLoader(query);

        try {
            // Send query with execution_mode parameter to let Supervisor remain authoritative
            const response = await sendChatQuery(query, currentMode);
            currentResponse = response;

            // Render Response Card
            renderAgentResponse(response);

            // Render Evidence Cards in Drawer by default
            renderEvidenceCards(response.evidence || []);

            // Set tab active
            setActiveDrawerTab("tab-evidence-btn");

        } catch (err) {
            renderAgentResponse({
                status: "ERROR",
                answer: `Pipeline execution failed: ${err.message || 'Connection lost to supervisor'}`,
                evidence: []
            });
        }
    }

    if (sendBtn) sendBtn.addEventListener("click", () => handleSend());
    if (chatInput) {
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") handleSend();
        });
    }

    // 3. Bind Prompt Chips
    document.querySelectorAll(".prompt-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            if (q) handleSend(q);
        });
    });

    // 4. Bind Mode Switcher (Simple vs Investigation)
    const simpleBtn = document.getElementById("mode-simple-btn");
    const investBtn = document.getElementById("mode-investigation-btn");
    
    if (simpleBtn && investBtn) {
        simpleBtn.addEventListener("click", () => {
            currentMode = "simple";
            simpleBtn.classList.add("active");
            investBtn.classList.remove("active");
        });
        investBtn.addEventListener("click", () => {
            currentMode = "investigation";
            investBtn.classList.add("active");
            simpleBtn.classList.remove("active");
        });
    }

    // 5. Bind Drawer Tabs
    const tabEv = document.getElementById("tab-evidence-btn");
    const tabTime = document.getElementById("tab-timeline-btn");
    const tabVerif = document.getElementById("tab-verification-btn");

    if (tabEv) {
        tabEv.addEventListener("click", () => {
            setActiveDrawerTab("tab-evidence-btn");
            renderEvidenceCards((currentResponse && currentResponse.evidence) || []);
        });
    }
    if (tabTime) {
        tabTime.addEventListener("click", () => {
            setActiveDrawerTab("tab-timeline-btn");
            renderTimeline((currentResponse && currentResponse.evidence) || []);
        });
    }
    if (tabVerif) {
        tabVerif.addEventListener("click", () => {
            setActiveDrawerTab("tab-verification-btn");
            if (currentResponse) {
                renderVerificationSummary(currentResponse);
            } else {
                document.getElementById("drawer-content").innerHTML = `
                    <div style="text-align: center; color: var(--ink-3); margin-top: 40px;">No query executed yet.</div>
                `;
            }
        });
    }

    // 6. Bind Modal Close buttons
    const videoModal = document.getElementById("video-modal");
    const videoClose = document.getElementById("modal-video-close");
    if (videoClose && videoModal) {
        videoClose.addEventListener("click", () => videoModal.classList.remove("open"));
    }

    const telemetryModal = document.getElementById("telemetry-modal");
    const telemetryClose = document.getElementById("telemetry-close");
    if (telemetryClose && telemetryModal) {
        telemetryClose.addEventListener("click", () => telemetryModal.classList.remove("open"));
    }
});

async function initHealthAndAuthCheck() {
    const dot = document.getElementById("health-dot");
    const statusText = document.getElementById("health-status");
    const authBadge = document.getElementById("auth-dev-badge");

    const { isDevMode } = getDevAuthToken();
    if (authBadge) {
        authBadge.style.display = isDevMode ? "inline-flex" : "none";
    }

    const res = await checkHealth();
    if (res.status === "ok" || res.status === "healthy" || res.status === "SUCCESS") {
        if (dot) dot.style.background = "var(--accent-green)";
        if (statusText) statusText.innerText = "System Ready";
    } else {
        if (dot) dot.style.background = "var(--accent-amber)";
        if (statusText) statusText.innerText = "Ready (Local)";
    }
}

function setActiveDrawerTab(activeId) {
    ["tab-evidence-btn", "tab-timeline-btn", "tab-verification-btn"].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            if (id === activeId) btn.classList.add("active");
            else btn.classList.remove("active");
        }
    });
}
