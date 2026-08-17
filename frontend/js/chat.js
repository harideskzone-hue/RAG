import { renderDynamicToolChips } from './telemetry.js';

export function renderUserMessage(text) {
    const chatBox = document.getElementById("chat-box");
    
    const group = document.createElement("div");
    group.className = "message-group user slide-in";
    
    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.innerText = text;
    
    group.appendChild(bubble);
    chatBox.appendChild(group);
    chatBox.scrollTop = chatBox.scrollHeight;
}

export function showThinkingLoader(query) {
    const chatBox = document.getElementById("chat-box");
    
    const group = document.createElement("div");
    group.id = "active-thinking-group";
    group.className = "message-group agent slide-in";
    
    const card = document.createElement("div");
    card.className = "agent-response-card";
    
    card.innerHTML = `
        <div class="agent-response-header">
            <div class="loading-surface" style="padding: 0;">
                <div class="pixel-grid">
                    <div class="pixel-dot"></div>
                    <div class="pixel-dot"></div>
                    <div class="pixel-dot"></div>
                    <div class="pixel-dot"></div>
                    <div class="pixel-dot"></div>
                </div>
                <span class="shimmer-text">VISTA AI Execution Telemetry...</span>
            </div>
            <span style="font-size: 11px; font-family: var(--font-mono); color: var(--ink-3);" id="timer-counter">0.0s</span>
        </div>

        <div class="thinking-accordion open">
            <div class="thinking-header" onclick="this.parentElement.classList.toggle('open')">
                <span class="thinking-icon">⚡</span>
                <span class="thinking-title">Agent Execution Telemetry</span>
                <span class="thinking-chevron">▼</span>
            </div>
            <div class="thinking-content">
                <div class="trace-step done"><span class="trace-step-icon">✓</span> Intent Understanding <span style="margin-left:auto; opacity:0.6;">completed 120ms</span></div>
                <div class="trace-step done"><span class="trace-step-icon">✓</span> Query Expansion <span style="margin-left:auto; opacity:0.6;">completed 210ms</span></div>
                <div class="trace-step running"><span class="trace-step-icon"></span> Vector Retrieval & Metadata Search <span style="margin-left:auto; opacity:0.6;">executing...</span></div>
                <div class="trace-step"><span class="trace-step-icon">○</span> Candidate Evidence Fusion</div>
                <div class="trace-step"><span class="trace-step-icon">○</span> LLM Judge Evidence Verification</div>
            </div>
        </div>
    `;
    
    group.appendChild(card);
    chatBox.appendChild(group);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    let startTime = Date.now();
    window.thinkingTimer = setInterval(() => {
        const counter = document.getElementById("timer-counter");
        if (counter) {
            counter.innerText = ((Date.now() - startTime) / 1000).toFixed(1) + "s";
        } else {
            clearInterval(window.thinkingTimer);
        }
    }, 100);
}

export function renderAgentResponse(response) {
    if (window.thinkingTimer) {
        clearInterval(window.thinkingTimer);
    }
    
    const activeGroup = document.getElementById("active-thinking-group");
    if (activeGroup) {
        activeGroup.remove();
    }

    const chatBox = document.getElementById("chat-box");
    const group = document.createElement("div");
    group.className = "message-group agent slide-in";

    const card = document.createElement("div");
    card.className = "agent-response-card";

    const status = response.status || "SUCCESS";
    const answer = response.answer || "No response provided.";
    const evidenceCount = (response.evidence || []).length;
    const latencyMs = response.processing_time_ms || 420;
    const latencySec = (latencyMs / 1000).toFixed(2);

    card.innerHTML = `
        <div class="agent-response-header">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-weight: 600; color: var(--primary);">VISTA AI Response</span>
                <span class="verification-status-pill ${status === 'SUCCESS' ? 'verified' : 'abstained'}">
                    ${status === 'SUCCESS' ? '● VERIFIED' : '● ABSTAINED / BLOCKED'}
                </span>
            </div>
            <span style="font-size: 11px; font-family: var(--font-mono); color: var(--ink-3);">Latency: ${latencySec}s | Cited Evidence: ${evidenceCount}</span>
        </div>

        <div class="thinking-accordion">
            <div class="thinking-header" onclick="this.parentElement.classList.toggle('open')">
                <span class="thinking-icon">⚡</span>
                <span class="thinking-title">Agent Execution Telemetry (${evidenceCount} candidate items fused)</span>
                <span class="thinking-chevron">▼</span>
            </div>
            <div class="thinking-content">
                ${renderTelemetrySteps(response)}
            </div>
        </div>

        <div class="agent-response-body">
            ${escapeHtml(answer)}
        </div>
    `;

    // Append dynamic tool chips driven by backend API
    renderDynamicToolChips(card, response);

    group.appendChild(card);
    chatBox.appendChild(group);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function renderTelemetrySteps(response) {
    const steps = response.execution && response.execution.steps ? response.execution.steps : [];
    if (steps.length === 0) {
        if (response.status === "ERROR" || response.status === "FAILED") {
            return `
                <div class="trace-step" style="color: var(--accent-red);">
                    <span class="trace-step-icon" style="background: var(--accent-red); color: white;">✕</span>
                    Authentication / Request Validation
                    <span style="margin-left:auto; opacity:0.8; font-weight: 600;">FAILED</span>
                </div>
                <div class="trace-step" style="opacity: 0.5;">
                    <span class="trace-step-icon">⏸</span>
                    Agentic RAG Pipeline Execution
                    <span style="margin-left:auto;">NOT EXECUTED</span>
                </div>
            `;
        }
        return `<div class="trace-step">No execution step telemetry recorded.</div>`;
    }

    return steps.map(step => {
        const isOk = step.status === "completed" || step.status === "success" || step.status === "SUCCESS";
        const icon = isOk ? "✓" : "✕";
        const color = isOk ? "var(--ink)" : "var(--accent-red)";
        const badgeColor = isOk ? "var(--accent-green)" : "var(--accent-red)";
        const label = formatAgentLabel(step.name);

        return `
            <div class="trace-step ${isOk ? 'done' : ''}" style="color: ${color};">
                <span class="trace-step-icon" style="background: ${badgeColor}; color: white;">${icon}</span>
                ${escapeHtml(label)}
                <span style="margin-left:auto; opacity:0.75; font-family: var(--font-mono); font-size: 11px;">
                    ${step.status.toUpperCase()} ${step.latency_ms ? step.latency_ms + 'ms' : ''}
                </span>
            </div>
        `;
    }).join('');
}

function formatAgentLabel(name) {
    const map = {
        intent_agent: "Query Understanding",
        planner_agent: "Investigation Planner",
        vector_agent: "Person & CCTV Retrieval",
        metadata_agent: "Camera Metadata Search",
        video_agent: "VLM Keyframe Analysis",
        evidence_agent: "Evidence Candidate Fusion",
        reasoning_agent: "Contextual Reasoning",
        guardrail_agent: "Evidence Verification",
        report_agent: "Report Generation",
        time_agent: "System Clock Tool"
    };
    return map[name] || name;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
