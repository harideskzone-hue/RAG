/**
 * VISTA UI - Telemetry & Tool Chips Module
 * Visualizes dynamic execution telemetry returned by the backend API.
 * Does NOT expose internal LLM chain-of-thought.
 */

export function renderDynamicToolChips(responseContainer, responseData) {
    const chipsRow = document.createElement("div");
    chipsRow.className = "tool-chips-row";

    const execSteps = (responseData.execution && responseData.execution.steps) ? responseData.execution.steps : [];
    
    if (execSteps.length === 0) {
        if (responseData.status === "ERROR" || responseData.status === "FAILED") {
            const chip = document.createElement("div");
            chip.className = "tool-chip";
            chip.style.borderColor = "var(--accent-red)";
            chip.style.color = "var(--accent-red)";
            chip.innerHTML = `<span class="tool-chip-badge" style="background: var(--accent-red); color: white;">FAILED</span> Request Validation / Auth`;
            chipsRow.appendChild(chip);
            responseContainer.appendChild(chipsRow);
        }
        return;
    }

    execSteps.forEach(step => {
        const chip = document.createElement("div");
        chip.className = "tool-chip";
        const isOk = step.status === "completed" || step.status === "success" || step.status === "SUCCESS";
        const userLabel = formatToolUserLabel(step.name);
        
        chip.innerHTML = `<span class="tool-chip-badge" style="${isOk ? '' : 'background: var(--accent-red); color: white;'}">${isOk ? 'COMPLETED' : 'BLOCKED'}</span> ${escapeHtml(userLabel)} <span style="opacity:0.6; font-size:10px;">(${step.latency_ms}ms)</span>`;
        
        chip.addEventListener("click", () => showToolTelemetryModal(step, responseData));
        chipsRow.appendChild(chip);
    });

    responseContainer.appendChild(chipsRow);
}

function formatToolUserLabel(name) {
    const map = {
        intent_agent: "Query Understanding",
        planner_agent: "Investigation Planner",
        vector_agent: "Person & CCTV Search",
        metadata_agent: "Camera Metadata",
        video_agent: "VLM Frame Analysis",
        evidence_agent: "Evidence Fusion",
        reasoning_agent: "Contextual Reasoning",
        guardrail_agent: "Evidence Verification",
        report_agent: "Report Generator",
        time_agent: "System Clock Tool"
    };
    return map[name] || name;
}

export function showToolTelemetryModal(tool, responseData) {
    const modal = document.getElementById("telemetry-modal");
    const title = document.getElementById("telemetry-title");
    const body = document.getElementById("telemetry-body");

    if (!modal || !body) return;

    title.innerText = `Execution Telemetry: [${tool.label || tool.name}]`;

    const evidenceCount = (responseData.evidence || []).length;
    const details = {
        "Tool/Component": tool.label || tool.name,
        "Status": (tool.status || "completed").toUpperCase(),
        "Latency": `${tool.latency_ms || 150} ms`,
        "Canonical Evidence Items": evidenceCount,
        "Trace ID": responseData.trace_id || "N/A",
        "Pipeline Decision": responseData.status === "SUCCESS" ? "Grounded Output" : "Fail-Closed Abstention"
    };

    let html = `<div style="font-family: var(--font-mono); color: var(--primary); font-weight: 600; margin-bottom: 8px;">Execution Summary</div>`;
    for (const [k, v] of Object.entries(details)) {
        html += `<div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-line); padding: 6px 0;">
            <span style="color: var(--ink-2);">${k}:</span>
            <span style="color: var(--ink); font-weight: 500; font-family: var(--font-mono);">${v}</span>
        </div>`;
    }

    body.innerHTML = html;
    modal.classList.add("open");
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
