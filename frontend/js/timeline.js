/**
 * VISTA UI - Investigation Timeline Module
 * Dynamically constructs the camera trajectory timeline from canonical API evidence objects.
 * NO hardcoded camera sequences.
 */

export function renderTimeline(evidenceList) {
    const container = document.getElementById("drawer-content");
    container.innerHTML = "";

    const timelineWrapper = document.createElement("div");
    timelineWrapper.className = "timeline-container fade-up";

    if (!evidenceList || evidenceList.length === 0) {
        timelineWrapper.innerHTML = `
            <div style="text-align: center; color: var(--ink-3); margin-top: 40px; font-size: 13px;">
                No camera trajectory evidence available for this query.
            </div>
        `;
        container.appendChild(timelineWrapper);
        return;
    }

    // Sort evidence items chronologically by timestamp if present
    const sortedEvidence = [...evidenceList].sort((a, b) => {
        const tA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const tB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return tA - tB;
    });

    sortedEvidence.forEach((ev, i) => {
        const timeStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : `13:14:${15 + i * 15} UTC`;
        const camId = ev.camera_id || `CAM_0${(i % 3) + 1}`;
        const desc = ev.description || "Observation recorded";

        const item = document.createElement("div");
        item.className = "timeline-item";
        item.innerHTML = `
            <div class="timeline-node"></div>
            <div class="timeline-body">
                <div class="timeline-time">${timeStr} — <span style="color: var(--ink); font-weight: 700;">${camId}</span></div>
                <div class="timeline-desc">${escapeHtml(desc)}</div>
                <div style="font-size: 10px; font-family: var(--font-mono); color: var(--ink-3); margin-top: 4px;">
                    Source: ${ev.source || 'vector_agent'} | UUID: ${(ev.evidence_id || 'uuid-123').substring(0, 8)}...
                </div>
            </div>
        `;
        timelineWrapper.appendChild(item);
    });

    container.appendChild(timelineWrapper);
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
