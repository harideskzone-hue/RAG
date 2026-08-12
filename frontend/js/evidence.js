export function renderEvidence(evidenceList) {
    const chatBox = document.getElementById("chat-box");
    
    if (evidenceList && evidenceList.length > 0) {
        const evidenceContainer = document.createElement("div");
        evidenceContainer.style.display = "flex";
        evidenceContainer.style.flexDirection = "column";
        evidenceContainer.style.gap = "8px";
        evidenceContainer.style.marginTop = "8px";
        evidenceContainer.style.alignSelf = "flex-start";
        evidenceContainer.style.width = "85%";
        
        evidenceList.forEach(ev => {
            const card = document.createElement("div");
            card.style.background = "var(--bg-panel)";
            card.style.padding = "12px";
            card.style.borderRadius = "var(--radius-md)";
            card.style.border = "1px solid var(--glass-border)";
            card.className = "slide-in";
            
            card.innerHTML = `
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Source: ${ev.camera_id || 'Unknown'} | Timestamp: ${ev.timestamp || 'N/A'}</div>
                <div style="font-size: 0.85rem; color: var(--text-main); border-left: 2px solid var(--info); padding-left: 8px;">
                    ${ev.description || 'No description available.'}
                </div>
            `;
            evidenceContainer.appendChild(card);
        });
        
        chatBox.appendChild(evidenceContainer);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}
