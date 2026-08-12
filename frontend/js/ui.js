// ui.js - General UI utilities

export function setHealthStatus(statusInfo) {
    const statusDot = document.querySelector("#health-status").previousElementSibling;
    const statusText = document.getElementById("health-status");
    const modeText = document.getElementById("mode-status");
    
    if (statusInfo.status === "healthy") {
        statusDot.style.background = "var(--success)";
        statusDot.style.boxShadow = "0 0 8px var(--success)";
        statusText.innerText = "Healthy";
    } else {
        statusDot.style.background = "var(--danger)";
        statusDot.style.boxShadow = "0 0 8px var(--danger)";
        statusText.innerText = "Offline";
    }
    
    modeText.innerText = `Mode: ${statusInfo.mode || "Unknown"} | DB: ${statusInfo.database || "Unknown"}`;
}
