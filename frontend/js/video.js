/**
 * VISTA UI - Video Evidence Module
 * Manages CCTV clip player modal popups and video frame canvas rendering.
 * Explicitly labels mock previews to prevent user confusion with real feeds.
 */

export function openVideoModal(info) {
    const modal = document.getElementById("video-modal");
    if (!modal) return;

    document.getElementById("modal-camera-id").innerText = info.cameraId || "CAM_01";
    document.getElementById("modal-timestamp").innerText = `${info.timestamp || '13:14:15'} UTC`;
    document.getElementById("modal-track-id").innerText = info.trackId || "TRK_101";
    document.getElementById("modal-video-desc").innerText = info.description || "CCTV Evidence Clip";

    // Draw canvas bounding box preview
    const canvas = document.getElementById("video-canvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#090d16";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // CCTV Grid Lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        // Bounding Box
        ctx.strokeStyle = "#06b6d4";
        ctx.lineWidth = 3;
        ctx.strokeRect(180, 45, 240, 160);

        // Bounding Box Label
        ctx.fillStyle = "#06b6d4";
        ctx.fillRect(180, 20, 160, 25);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px monospace";
        ctx.fillText(`${info.trackId || 'TRK_101'} [EVIDENCE]`, 190, 36);

        // Explicit Mock Banner in Canvas
        ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
        ctx.fillRect(10, 10, 180, 22);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px sans-serif";
        ctx.fillText("MOCK CCTV EVIDENCE PREVIEW", 18, 25);
    }

    modal.classList.add("open");
}
