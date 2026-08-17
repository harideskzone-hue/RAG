const BASE_URL = "/api/v1";

/**
 * Development authentication check.
 * Strictly local dev helper. In production, dev fallback is disabled.
 */
export function getDevAuthToken() {
    const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    
    if (isLocalhost) {
        const validToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkZXZfdXNlciIsInJvbGUiOiJhZG1pbiIsImFsbG93ZWRfY2FtZXJhcyI6WyJjYW1fMDEiLCJDQU1fMDIiLCJDQU1fMDMiXSwiZXhwIjoxNzg2NjgyNzE4fQ.Phn7_Z7lTtVxZ8L1hmzN6JvpOY-MYgzuUuLnqDaWYFg";
        let token = sessionStorage.getItem("vista_dev_token");
        if (!token || token.includes("dev_sig")) {
            token = validToken;
            sessionStorage.setItem("vista_dev_token", token);
        }
        return { token, isDevMode: true };
    }
    
    const userToken = sessionStorage.getItem("vista_token");
    return { token: userToken, isDevMode: false };
}

export async function checkHealth() {
    try {
        const res = await fetch(`${BASE_URL}/health`);
        if (!res.ok) throw new Error("Health check failed");
        return await res.json();
    } catch (e) {
        console.error("API Health Error:", e);
        return { status: "error", mode: "offline" };
    }
}

export async function sendChatQuery(query, executionMode = "simple") {
    const { token, isDevMode } = getDevAuthToken();
    if (!token) {
        throw new Error("Authentication required. Please log in with a valid JWT token.");
    }

    try {
        const res = await fetch(`${BASE_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                query: query,
                conversation_id: "dashboard-session",
                execution_mode: executionMode
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `API Error (${res.status})`);
        }
        
        return await res.json();
    } catch (e) {
        console.error("Chat API Error:", e);
        throw e;
    }
}
