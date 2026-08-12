const BASE_URL = "http://localhost:8000/api/v1";

/**
 * Retrieve the auth token.
 * In development, a token can be set via the browser console:
 *   sessionStorage.setItem("vista_token", "<your_jwt>");
 * In production, this should come from a proper login/OAuth flow.
 */
function getAuthToken() {
    const token = sessionStorage.getItem("vista_token");
    if (!token) {
        console.warn(
            "No auth token found. Set one with: " +
            'sessionStorage.setItem("vista_token", "<your_jwt>")'
        );
    }
    return token;
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

export async function sendChatQuery(query) {
    const token = getAuthToken();
    if (!token) {
        throw new Error(
            "Authentication required. Set a token with: " +
            'sessionStorage.setItem("vista_token", "<your_jwt>")'
        );
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
                conversation_id: "dashboard-session"
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "API Error");
        }
        
        return await res.json();
    } catch (e) {
        console.error("Chat Error:", e);
        throw e;
    }
}

