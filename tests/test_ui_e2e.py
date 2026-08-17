import pytest
from playwright.async_api import async_playwright, expect
import asyncio
import subprocess
import os

@pytest.mark.asyncio
async def test_ui_e2e_integration():
    """
    E2E UI Test:
    1. Starts the Vite dev server.
    2. Opens the browser.
    3. Mocks the FastAPI /api/v1/chat response to return a VALID grounded response.
    4. Simulates a user query.
    5. Verifies the UI correctly renders the Thinking status, Streaming Text, Tool Chips, and Evidence Panel.
    """
    # Start the Vite server
    env = os.environ.copy()
    process = subprocess.Popen(
        ["npm", "run", "dev"], 
        cwd="frontend",
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    # Wait for vite to start
    await asyncio.sleep(2)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Mock the FastAPI endpoint
        async def handle_chat_route(route):
            mock_response = {
                "status": "SUCCESS",
                "answer": "There are 5 people tracked by CAM_03.",
                "grounding_status": "VALID",
                "confidence": 0.95,
                "citations": [],
                "evidence": [
                    {
                        "evidence_id": "e_001",
                        "source": "vector_db",
                        "camera_id": "CAM_03",
                        "timestamp": "2026-08-15T12:00:00Z",
                        "description": "Person tracked",
                        "confidence": 0.99
                    }
                ],
                "timeline": [{"timestamp": "12:00:00Z", "event": "appeared"}],
                "processing": {},
                "execution": {
                    "status": "completed",
                    "steps": [
                        {"name": "Query Understanding", "status": "completed", "latency_ms": 10, "error": None},
                        {"name": "Evidence Retrieval", "status": "completed", "latency_ms": 50, "error": None},
                        {"name": "Answer Generation", "status": "completed", "latency_ms": 100, "error": None}
                    ]
                },
                "processing_time_ms": 160,
                "trace_id": "trace_mock"
            }
            await route.fulfill(json=mock_response, status=200)

        await page.route("**/api/v1/chat", handle_chat_route)
        
        # Navigate to the frontend
        await page.goto("http://localhost:5173")
        
        # Wait for the layout to load
        await expect(page.locator("text=VISTA AI").first).to_be_visible(timeout=5000)
        
        # Ensure evidence panel is empty initially
        await expect(page.locator("text=Select an AI response to view authoritative evidence")).to_be_visible()

        # Type a query
        await page.fill("input.chat-input", "How many people on CAM_03?")
        
        # Submit
        await page.click("button.chat-submit")
        
        # Wait for the mock response to be processed
        # 1. Thinking status should be rendered
        await expect(page.locator("text=Query Understanding")).to_be_visible()
        await expect(page.locator("text=Evidence Retrieval")).to_be_visible()
        await expect(page.locator("text=Answer Generation")).to_be_visible()
        
        # 2. Tool Chips should be visible
        await expect(page.locator("text=1 Observations")).to_be_visible()
        await expect(page.locator("text=Timeline Generated")).to_be_visible()
        
        # 3. Answer should be streamed/visible
        await expect(page.locator("text=There are 5 people tracked by CAM_03.")).to_be_visible()
        
        # 4. Evidence Panel should automatically update
        await expect(page.locator("text=GROUNDED")).to_be_visible()
        await expect(page.locator("text=CAM_03").first).to_be_visible()
        await expect(page.locator("text=2026-08-15T12:00:00Z")).to_be_visible()

        await browser.close()
        
    process.terminate()
