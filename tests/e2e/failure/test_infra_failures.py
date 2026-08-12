from unittest.mock import patch

from app.platform.errors.tool import ToolError


def test_milvus_unavailable(client, valid_token):
    """
    Scenario: The Milvus database is down when trying to search for a person.
    Expected: Supervisor catches ToolError and returns graceful failure response or omits vector context.
    """
    # Mocking the milvus tool execute method to simulate unavailability
    with patch("app.tools.vector.milvus_tool.MilvusTool.execute", side_effect=ToolError("Connection timeout to Milvus.")):
        payload = {
            "query": "Find the person in a blue hat."
        }
        
        # In a resilient pipeline, if one tool fails, the agent might return a partial result,
        # or if it's critical, the workflow catches it and gracefully reports it to the user.
        # Since our supervisor wraps agents safely, the chat endpoint shouldn't return 500,
        # it might return a 200 with an answer explaining the failure or partial results.
        
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {valid_token}"},
            json=payload
        )
        
        # It's an internal workflow error if the core pipeline fails and bubbles up.
        # But in a resilient system, the workflow degraded gracefully and returned a 200.
        assert response.status_code == 200
        assert "answer" in response.json()
