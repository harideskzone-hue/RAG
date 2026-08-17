from unittest.mock import patch

from app.platform.errors.service import ServiceError


def test_vlm_timeout(client, valid_token):
    """
    Scenario: VLM API (e.g. Gemini) times out.
    """
    # Mock the LLM client returned by the registry to raise a ServiceError on generation
    from unittest.mock import Mock, AsyncMock
    mock_client = Mock()
    mock_client.generate_structured = AsyncMock(side_effect=ServiceError("VLM Timeout"))
    
    with patch("app.infrastructure.llm.model_registry.ModelRegistry.get_client", return_value=mock_client):
        payload = {
            "query": "Find the person in a blue hat." # Triggers Vector + Video
        }
        
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {valid_token}"},
            json=payload
        )
        
        # If it bubbles up to the supervisor as an unhandled error but gets caught:
        assert response.status_code == 200
        assert "answer" in response.json()
