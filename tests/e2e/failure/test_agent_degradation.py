from unittest.mock import patch

from app.platform.errors.service import ServiceError


def test_vlm_timeout(client, valid_token):
    """
    Scenario: VLM API (e.g. Gemini) times out.
    """
    # Assuming GeminiAdapter raises ServiceError on timeout
    with patch("app.services.video_service.vlm_adapter.GeminiAdapter.analyze", side_effect=ServiceError("VLM Timeout")):
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
