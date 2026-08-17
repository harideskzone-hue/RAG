import pytest
from fastapi.testclient import TestClient
from app.app import create_app
from app.api.dependencies.security import get_current_user

app = create_app()

def override_get_current_user():
    return {"sub": "test_user_1", "role": "admin"}

app.dependency_overrides[get_current_user] = override_get_current_user
client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_live_api_end_to_end_investigation():
    """
    Live API E2E:
    1. How many people are in the video?
    2. What happened around 30 seconds?
    """
    
    # Test Count Query
    response = client.post(
        "/api/v1/chat",
        json={
            "query": "How many people are in the video?",
            "video_id": "vid_mock_1"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "evidence" in data
    
    # The pipeline should have executed end-to-end
    assert data["status"] in ("success", "NO_AUTHORIZED_EVIDENCE")
    
    # Test Temporal Event Query
    response = client.post(
        "/api/v1/chat",
        json={
            "query": "What happened around 30 seconds?",
            "video_id": "vid_mock_1"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["status"] in ("success", "NO_AUTHORIZED_EVIDENCE")
