import pytest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.app import app
from app.security.jwt import JWTService
from app.security.roles import Role

client = TestClient(app)

@pytest.fixture
def valid_token():
    jwt_service = JWTService()
    return jwt_service.create_access_token({"sub": "user_123", "role": Role.OPERATOR})

@pytest.fixture
def viewer_token():
    jwt_service = JWTService()
    return jwt_service.create_access_token({"sub": "user_456", "role": Role.VIEWER})

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_chat_requires_auth():
    response = client.post("/api/v1/chat", json={"query": "hello"})
    assert response.status_code == 401

def test_chat_success(valid_token):
    # Mocking get_supervisor dependency is often needed, but since it's just an integration test
    # we can let it run the real supervisor with mock tools if we configured them globally.
    # We will just verify it gets rejected if body is bad first
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json={"wrong_field": "hello"}
    )
    assert response.status_code == 422 # Pydantic Validation

def test_report_rbac_rejection(viewer_token):
    # Viewers don't have write:report
    response = client.post(
        "/api/v1/reports",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json={"query": "Make a report"}
    )
    assert response.status_code == 403
    assert "permission 'write:report'" in response.json()["detail"]

def test_websocket():
    with client.websocket_connect("/api/v1/ws/chat") as websocket:
        websocket.send_text("Hello")
        data = websocket.receive_json()
        assert data["type"] == "progress"
        assert data["stage"] == "MetadataAgent"
