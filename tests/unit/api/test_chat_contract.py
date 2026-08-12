import pytest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.app import app
client = TestClient(app)

def test_chat_contract_success(monkeypatch):
    from app.api.dependencies.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"sub": "1", "role": "admin", "allowed_cameras": ["CAM_02"]}
    
    response = client.post("/api/v1/chat", json={"query": "find person"}, headers={"Authorization": f"Bearer dummy"})
    app.dependency_overrides.clear()
    
    # Check if REASONING_BLOCKED is returned because model is not configured properly in CI,
    # or if we get a standard answer.
    assert response.status_code == 200
    data = response.json()
    
    # Check required schema fields
    assert "status" in data
    assert "answer" in data
    assert "evidence" in data
    assert "citations" in data
    assert "confidence" in data
    assert "trace_id" in data
    
    # If reasoning is blocked due to missing API keys
    if data["status"] == "REASONING_BLOCKED":
        assert data["answer"] == "Reasoning model unavailable."
        
def test_chat_contract_no_authorized_evidence(monkeypatch):
    from app.api.dependencies.security import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"sub": "1", "role": "admin", "allowed_cameras": ["CAM_XYZ"]}
    
    response = client.post("/api/v1/chat", json={"query": "find person"}, headers={"Authorization": f"Bearer dummy"})
    app.dependency_overrides.clear()
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "NO_AUTHORIZED_EVIDENCE"
    assert "No authorized evidence" in data["answer"]
    assert len(data["evidence"]) == 0
