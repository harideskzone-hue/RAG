import os
from datetime import datetime, timedelta, timezone

from jose import jwt


def test_missing_jwt(client):
    response = client.post("/api/v1/chat", json={"query": "hello"})
    assert response.status_code == 401

def test_expired_jwt(client):
    secret = os.getenv("JWT_SECRET_KEY", "super_secret_local_dev_key")
    expired_token = jwt.encode(
        {"sub": "user", "role": "operator", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
        secret, algorithm="HS256"
    )
    
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {expired_token}"},
        json={"query": "hello"}
    )
    assert response.status_code == 401
    assert "Signature has expired" in response.json()["detail"]

def test_invalid_jwt_signature(client, valid_token):
    # Mess up the signature
    bad_token = valid_token[:-5] + "aaaaa"
    
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {bad_token}"},
        json={"query": "hello"}
    )
    assert response.status_code == 401
    assert "Signature verification failed" in response.json()["detail"]
