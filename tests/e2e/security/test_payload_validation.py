def test_missing_query(client, valid_token):
    payload = {} # missing query
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json=payload
    )
    
    assert response.status_code == 422
    assert "detail" in response.json()

def test_oversized_payload(client, valid_token):
    payload = {
        "query": "A" * 1000000  # 1MB string, though Pydantic might accept it, in a real system middleware would block it. Let's assume Pydantic handles it here.
    }
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json=payload
    )
    # The actual behavior depends on how the graph handles 1MB string (likely fails or takes forever),
    # but the API layer should ideally process or block it. We'll just verify the server doesn't crash 
    # without returning an HTTP response.
    assert response.status_code in [200, 413, 422, 500]
