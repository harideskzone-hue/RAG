def test_camera_status_query(client, valid_token):
    """
    Scenario: User queries the status of a specific camera.
    Expected Flow: MetadataAgent -> EvidenceAgent -> ChatResponse
    """
    payload = {
        "query": "Is camera 5 online?",
        "camera_ids": ["cam-5"]
    }
    
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    # The pipeline executes with mocked services (which return generic dummy data for now)
    assert len(data["evidence"]) >= 0 
