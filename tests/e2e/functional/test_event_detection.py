def test_event_detection(client, valid_token):
    """
    Scenario: User asks about an event.
    Expected Flow: EventAgent -> EvidenceAgent
    """
    payload = {
        "query": "Was there a fight in the lobby?",
    }
    
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
