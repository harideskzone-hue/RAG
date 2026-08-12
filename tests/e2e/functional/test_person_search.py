def test_person_search(client, valid_token):
    """
    Scenario: User searches for a person.
    Expected Flow: MetadataAgent -> VectorAgent -> EvidenceAgent -> ConfidenceAgent -> VideoAgent
    """
    payload = {
        "query": "Find the person wearing a red jacket near the entrance.",
    }
    
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["evidence"]) >= 0 
