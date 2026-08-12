def test_viewer_access_denied_reports(client, viewer_token):
    # Viewers can't generate reports (write:report)
    payload = {
        "query": "Generate report",
        "time_range_hours": 24
    }
    response = client.post(
        "/api/v1/reports",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json=payload
    )
    
    assert response.status_code == 403
    assert "permission 'write:report'" in response.json()["detail"]

def test_viewer_access_allowed_chat(client, viewer_token):
    # Viewers can use chat (read:chat)
    payload = {
        "query": "Is camera 5 online?"
    }
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json=payload
    )
    
    assert response.status_code == 200
