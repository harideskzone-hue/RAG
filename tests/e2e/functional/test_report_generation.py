def test_report_generation(client, admin_token):
    """
    Scenario: User generates a weekly report (async background task via /reports).
    """
    payload = {
        "query": "Generate weekly incident report.",
        "time_range_hours": 168
    }
    
    response = client.post(
        "/api/v1/reports",
        headers={"Authorization": f"Bearer {admin_token}"},
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
