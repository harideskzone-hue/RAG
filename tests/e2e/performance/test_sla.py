import time


def test_sla_camera_status(client, valid_token):
    start_time = time.time()
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json={"query": "Is camera 5 online?"}
    )
    duration = time.time() - start_time
    assert response.status_code == 200
    
    # SLA Target: < 500ms
    # Note: Since the graph executes with mocks locally, this should be well under 500ms.
    assert duration < 0.5, f"Camera status SLA violated: {duration}s"

def test_sla_person_search(client, valid_token):
    start_time = time.time()
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {valid_token}"},
        json={"query": "Find the person wearing a red jacket."}
    )
    duration = time.time() - start_time
    assert response.status_code == 200
    
    # SLA Target: < 60s (Local LLM reasoning execution takes ~30s)
    assert duration < 60.0, f"Person search SLA violated: {duration}s"
