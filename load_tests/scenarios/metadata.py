import random

from locust import HttpUser, between, task


class MetadataScenario(HttpUser):
    """
    Simulates high-volume, low-latency requests (Metadata queries).
    Represents 70% of the overall traffic.
    """
    wait_time = between(1, 3)
    weight = 70
    
    @task
    def query_camera_status(self):
        camera_id = random.randint(1, 100)
        payload = {
            "query": f"Is camera {camera_id} online?",
            "conversation_id": f"locust-metadata-{random.randint(1000, 9999)}"
        }
        
        # In a real setup, we would need to pass Auth headers
        # Assuming the load test runs against an environment where auth is mocked or we use a valid token
        with self.client.post("/api/v1/chat", json=payload, headers={"Authorization": "Bearer TEST_TOKEN"}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")
