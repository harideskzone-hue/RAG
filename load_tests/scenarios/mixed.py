import random

from locust import HttpUser, between, task


class VideoReasoningScenario(HttpUser):
    """
    Simulates high-latency complex reasoning (Video analysis).
    Represents 5% of traffic.
    """
    wait_time = between(5, 10)
    weight = 5
    
    @task
    def complex_event_search(self):
        actions = ["fighting", "running fast", "falling down", "loitering"]
        
        payload = {
            "query": f"Is anyone {random.choice(actions)} near the north gate?",
            "conversation_id": f"locust-video-{random.randint(1000, 9999)}"
        }
        
        with self.client.post("/api/v1/chat", json=payload, headers={"Authorization": "Bearer TEST_TOKEN"}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")
