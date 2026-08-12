import random

from locust import HttpUser, between, task


class PersonSearchScenario(HttpUser):
    """
    Simulates medium-latency requests requiring Vector Search + Metadata.
    Represents 20% of traffic.
    """
    wait_time = between(2, 5)
    weight = 20
    
    @task
    def search_person(self):
        colors = ["red", "blue", "black", "white"]
        items = ["backpack", "hat", "jacket", "glasses"]
        
        payload = {
            "query": f"Find the person wearing a {random.choice(colors)} {random.choice(items)}.",
            "conversation_id": f"locust-person-{random.randint(1000, 9999)}"
        }
        
        with self.client.post("/api/v1/chat", json=payload, headers={"Authorization": "Bearer TEST_TOKEN"}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")
