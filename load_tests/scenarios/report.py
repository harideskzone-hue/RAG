import random

from locust import HttpUser, between, task


class ReportScenario(HttpUser):
    """
    Simulates high-latency reporting workloads.
    Represents 5% of traffic.
    """
    wait_time = between(10, 20)
    weight = 5
    
    @task
    def generate_report(self):
        payload = {
            "query": "Generate a weekly security summary report for the main lobby.",
            "conversation_id": f"locust-report-{random.randint(1000, 9999)}"
        }
        
        with self.client.post("/api/v1/chat", json=payload, headers={"Authorization": "Bearer TEST_TOKEN"}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")
