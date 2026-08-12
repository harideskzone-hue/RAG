from locust import HttpUser

# Locust will automatically discover these classes when run
# locust -f load_tests/locustfile.py

class VistaAILoadTest(HttpUser):
    """
    Main entry point for Locust.
    We delegate execution to the scenario classes which have their own weights.
    Locust distributes users based on the 'weight' attribute of each class.
    
    To run:
    locust -f load_tests/locustfile.py --host=http://localhost:8000
    """
    # This class is just here to provide documentation and show it imports the others.
    # Actually, Locust will pick up all HttpUser subclasses in the file.
