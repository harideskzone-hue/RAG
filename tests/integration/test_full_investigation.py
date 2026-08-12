import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_full_investigation():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("full_investigation")
    
    expected_transitions = [
        "Execution Plan Created",
        "Plan Modified",
        "Iteration Count = 2",
        "Entity Count = 14",
        "Episode Updated",
        "Hypothesis Generated",
        "Score Produced"
    ]
    
    assert pipeline.state_transitions == expected_transitions
