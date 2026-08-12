import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_simple_query():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("simple")
    
    # Verify outputs
    assert "planner" in results
    assert "supervisor" in results
    assert "graph" in results
    
    # Verify state transitions
    assert "Execution Plan Created" in pipeline.state_transitions
    assert "Plan Modified" in pipeline.state_transitions
    assert "Iteration Count = 2" in pipeline.state_transitions
    assert "Entity Count = 14" in pipeline.state_transitions
    assert "Episode Updated" in pipeline.state_transitions
    assert "Hypothesis Generated" in pipeline.state_transitions
    assert "Score Produced" in pipeline.state_transitions
