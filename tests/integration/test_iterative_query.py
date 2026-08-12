import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_iterative_query():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("iterative")
    
    # Verify iterative state transitions
    assert "Iteration Count = 2" in pipeline.state_transitions
