import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_memory_empty():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("memory_empty")
    
    assert "No Entity" in pipeline.state_transitions
    assert "Reasoning Continues" in pipeline.state_transitions
