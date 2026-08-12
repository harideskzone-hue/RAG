import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_memory_query():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("memory")
    
    assert "Episode Updated" in pipeline.state_transitions
