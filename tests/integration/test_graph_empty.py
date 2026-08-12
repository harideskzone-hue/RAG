import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_graph_empty():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("graph_empty")
    
    assert "No Relationships" in pipeline.state_transitions
    assert "Gap Analyzer" in pipeline.state_transitions
    assert "Video Requested" in pipeline.state_transitions
