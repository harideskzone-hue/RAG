import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_planner_failure():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("planner_fail")
    
    assert "Planner -> Exception" in pipeline.state_transitions
    assert "Supervisor Recovery" in pipeline.state_transitions
