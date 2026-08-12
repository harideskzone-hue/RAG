import pytest
from app.domain.integration.manifest import IntegrationManifest
from tests.integration.framework import MockAgenticPipeline

def test_policy_reject():
    manifest = IntegrationManifest()
    pipeline = MockAgenticPipeline(manifest)
    
    results = pipeline.run("policy_reject")
    
    assert "Policy Reject" in pipeline.state_transitions
    assert "Abort" in pipeline.state_transitions
