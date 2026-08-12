import pytest
from uuid import uuid4

from app.domain.evidence import EvidenceBundle, PersonEvidence
from app.agents.reasoning.engine.alias_map import EvidenceAliasMap

from datetime import datetime

def test_alias_resolution_success():
    uuid1 = uuid4()
    uuid2 = uuid4()
    bundle = EvidenceBundle(
        evidence=[
            PersonEvidence(evidence_id=uuid1, confidence=0.9, source="test", timestamp=datetime.now()),
            PersonEvidence(evidence_id=uuid2, confidence=0.8, source="test", timestamp=datetime.now()),
        ]
    )
    
    alias_map = EvidenceAliasMap(bundle)
    
    assert alias_map.resolve_alias("E1") == str(uuid1)
    assert alias_map.resolve_alias("E2") == str(uuid2)

def test_alias_resolution_failure_invalid_alias():
    uuid1 = uuid4()
    bundle = EvidenceBundle(evidence=[PersonEvidence(evidence_id=uuid1, confidence=0.9, source="test", timestamp=datetime.now())])
    alias_map = EvidenceAliasMap(bundle)
    
    with pytest.raises(ValueError, match="Unknown evidence alias: E999"):
        alias_map.resolve_alias("E999")

def test_alias_resolution_failure_direct_uuid():
    uuid1 = uuid4()
    bundle = EvidenceBundle(evidence=[PersonEvidence(evidence_id=uuid1, confidence=0.9, source="test", timestamp=datetime.now())])
    alias_map = EvidenceAliasMap(bundle)
    
    with pytest.raises(ValueError, match="Unknown evidence alias:"):
        alias_map.resolve_alias(str(uuid1))

def test_empty_bundle():
    alias_map = EvidenceAliasMap(None)
    with pytest.raises(ValueError):
        alias_map.resolve_alias("E1")
    assert alias_map.to_llm_context_string() == "No authorized evidence available."
